// top_fsm.sv
//
// Master sequencer: wires qk_systolic -> softmax_unit -> av_multiply into
// the full attention pipeline  O = softmax(Q*K^T) * V  and drives each
// stage's start/done handshake plus the skewed feeds each systolic stage
// expects.
//
// SCOPE NOTE: this module takes Q, K, V as full N x D arrays and produces
// a full N x D array O -- NOT the doc's literal serial streaming port
// interface (q_data[15:0]/data_valid/etc, section 3.2). That streaming
// interface is bram_if's job: it's the memory front-end that would sit
// in front of this module, loading Q/K/V from BRAM and exposing them as
// full arrays the way this module expects, then capturing O the same
// way. Deferred deliberately (see project decision to do simulation only,
// no bram_if built yet) -- this module is the compute-orchestration core,
// independent of how Q/K/V actually get loaded.
//
// Why three different feed conventions live inside one FSM:
//   - qk_systolic and av_multiply each want a SKEWED, ONE-ROW/COLUMN-PER-
//     CYCLE stream (see their own file headers for the exact skew
//     contract) -- this module generates that stream from the full
//     arrays it holds, exactly mirroring what tb_qk_systolic.sv's
//     run_qk() / tb_av_multiply.sv's run_av() do in simulation, except
//     as real synthesizable hardware instead of testbench code.
//   - softmax_unit takes a full array directly -- no feed generation
//     needed, s_in is just wired straight to qk_systolic's s_out.
//
// Pipeline sequencing (one state per phase, matching each submodule's
// own "pulse start for 1 cycle, then feed begins next cycle" contract):
//   IDLE -> QK_START -> QK_FEED -> QK_DRAIN
//        -> SM_START -> SM_WAIT
//        -> AV_START -> AV_FEED -> AV_DRAIN
//        -> ALL_DONE -> IDLE
//
// O and `done` follow the same convention as every other module in this
// project: O is only guaranteed valid the cycle `done` is high (it holds
// the previous run's stale value before that, same as s_out/a_out do
// mid-pipeline).

`timescale 1ns/1ps

module top_fsm #(
    parameter int N          = 64,   // sequence length
    parameter int D          = 64,   // head dimension
    parameter int DATA_WIDTH = 16,   // Q8.8 element width
    parameter int ACC_WIDTH  = 32,   // systolic accumulator width
    parameter int FRAC_BITS  = 8     // Q8.8 fractional bits
)(
    input  logic                  clk,
    input  logic                  rst_n,         // active-low, synchronous
    input  logic                  start,         // 1-cycle pulse to begin a full pass

    input  logic [DATA_WIDTH-1:0] Q [0:N-1][0:D-1],
    input  logic [DATA_WIDTH-1:0] K [0:N-1][0:D-1],
    input  logic [DATA_WIDTH-1:0] V [0:N-1][0:D-1],

    output logic [DATA_WIDTH-1:0] O [0:N-1][0:D-1],
    output logic                  done
);

    localparam int FEED_CYCLES_QK = N + D - 1;
    localparam int MAX_ND         = (N > D) ? N : D;
    localparam int FEED_CYCLES_AV = N + MAX_ND - 1;

    localparam int QK_CNT_WIDTH = $clog2(FEED_CYCLES_QK);
    localparam int AV_CNT_WIDTH = $clog2(FEED_CYCLES_AV);

    // -------------------------------------------------------------------------
    // Stage instances
    // -------------------------------------------------------------------------
    logic                  qk_start, qk_done;
    logic [DATA_WIDTH-1:0] q_in [0:N-1];
    logic [DATA_WIDTH-1:0] k_in [0:N-1];
    logic [DATA_WIDTH-1:0] s_out [0:N-1][0:N-1];

    qk_systolic #(
        .N(N), .D(D), .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) qk_inst (
        .clk(clk), .rst_n(rst_n), .start(qk_start),
        .q_in(q_in), .k_in(k_in), .s_out(s_out), .done(qk_done)
    );

    logic                  sm_start, sm_done;
    logic [DATA_WIDTH-1:0] a_out [0:N-1][0:N-1];

    softmax_unit #(
        .N(N), .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) sm_inst (
        .clk(clk), .rst_n(rst_n), .start(sm_start),
        .s_in(s_out), .a_out(a_out), .done(sm_done)
    );

    logic                  av_start, av_done;
    logic [DATA_WIDTH-1:0] a_in [0:N-1];
    logic [DATA_WIDTH-1:0] v_in [0:D-1];
    logic [DATA_WIDTH-1:0] o_out [0:N-1][0:D-1];

    av_multiply #(
        .N(N), .D(D), .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) av_inst (
        .clk(clk), .rst_n(rst_n), .start(av_start),
        .a_in(a_in), .v_in(v_in), .o_out(o_out), .done(av_done)
    );

    assign O = o_out;

    // -------------------------------------------------------------------------
    // FSM state
    // -------------------------------------------------------------------------
    typedef enum logic [3:0] {
        IDLE, QK_START, QK_FEED, QK_DRAIN,
        SM_START, SM_WAIT,
        AV_START, AV_FEED, AV_DRAIN,
        ALL_DONE
    } state_t;
    state_t state;

    logic [QK_CNT_WIDTH-1:0] t_qk;
    logic [AV_CNT_WIDTH-1:0] t_av;

    assign qk_start = (state == QK_START);
    assign sm_start = (state == SM_START);
    assign av_start = (state == AV_START);

    // -------------------------------------------------------------------------
    // QK feed generation: q_in[i] = Q[i][t_qk-i], k_in[j] = K[j][t_qk-j]
    // (identical skew formula to tb_qk_systolic.sv's run_qk() task).
    // -------------------------------------------------------------------------
    always_comb begin
        int k_idx;
        for (int i = 0; i < N; i++) begin
            k_idx = int'(t_qk) - i;
            if (state == QK_FEED && k_idx >= 0 && k_idx < D)
                q_in[i] = Q[i][k_idx];
            else
                q_in[i] = '0;
        end
        for (int j = 0; j < N; j++) begin
            k_idx = int'(t_qk) - j;
            if (state == QK_FEED && k_idx >= 0 && k_idx < D)
                k_in[j] = K[j][k_idx];
            else
                k_in[j] = '0;
        end
    end

    // -------------------------------------------------------------------------
    // AV feed generation: a_in[i] = A[i][t_av-i] (row-fed, from softmax's
    // a_out), v_in[c] = V[t_av-c][c] (column-fed -- see av_multiply.sv's
    // header for why V is fed by column, not row, unlike K in the QK stage).
    // Identical skew formulas to tb_av_multiply.sv's run_av() task.
    // -------------------------------------------------------------------------
    always_comb begin
        int k_idx;
        for (int i = 0; i < N; i++) begin
            k_idx = int'(t_av) - i;
            if (state == AV_FEED && k_idx >= 0 && k_idx < N)
                a_in[i] = a_out[i][k_idx];
            else
                a_in[i] = '0;
        end
        for (int c = 0; c < D; c++) begin
            k_idx = int'(t_av) - c;
            if (state == AV_FEED && k_idx >= 0 && k_idx < N)
                v_in[c] = V[k_idx][c];
            else
                v_in[c] = '0;
        end
    end

    // -------------------------------------------------------------------------
    // Sequencing
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            t_qk  <= '0;
            t_av  <= '0;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    if (start) state <= QK_START;
                end

                // start pulses this cycle (qk_start is combinational on
                // state==QK_START); qk_systolic begins accepting its feed
                // the cycle after, which is exactly when QK_FEED starts.
                QK_START: begin
                    t_qk  <= '0;
                    state <= QK_FEED;
                end

                QK_FEED: begin
                    if (t_qk == QK_CNT_WIDTH'(FEED_CYCLES_QK - 1))
                        state <= QK_DRAIN;
                    else
                        t_qk <= t_qk + 1'b1;
                end

                QK_DRAIN: begin
                    if (qk_done) state <= SM_START;
                end

                // softmax_unit reads s_in directly (no feed needed) -- it
                // already sees a stable s_out from qk_systolic by now.
                SM_START: begin
                    state <= SM_WAIT;
                end

                SM_WAIT: begin
                    if (sm_done) state <= AV_START;
                end

                AV_START: begin
                    t_av  <= '0;
                    state <= AV_FEED;
                end

                AV_FEED: begin
                    if (t_av == AV_CNT_WIDTH'(FEED_CYCLES_AV - 1))
                        state <= AV_DRAIN;
                    else
                        t_av <= t_av + 1'b1;
                end

                AV_DRAIN: begin
                    if (av_done) state <= ALL_DONE;
                end

                ALL_DONE: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
