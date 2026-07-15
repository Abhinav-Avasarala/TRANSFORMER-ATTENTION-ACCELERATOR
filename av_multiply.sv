// av_multiply.sv
//
// Computes O = A * V  (the "blend" stage of attention).
//   A : [N x N]  Attention weights (softmax output). Each row sums to ~1.0.
//                Fed ROW-wise, exactly like Q in qk_systolic -- no transpose
//                involved, A is used as-is. Format is Q1.A_FRAC_BITS
//                (softmax_unit emits Q1.15), NOT the Q8.8 used elsewhere:
//                weights are non-negative and < 1, clamped to 0x7FFF so
//                mac_pe's signed multiply reads them correctly as positive.
//   V : [N x D]  Value matrix, one row per token, signed Q8.8.
//                Fed COLUMN-wise (NOT row-wise like K was). O = A*V is a
//                plain matmul, so there's no transpose trick available here
//                -- column c of V is streamed into PE-column c over time.
//   O : [N x D]  O[i][c] = sum_j A[i][j] * V[j][c], in Q8.8. Since A and V
//                have different fractional-bit counts, the output truncation
//                drops A_FRAC_BITS bits (not FRAC_BITS) -- see below.
//
// Same output-stationary systolic architecture as qk_systolic.sv, but the
// CONTRACTION (summed) dimension here is N (sequence length), not D (head
// dim) -- A is N x N and V is N x D, so the shared/summed index is the
// N-length one. That changes the feed skew, the PE grid shape (N x D
// instead of N x N), and where the "+2" latency term lands, even though
// the final LAT formula is textually identical to qk_systolic's.
//
// SKEW CONTRACT (mirrors qk_systolic's, but note v_in's index is different):
//   The CALLER feeds skewed, zero-padded inputs every cycle:
//     at feed-cycle t:  a_in[i] = A[i][t-i]     (zero if t-i out of [0,N-1])
//                       v_in[c] = V[t-c][c]     (zero if t-c out of [0,N-1])
//   Pulse `start` for 1 cycle, then begin feeding on the NEXT cycle.
//   `done` pulses for 1 cycle when o_out is fully valid.

`timescale 1ns/1ps

module av_multiply #(
    parameter int N          = 64,   // sequence length: rows of A/V/O, PE rows, contraction length
    parameter int D          = 64,   // head dimension: columns of V/O, PE columns
    parameter int DATA_WIDTH = 16,   // Q8.8 element width
    parameter int ACC_WIDTH  = 32,   // 16x16 product + accumulation headroom
    parameter int FRAC_BITS  = 8,    // V and O fractional bits (Q8.8)
    parameter int A_FRAC_BITS = 8    // A (attention-weight) fractional bits.
                                      // Defaults to FRAC_BITS (symmetric Q8.8);
                                      // softmax_unit now emits A as Q1.15, so
                                      // top_fsm instantiates this with A_FRAC_BITS=15.
)(
    input  logic                  clk,
    input  logic                  rst_n,         // active-low, synchronous
    input  logic                  start,         // 1-cycle pulse to begin

    input  logic [DATA_WIDTH-1:0] a_in [0:N-1],  // skewed A feed (per row), Q1.A_FRAC_BITS,
                                                  // non-negative (mac_pe reads it signed)
    input  logic [DATA_WIDTH-1:0] v_in [0:D-1],  // skewed V feed (per column), signed Q8.8

    output logic [DATA_WIDTH-1:0] o_out [0:N-1][0:D-1],  // context output, Q8.8
    output logic                  done
);

    // -------------------------------------------------------------------------
    // Latency: PE[i][c] folds its last input (local j=N-1) into sum_out at
    // cycle i + c + N + 2 (same derivation as qk_systolic, contraction
    // length is N here instead of D). Slowest PE is [N-1][D-1], so all
    // results are valid at:
    //   cnt = (N-1) + (D-1) + N + 2 = 2*N + D
    localparam int LAT       = 2*N + D;
    localparam int CNT_WIDTH = $clog2(LAT + 1);

    // -------------------------------------------------------------------------
    // Control: identical shape to qk_systolic -- start -> busy, count to LAT,
    // capture + done, idle. While idle (!busy) the datapath is held cleared.
    // -------------------------------------------------------------------------
    logic                  busy;
    logic [CNT_WIDTH-1:0]  cnt;
    logic                  capture;   // high the cycle results are valid

    assign capture = busy && (cnt == CNT_WIDTH'(LAT));

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0;
            cnt  <= '0;
            done <= 1'b0;
        end else begin
            done <= capture;                  // done pulses 1 cycle after capture
            if (start) begin
                busy <= 1'b1;
                cnt  <= '0;
            end else if (busy) begin
                if (capture) busy <= 1'b0;
                else         cnt  <= cnt + 1'b1;
            end
        end
    end

    // Synchronous clear for accumulators and shift registers.
    logic clr;
    assign clr = !rst_n || !busy;

    // -------------------------------------------------------------------------
    // Edge wires + propagation shift registers. Grid is N rows x D columns
    // (not necessarily square, unlike qk_systolic's N x N).
    //   a_wire[i][c] = value entering PE[i][c] from the left (A, row-fed)
    //   b_wire[i][c] = value entering PE[i][c] from the top  (V, column-fed)
    // -------------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] a_wire [0:N-1][0:D];
    logic [DATA_WIDTH-1:0] b_wire [0:N][0:D-1];

    genvar i, c;
    generate
        for (i = 0; i < N; i++) begin : a_edge_connect
            assign a_wire[i][0] = a_in[i];   // left edge <- A feed
        end
        for (c = 0; c < D; c++) begin : b_edge_connect
            assign b_wire[0][c] = v_in[c];   // top edge  <- V feed
        end
    endgenerate

    generate
        for (i = 0; i < N; i++) begin : a_shift_row
            for (c = 0; c < D; c++) begin : a_shift_col
                always_ff @(posedge clk) begin
                    if (clr) a_wire[i][c+1] <= '0;
                    else     a_wire[i][c+1] <= a_wire[i][c];   // shift right
                end
            end
        end
    endgenerate

    generate
        for (c = 0; c < D; c++) begin : b_shift_col
            for (i = 0; i < N; i++) begin : b_shift_row
                always_ff @(posedge clk) begin
                    if (clr) b_wire[i+1][c] <= '0;
                    else     b_wire[i+1][c] <= b_wire[i][c];   // shift down
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // PE grid: PE[i][c] accumulates sum_j A[i][j] * V[j][c].
    // mac_pe's `reset` (active-high, sync) is driven by clr.
    // -------------------------------------------------------------------------
    logic [ACC_WIDTH-1:0] pe_out [0:N-1][0:D-1];

    generate
        for (i = 0; i < N; i++) begin : pe_row
            for (c = 0; c < D; c++) begin : pe_col
                mac_pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH (ACC_WIDTH)
                ) pe_inst (
                    .clk      (clk),
                    .reset    (clr),
                    .valid_in (1'b1),
                    .a_in     (a_wire[i][c]),
                    .b_in     (b_wire[i][c]),
                    .sum_out  (pe_out[i][c]),
                    .valid_out()
                );
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Capture + truncate to Q8.8 when results are valid. Each PE accumulates
    // A(Q1.A_FRAC_BITS) * V(Q8.FRAC_BITS), so pe_out has (A_FRAC_BITS +
    // FRAC_BITS) fractional bits. To emit O as Q8.FRAC_BITS, drop A_FRAC_BITS
    // frac bits -> bit-select starts at A_FRAC_BITS (= FRAC_BITS in the
    // symmetric Q8.8 default; = 15 when A is Q1.15). No rounding/saturation,
    // same caveat as qk_systolic's header.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (capture) begin
            for (int r = 0; r < N; r++)
                for (int col = 0; col < D; col++)
                    o_out[r][col] <= pe_out[r][col][A_FRAC_BITS +: DATA_WIDTH];
        end
    end

endmodule
