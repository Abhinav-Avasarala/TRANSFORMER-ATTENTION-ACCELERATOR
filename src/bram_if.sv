// bram_if.sv
//
// Memory front-end: the translation layer between a real serial external
// interface (one word per cycle, matching the project doc's port spec)
// and top_fsm's full-array interface (Q[N][D] etc, which only a testbench
// or another RTL module can present -- nothing outside a chip can expose
// 4096 parallel wires).
//
//   LOAD  : q_data/k_data/v_data + data_valid stream in, one element of
//           each matrix per cycle (row-major), for N*D cycles. Written
//           directly into the Q_out/K_out/V_out output registers, which
//           top_fsm reads as full arrays once load_done pulses.
//   DRAIN : once top_fsm finishes (external capture_o pulse) and O_in
//           holds the full result, this module copies it internally in
//           one cycle, then streams it back out via out_data/out_valid,
//           one element per cycle, for N*D cycles. out_done pulses when
//           the last word has been sent.
//
// SCOPE DECISIONS (deliberate simplifications vs. the doc's full spec):
//   - Direct-port only, no AXI4-Lite: there's no real host in this
//     project (simulation-only scope), so the doc's AXI4-Lite option is
//     unnecessary complexity. This is the doc's simpler alternative.
//   - No ping-pong buffering: the doc mentions double-buffering V so the
//     next load can overlap with the current compute pass. This design
//     is strictly sequential (load -> compute -> drain -> next load) --
//     fine for one pass at a time, would need revisiting for pipelined
//     back-to-back passes.
//   - Q_out/K_out/V_out are exposed as full N x D arrays, matching
//     top_fsm's existing interface exactly, so no changes to top_fsm are
//     needed. Be aware this is the same "golden functional model" caveat
//     flagged elsewhere in this project (e.g. qk_systolic's PE count):
//     real BRAM only has 1-2 read ports, not simultaneous access to all
//     N*D elements at once, so Q_out/K_out/V_out will synthesize as
//     distributed RAM/registers, not a literal BRAM primitive, despite
//     the module's name. The LOAD side (serial writes) and the DRAIN
//     side (serial reads of O_mem) are both realistic single-port-style
//     access patterns; it's specifically the "expose everything as a
//     full array for top_fsm" step that isn't literally how BRAM works.
//   - Two separate completion signals (load_done, out_done) instead of
//     one generic `done` like every other module here: this module has
//     two genuinely distinct jobs happening at different points in the
//     pipeline (before compute, after compute), not one request/response,
//     so collapsing them into a single `done` would lose information a
//     caller needs (which phase just finished).
//
// ADDRESSING ASSUMPTION: row/col are recovered from the flat address via
// bit-slicing (col = addr[COL_BITS-1:0], row = the remaining high bits),
// which is only valid because D is a power of two (D=64 in this
// project). A non-power-of-two D would need real divide/modulo instead.

`timescale 1ns/1ps

module bram_if #(
    parameter int N          = 64,
    parameter int D          = 64,
    parameter int DATA_WIDTH = 16
)(
    input  logic                  clk,
    input  logic                  rst_n,        // active-low, synchronous
    input  logic                  start,        // 1-cycle pulse: begin LOAD

    // ---- Load side: serial in, one element of each matrix per cycle ----
    input  logic [DATA_WIDTH-1:0] q_data,
    input  logic [DATA_WIDTH-1:0] k_data,
    input  logic [DATA_WIDTH-1:0] v_data,
    input  logic                  data_valid,

    output logic [DATA_WIDTH-1:0] Q_out [0:N-1][0:D-1],  // full arrays, feed top_fsm
    output logic [DATA_WIDTH-1:0] K_out [0:N-1][0:D-1],
    output logic [DATA_WIDTH-1:0] V_out [0:N-1][0:D-1],
    output logic                  load_done,    // 1-cycle pulse: Q_out/K_out/V_out valid

    // ---- Drain side: full array in, serial out ----
    input  logic                  capture_o,    // 1-cycle pulse: O_in is valid, begin DRAIN
    input  logic [DATA_WIDTH-1:0] O_in [0:N-1][0:D-1],   // from top_fsm.O

    output logic [DATA_WIDTH-1:0] out_data,
    output logic                  out_valid,
    output logic                  out_done      // 1-cycle pulse: last word sent
);

    localparam int TOTAL     = N * D;
    localparam int ADDR_WIDTH = $clog2(TOTAL);
    localparam int COL_BITS   = $clog2(D);

    logic [DATA_WIDTH-1:0] O_mem [0:N-1][0:D-1];

    typedef enum logic [1:0] {IDLE, LOADING, DRAINING} state_t;
    state_t state;

    logic [ADDR_WIDTH-1:0] addr;
    logic [ADDR_WIDTH-COL_BITS-1:0] row;
    logic [COL_BITS-1:0]            col;
    assign row = addr[ADDR_WIDTH-1:COL_BITS];
    assign col = addr[COL_BITS-1:0];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state     <= IDLE;
            addr      <= '0;
            load_done <= 1'b0;
            out_valid <= 1'b0;
            out_done  <= 1'b0;
        end else begin
            load_done <= 1'b0;
            out_done  <= 1'b0;

            case (state)
                IDLE: begin
                    out_valid <= 1'b0;
                    if (start) begin
                        addr  <= '0;
                        state <= LOADING;
                    end else if (capture_o) begin
                        for (int r = 0; r < N; r++)
                            for (int c = 0; c < D; c++)
                                O_mem[r][c] <= O_in[r][c];
                        addr  <= '0;
                        state <= DRAINING;
                    end
                end

                LOADING: begin
                    if (data_valid) begin
                        Q_out[row][col] <= q_data;
                        K_out[row][col] <= k_data;
                        V_out[row][col] <= v_data;
                        if (addr == ADDR_WIDTH'(TOTAL-1)) begin
                            state     <= IDLE;
                            load_done <= 1'b1;
                        end else begin
                            addr <= addr + 1'b1;
                        end
                    end
                end

                DRAINING: begin
                    out_data  <= O_mem[row][col];
                    out_valid <= 1'b1;
                    if (addr == ADDR_WIDTH'(TOTAL-1)) begin
                        state    <= IDLE;
                        out_done <= 1'b1;
                    end else begin
                        addr <= addr + 1'b1;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
