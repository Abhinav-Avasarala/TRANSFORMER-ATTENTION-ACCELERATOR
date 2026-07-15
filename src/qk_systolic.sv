// qk_systolic.sv
//
// Computes S = Q * K^T  (the score stage of attention).
//   Q : [N x D]  Key fact: we feed Q row by row.
//   K : [N x D]  We feed K row by row too -- no explicit transpose needed,
//                because feeding K row j down column j already gives K^T.
//   S : [N x N]  S[i][j] = dot(Q row i, K row j), in Q8.8.
//
// Built from an N x N grid of mac_pe (output-stationary systolic array).
// This is your previous systolic_array, widened to 16/32-bit, plus:
//   1. Q8.8 output truncation (Q8.8 * Q8.8 = Q16.16 internally -> shift back).
//   2. start/done control so top_fsm can sequence it.
//
// SKEW CONTRACT (same as your previous testbench):
//   The CALLER feeds skewed, zero-padded inputs every cycle:
//     at feed-cycle t:  q_in[i] = Q[i][t-i]   (zero if t-i out of [0,D-1])
//                       k_in[j] = K[j][t-j]   (zero if t-j out of [0,D-1])
//   Pulse `start` for 1 cycle, then begin feeding on the NEXT cycle.
//   `done` pulses for 1 cycle when s_out is fully valid.

`timescale 1ns/1ps

module qk_systolic #(
    parameter int N          = 64,   // sequence length (rows of Q and K)
    parameter int D          = 64,   // head dimension (inner-product length)
    parameter int DATA_WIDTH = 16,   // Q8.8 element width
    parameter int ACC_WIDTH  = 32,   // 16x16 product + accumulation headroom
    parameter int FRAC_BITS  = 8     // Q8.8 fractional bits
)(
    input  logic                  clk,
    input  logic                  rst_n,         // active-low, synchronous
    input  logic                  start,         // 1-cycle pulse to begin

    input  logic [DATA_WIDTH-1:0] q_in [0:N-1],  // skewed Q feed (per row)
    input  logic [DATA_WIDTH-1:0] k_in [0:N-1],  // skewed K feed (per row)

    output logic [DATA_WIDTH-1:0] s_out [0:N-1][0:N-1],  // score matrix, Q8.8
    output logic                  done
);

    // -------------------------------------------------------------------------
    // Latency: PE[i][j] folds its last input (k=D-1) into sum_out at cycle
    // i + j + D + 2. The slowest PE is [N-1][N-1], so all results are valid at:
    //   cnt = 2(N-1) + D + 2 = 2*N + D
    // We capture at exactly that cycle (sum_out is valid DURING the cnt==LAT
    // cycle; the s_out register then latches it on the following edge).
    localparam int LAT       = 2*N + D;
    localparam int CNT_WIDTH = $clog2(LAT + 1);

    // -------------------------------------------------------------------------
    // Control: start -> busy, count up to LAT, capture + done, idle.
    // While idle (!busy) the whole datapath is held cleared.
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

    // Synchronous clear for accumulators and shift registers:
    // active during global reset OR whenever we are not running a multiply.
    logic clr;
    assign clr = !rst_n || !busy;

    // -------------------------------------------------------------------------
    // Edge wires + propagation shift registers (identical structure to your
    // previous array): Q flows right, K flows down.
    //   a_wire[i][j] = value entering PE[i][j] from the left
    //   b_wire[i][j] = value entering PE[i][j] from the top
    // -------------------------------------------------------------------------
    logic [DATA_WIDTH-1:0] a_wire [0:N-1][0:N];
    logic [DATA_WIDTH-1:0] b_wire [0:N][0:N-1];

    genvar i, j;
    generate
        for (i = 0; i < N; i++) begin : edge_connect
            assign a_wire[i][0] = q_in[i];   // left edge  <- Q feed
            assign b_wire[0][i] = k_in[i];   // top edge   <- K feed
        end
    endgenerate

    generate
        for (i = 0; i < N; i++) begin : a_shift_row
            for (j = 0; j < N; j++) begin : a_shift_col
                always_ff @(posedge clk) begin
                    if (clr) a_wire[i][j+1] <= '0;
                    else     a_wire[i][j+1] <= a_wire[i][j];   // shift right
                end
            end
        end
    endgenerate

    generate
        for (j = 0; j < N; j++) begin : b_shift_col
            for (i = 0; i < N; i++) begin : b_shift_row
                always_ff @(posedge clk) begin
                    if (clr) b_wire[i+1][j] <= '0;
                    else     b_wire[i+1][j] <= b_wire[i][j];   // shift down
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // PE grid: PE[i][j] accumulates dot(Q row i, K row j).
    // mac_pe's `reset` (active-high, sync) is driven by clr.
    // -------------------------------------------------------------------------
    logic [ACC_WIDTH-1:0] pe_out [0:N-1][0:N-1];

    generate
        for (i = 0; i < N; i++) begin : pe_row
            for (j = 0; j < N; j++) begin : pe_col
                mac_pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH (ACC_WIDTH)
                ) pe_inst (
                    .clk      (clk),
                    .reset    (clr),
                    .valid_in (1'b1),
                    .a_in     (a_wire[i][j]),
                    .b_in     (b_wire[i][j]),
                    .sum_out  (pe_out[i][j]),
                    .valid_out()
                );
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Capture + truncate to Q8.8 when results are valid.
    // Product of two Q8.8 numbers has 16 fractional bits (Q16.16);
    // take bits [FRAC_BITS +: DATA_WIDTH] to shift the point back to Q8.8.
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (capture) begin
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    s_out[r][c] <= pe_out[r][c][FRAC_BITS +: DATA_WIDTH];
        end
    end

endmodule