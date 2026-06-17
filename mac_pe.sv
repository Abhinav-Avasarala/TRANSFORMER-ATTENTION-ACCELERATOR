// mac_pe.sv
//
// 3-stage pipelined multiply-accumulate unit.
// Computes:  acc += a_in * b_in  every cycle while valid_in is high.
// Reset clears the accumulator and pipeline registers.
//
// Pipeline stages:
//   Stage 1 (clk edge 1): latch a_in, b_in into reg_a, reg_b
//   Stage 2 (clk edge 2): compute reg_product = reg_a * reg_b
//   Stage 3 (clk edge 3): acc += reg_product; sum_out = acc
//
// SIGNED: both inputs and accumulator are two's complement.
// This is required for Q8.8 attention scores, which can be negative.
//
// valid_out is valid_in delayed by 3 cycles to match data latency.
// Caller contract: assert reset between independent dot products.

`timescale 1ns/1ps

module mac_pe #(
    parameter int DATA_WIDTH = 16,   // changed from 8 to match Q8.8
    parameter int ACC_WIDTH  = 32    // changed from 16 to hold 16x16 products
)(
    input  logic                         clk,
    input  logic                         reset,     // active-high, synchronous
    input  logic                         valid_in,
    input  logic signed [DATA_WIDTH-1:0] a_in,
    input  logic signed [DATA_WIDTH-1:0] b_in,
    output logic signed [ACC_WIDTH-1:0]  sum_out,
    output logic                         valid_out
);

    logic signed [DATA_WIDTH-1:0]   reg_a;
    logic signed [DATA_WIDTH-1:0]   reg_b;
    logic signed [2*DATA_WIDTH-1:0] reg_product;
    logic signed [ACC_WIDTH-1:0]    acc;
    logic [2:0]                     valid_pipe;

    always_ff @(posedge clk) begin
        if (reset) begin
            reg_a      <= '0;
            reg_b      <= '0;
            reg_product<= '0;
            acc        <= '0;
            sum_out    <= '0;
            valid_pipe <= 3'b0;
        end else begin
            reg_a       <= a_in;                    // Stage 1: latch
            reg_b       <= b_in;
            reg_product <= reg_a * reg_b;           // Stage 2: signed multiply
            acc         <= acc + ACC_WIDTH'(reg_product); // Stage 3: accumulate
            sum_out     <= acc + ACC_WIDTH'(reg_product);
            valid_pipe  <= {valid_pipe[1:0], valid_in};
        end
    end

    assign valid_out = valid_pipe[2];

endmodule