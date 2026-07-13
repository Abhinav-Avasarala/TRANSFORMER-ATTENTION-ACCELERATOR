// reciprocal_nr.sv
//
// Computes R ~= 1/D in Q8.8, using a bit-shift seed + Newton-Raphson.
//
// PRECONDITION (caller's responsibility): D >= 256 (i.e. D represents a
// real value >= 1.0). softmax_unit only ever calls this with D = a row's
// sum of exp() values, which is guaranteed >= 1.0 because the row's own
// max-scoring element always contributes exp(0) = 1.0 to the sum. This
// keeps the seed shift amount (16 - m) in a safe, non-negative range
// without needing an explicit divide-by-zero guard.
//
// Algorithm:
//   1. Seed: find m = position of D's MSB (priority encoder). The seed
//      r0 = 2^(16-m) is within ~2x of the true reciprocal 1/D (both are
//      expressed as Q8.8 integers). No multiply needed for the seed.
//   2. Newton-Raphson iterations: r_{n+1} = r_n * (2 - D*r_n)
//      Each iteration roughly doubles the number of correct bits, so 3
//      iterations starting from a within-2x seed is enough headroom for
//      Q8.8's 8 fractional bits.

`timescale 1ns/1ps

module reciprocal_nr #(
    parameter int DATA_WIDTH = 16,
    parameter int FRAC_BITS  = 8,
    parameter int ITERATIONS = 3
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  start,
    input  logic [DATA_WIDTH-1:0] d_in,    // Q8.8, unsigned, >= 256 (see precondition above)
    output logic [DATA_WIDTH-1:0] r_out,   // Q8.8, unsigned, ~= 1/d_in
    output logic                  done
);

    localparam int ITER_WIDTH = $clog2(ITERATIONS + 1);
    localparam logic [DATA_WIDTH-1:0] TWO_Q8_8 = (2 << FRAC_BITS); // 2.0 in Q8.8

    typedef enum logic [1:0] {IDLE, SEED, ITER, DONE_ST} state_t;
    state_t state;

    logic [DATA_WIDTH-1:0] d_reg, r_reg;
    logic [ITER_WIDTH-1:0] iter_cnt;

    // ---- Priority encoder: m = index of D's most-significant set bit ----
    logic [$clog2(DATA_WIDTH)-1:0] msb_pos;
    always_comb begin
        msb_pos = '0;
        for (int b = DATA_WIDTH-1; b >= 0; b--)
            if (d_reg[b]) begin
                msb_pos = b[$clog2(DATA_WIDTH)-1:0];
                break;
            end
    end

    // ---- One Newton-Raphson step, combinational ----
    //   t1 = (D * r) >> FRAC_BITS         (Q8.8 * Q8.8 -> Q8.8)
    //   t2 = 2.0 - t1                     (Q8.8)
    //   r_next = (r * t2) >> FRAC_BITS    (Q8.8)
    logic [2*DATA_WIDTH-1:0] prod_d_r, prod_r_t2;
    logic [DATA_WIDTH-1:0]   t1, t2, r_next;

    assign prod_d_r  = d_reg * r_reg;
    assign t1         = prod_d_r[FRAC_BITS +: DATA_WIDTH];
    assign t2         = TWO_Q8_8 - t1;
    assign prod_r_t2  = r_reg * t2;
    assign r_next      = prod_r_t2[FRAC_BITS +: DATA_WIDTH];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state    <= IDLE;
            d_reg    <= '0;
            r_reg    <= '0;
            iter_cnt <= '0;
            r_out    <= '0;
            done     <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    if (start) begin
                        d_reg <= d_in;
                        state <= SEED;
                    end
                end

                // r0 = 1 << (16 - msb_pos)  (see header comment for derivation)
                SEED: begin
                    r_reg    <= DATA_WIDTH'(1) << (DATA_WIDTH - msb_pos);
                    iter_cnt <= '0;
                    state    <= ITER;
                end

                ITER: begin
                    r_reg <= r_next;
                    if (iter_cnt == ITER_WIDTH'(ITERATIONS - 1)) begin
                        state <= DONE_ST;
                    end else begin
                        iter_cnt <= iter_cnt + 1'b1;
                    end
                end

                DONE_ST: begin
                    r_out <= r_reg;
                    done  <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
