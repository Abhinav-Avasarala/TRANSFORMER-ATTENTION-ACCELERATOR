// reciprocal_nr.sv
//
// Computes R ~= 1/D in Q8.8, using a bit-shift seed + Newton-Raphson.
//
// PRECONDITION (caller's responsibility): D >= 256 (i.e. D represents a
// real value >= 1.0). softmax_unit only ever calls this with D = a row's
// sum of exp() values, which is guaranteed >= 1.0 because the row's own
// max-scoring element always contributes exp(0) = 1.0 to the sum. This
// keeps msb_pos >= FRAC_BITS, so the seed's right-shift amount
// (msb_pos - FRAC_BITS) stays non-negative, without needing an explicit
// divide-by-zero guard.
//
// Algorithm:
//   1. Seed: find m = position of D's MSB (priority encoder), plus the
//      3 mantissa bits just below it, to pick a seed from an 8-entry
//      table -- within ~6% of the true reciprocal (Q8.8 integers), vs.
//      ~2x error if only m is used. No multiplier needed either way.
//   2. Newton-Raphson iterations: r_{n+1} = r_n * (2 - D*r_n)
//      Each iteration roughly doubles the number of correct bits, so 3
//      iterations starting from a within-6% seed comfortably covers
//      Q8.8's 8 fractional bits.
//
// WHY THE MANTISSA BITS MATTER (not just an accuracy nicety): a coarse
// seed using only m has up to 2x error, and Newton-Raphson division only
// self-corrects while D*r stays reasonably close to 1.0. If the seed is
// ~2x too big, D*r0 ~= 2.0, so the first correction factor (2 - D*r0)
// collapses to ~0 -- multiplying r by that destroys the estimate instead
// of refining it, and Q8.8's coarse resolution (nothing below 1/256)
// means it never recovers in the remaining iterations. Verified by hand:
// with the coarse (m-only) seed, D=511 (row_sum ~= 2.0) collapses to
// r=1 (decoded ~0.004) instead of converging to the correct r=128
// (~0.5) -- a ~50% error, not rounding noise. The 3-mantissa-bit seed
// below fixes this (D=511 now converges to exactly 128 in 1 iteration).

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

    // ---- Fine seed: 3 mantissa bits just below the MSB, i.e. which of
    // 8 sub-buckets D falls into within [2^m, 2^(m+1)). base_seed[frac]
    // is precomputed for the canonical bucket m=FRAC_BITS (D in
    // [2^FRAC_BITS, 2^(FRAC_BITS+1))); every other bucket is the same
    // shape just scaled by a power of two, handled by the right-shift in
    // SEED below. Table values: round(2^(FRAC_BITS+4) / (17 + 2*frac)),
    // i.e. round(256/x_center) for x_center = 1 + (2*frac+1)/16 -- the
    // midpoint of sub-bucket `frac` in normalized [1,2) space. Assumes
    // FRAC_BITS=8 (this project's only Q8.8 configuration); a different
    // FRAC_BITS would need a regenerated table.
    logic [2:0] frac_bits;
    assign frac_bits = d_reg[msb_pos-1 -: 3];

    logic [DATA_WIDTH-1:0] base_seed;
    always_comb begin
        case (frac_bits)
            3'd0: base_seed = 16'd241;
            3'd1: base_seed = 16'd216;
            3'd2: base_seed = 16'd195;
            3'd3: base_seed = 16'd178;
            3'd4: base_seed = 16'd164;
            3'd5: base_seed = 16'd152;
            3'd6: base_seed = 16'd141;
            3'd7: base_seed = 16'd132;
        endcase
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

                // r0 = base_seed[frac_bits] >> (msb_pos - FRAC_BITS)
                // (see header + table comments above for derivation)
                SEED: begin
                    r_reg    <= base_seed >> (msb_pos - FRAC_BITS);
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
