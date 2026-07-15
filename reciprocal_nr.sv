// reciprocal_nr.sv
//
// Computes R ~= 1/D using a bit-shift seed + Newton-Raphson, with the
// OUTPUT format independent of the INPUT format. softmax_unit needs the
// reciprocal at higher fractional precision than Q8.8: 1/row_sum is a
// small number (~0.0156 .. 1.0), and storing it at Q8.8's 1/256 step
// gives up to ~25% relative error on the small end, which dominates the
// final attention error (measured: A=Q8.8+recip=Q8.8 -> max O error
// 0.07; widening BOTH the weights and this reciprocal -> 0.006). So this
// module now takes:
//   input  d_in : Q(DATA_WIDTH-FRAC_BITS).FRAC_BITS   (unchanged, Q8.8)
//   output r_out: Q(OUT_WIDTH-OUT_FRAC).OUT_FRAC      (softmax uses 16 frac)
//
// PRECONDITION (caller's responsibility): D >= 2^FRAC_BITS (i.e. D >= 1.0).
// softmax_unit only ever calls this with D = a row's sum of exp() values,
// guaranteed >= 1.0 (the row's max-scoring element contributes exp(0)=1.0).
// This keeps msb_pos >= FRAC_BITS, so the seed's right-shift amount stays
// non-negative, without needing a divide-by-zero guard.
//
// Algorithm:
//   1. Seed: MSB position (priority encoder) picks the power-of-two
//      bucket; 3 mantissa bits below it pick 1 of 8 sub-buckets from
//      SEED_LUT. SEED_LUT is COMPUTED at elaboration from OUT_FRAC (not
//      hardcoded), so it is correct at any output precision -- the old
//      hardcoded-for-Q8.8 table was the source of an earlier convergence
//      bug and would have been silently wrong at 16 frac bits.
//      Seed is within ~6% of 1/D (well clear of the ~2x point where the
//      NR correction term collapses to zero and the estimate dies).
//   2. Newton-Raphson: r <- r * (2 - D*r). Quadratic convergence (~doubles
//      correct bits each step). ITERATIONS default 3 covers Q8.8; use 4
//      for 16-bit output precision (see softmax_unit's instantiation).

`timescale 1ns/1ps

module reciprocal_nr #(
    parameter int DATA_WIDTH = 16,   // input d width
    parameter int FRAC_BITS  = 8,    // input d fractional bits
    parameter int OUT_WIDTH  = 16,   // output r width  (default: same as input)
    parameter int OUT_FRAC   = 8,    // output r fractional bits (default: Q8.8)
    parameter int ITERATIONS = 3
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  start,
    input  logic [DATA_WIDTH-1:0] d_in,    // Q(.FRAC_BITS), unsigned, >= 1.0
    output logic [OUT_WIDTH-1:0]  r_out,   // Q(.OUT_FRAC),  unsigned, ~= 1/d_in
    output logic                  done
);

    localparam int ITER_WIDTH = $clog2(ITERATIONS + 1);
    localparam int MSB_WIDTH   = $clog2(DATA_WIDTH);

    // ---- Seed table, computed at elaboration for this OUT_FRAC ----
    // SEED_LUT[frac] = round(2^OUT_FRAC / mantissa_center(frac)), where
    // mantissa_center(frac) = 1 + (2*frac+1)/16 is the midpoint of the
    // frac-th sub-bucket of the normalized mantissa range [1, 2).
    // NOTE: int'() rounds to nearest on its own (unlike $rtoi(), which
    // truncates) -- no "+ 0.5" needed. See softmax_unit.sv's SCALE_CONST
    // comment for how the same mistake there caused a real, measured bug.
    // Harmless here in practice (Newton-Raphson's self-correction absorbs
    // a 1-off seed), but fixed for correctness and consistency.
    function automatic int unsigned seed_val(input int frac);
        real center;
        center = 1.0 + (2.0 * real'(frac) + 1.0) / 16.0;
        return int'(real'(64'd1 << OUT_FRAC) / center);
    endfunction

    localparam int unsigned SEED_LUT [0:7] = '{
        seed_val(0), seed_val(1), seed_val(2), seed_val(3),
        seed_val(4), seed_val(5), seed_val(6), seed_val(7)
    };

    typedef enum logic [1:0] {IDLE, SEED, ITER, DONE_ST} state_t;
    state_t state;

    logic [DATA_WIDTH-1:0] d_reg;
    logic [OUT_WIDTH-1:0]  r_reg;
    logic [ITER_WIDTH-1:0] iter_cnt;

    // ---- Priority encoder: msb_pos = index of D's most-significant set bit ----
    logic [MSB_WIDTH-1:0] msb_pos;
    always_comb begin
        msb_pos = '0;
        for (int b = DATA_WIDTH-1; b >= 0; b--)
            if (d_reg[b]) begin
                msb_pos = b[MSB_WIDTH-1:0];
                break;
            end
    end

    // 3 mantissa bits just below the MSB -> which of 8 sub-buckets.
    logic [2:0] frac_bits;
    assign frac_bits = d_reg[msb_pos-1 -: 3];

    // ---- One Newton-Raphson step, combinational, with mixed input/output
    // fractional bits tracked explicitly:
    //   t1 = (D * r) >> FRAC_BITS   -> product's OUT_FRAC frac bits, ~= 1.0
    //   t2 = 2.0 - t1               (at OUT_FRAC frac)
    //   r_next = (r * t2) >> OUT_FRAC
    logic [DATA_WIDTH+OUT_WIDTH-1:0] prod_d_r;
    logic [OUT_WIDTH+1:0]            t1, t2;
    logic [2*OUT_WIDTH+1:0]          prod_r_t2;
    logic [OUT_WIDTH-1:0]            r_next;

    localparam logic [OUT_WIDTH+1:0] TWO_OUT = (OUT_WIDTH+2)'(2) << OUT_FRAC;

    assign prod_d_r  = d_reg * r_reg;
    assign t1         = prod_d_r[FRAC_BITS +: (OUT_WIDTH+2)];
    assign t2         = TWO_OUT - t1;
    assign prod_r_t2  = r_reg * t2;
    assign r_next      = prod_r_t2[OUT_FRAC +: OUT_WIDTH];

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

                // r0 = SEED_LUT[frac_bits] >> (msb_pos - FRAC_BITS)
                SEED: begin
                    r_reg    <= OUT_WIDTH'(SEED_LUT[frac_bits]) >> (msb_pos - MSB_WIDTH'(FRAC_BITS));
                    iter_cnt <= '0;
                    state    <= ITER;
                end

                ITER: begin
                    r_reg <= r_next;
                    if (iter_cnt == ITER_WIDTH'(ITERATIONS - 1))
                        state <= DONE_ST;
                    else
                        iter_cnt <= iter_cnt + 1'b1;
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
