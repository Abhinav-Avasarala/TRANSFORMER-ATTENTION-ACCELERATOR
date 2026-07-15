// softmax_unit.sv
//
// Computes A = row-wise softmax(S / sqrt(D)) in Q8.8 (the "scale +
// normalize" stage of attention -- per the project doc's own module
// breakdown, softmax_unit owns BOTH the /sqrt(d_k) scale step and the
// normalize step; qk_systolic only computes the raw, unscaled Q*K^T).
// Takes the full score matrix from qk_systolic and produces the
// attention-weight matrix consumed by av_multiply.
//
//   S : [N x N]  Raw, UNSCALED scores from qk_systolic (signed Q8.8).
//   A : [N x N]  Attention weights, emitted as Q1.15 (unsigned [0,1),
//                clamped to 0x7FFF), NOT Q8.8 like the other stages.
//                Attention weights are small probabilities; Q8.8's 1/256
//                step is too coarse for them (caps end-to-end error at
//                ~0.07). Q1.15 gives 128x finer resolution for the same
//                16 bits, since weights never need the integer range.
//                The reciprocal 1/row_sum is likewise computed at 16 frac
//                bits (RECIP_FRAC) for the same reason -- both are needed
//                to reach the project's 0.01 max-error target.
//
// SCALING: rather than scaling row_max and every S element separately
// (which would add a multiply to MAX_SCAN too), this scales `delta`
// once, after the max-subtraction: scaling is multiplication by a
// positive constant, and SCALE*row_max - SCALE*s = SCALE*(row_max - s),
// so one multiply per element in EXP_SCAN is mathematically equivalent
// and cheaper. SCALE_CONST = round(256 / sqrt(D)) is computed once at
// elaboration time (D=64 makes this exactly 0.125 = a pure right-shift-
// by-3 in disguise, but the general multiply-by-constant form works for
// any D, not just this convenient special case).
//
// Unlike qk_systolic/av_multiply, this is NOT a systolic PE grid -- softmax
// is inherently row-sequential (you need a row's max and total sum before
// you can normalize any element in that row), so this is a single FSM that
// processes one row at a time, one element per cycle, in three passes:
//
//   MAX_SCAN  : find row_max = max(S[row][:])                    (N cycles)
//   EXP_SCAN  : delta = row_max - S[row][col] (always >= 0)
//               exp_buf[col] = LUT[delta], row_sum += exp_buf[col] (N cycles)
//   (reciprocal_nr computes 1/row_sum -- ~5 cycles)
//   NORM_SCAN : A[row][col] = exp_buf[col] * (1/row_sum)          (N cycles)
//
// This is a deliberately simple (not pipelined-across-rows) design: total
// latency is roughly N rows * (3N + ~7) cycles. At N=64 that's ~12,700
// cycles for this stage alone, more than the project doc's <10,000-cycle
// target for the WHOLE attention pass. Overlapping rows (e.g. starting
// row i+1's MAX_SCAN during row i's NORM_SCAN) would cut this significantly
// but adds real FSM complexity -- worth revisiting once this is verified
// correct, not before.
//
// LUT: exp_lut.hex must be generated with scripts/gen_exp_lut.py and be
// visible to whatever tool elaborates this file (Vivado resolves the
// $readmemh path relative to its own working directory -- same gotcha as
// tb_qk_systolic_golden.sv's GOLDEN_DIR). NOTE: reading exp_lut as a plain
// combinational array (no registered read port) will likely synthesize as
// distributed RAM (LUTs), not block RAM -- fine functionally, but if you
// want this counted as BRAM in your utilization report, register the read
// address/data instead.

`timescale 1ns/1ps

module softmax_unit #(
    parameter int N             = 64,   // sequence length: rows/cols of S and A
    parameter int D             = 64,   // head dimension -- used ONLY to compute the
                                         // /sqrt(D) scale constant below; softmax_unit
                                         // has no other reason to know D
    parameter int DATA_WIDTH    = 16,   // Q8.8 element width (all external I/O)
    parameter int FRAC_BITS     = 8,    // Q8.8 fractional bits (all external I/O)
    parameter int LUT_DEPTH     = 2048, // exp() LUT entries, domain delta in [0, 8.0)
    parameter int EXP_FRAC_BITS = 16,   // internal exp()/row_sum precision -- see
                                         // scripts/gen_exp_lut.py header for why this
                                         // needs to be wider than Q8.8's 8 fractional bits
    parameter string EXP_LUT_FILE = "exp_lut.hex"
)(
    input  logic                  clk,
    input  logic                  rst_n,        // active-low, synchronous
    input  logic                  start,        // 1-cycle pulse to begin

    input  logic [DATA_WIDTH-1:0] s_in  [0:N-1][0:N-1],  // scores from qk_systolic, signed Q8.8
    output logic [DATA_WIDTH-1:0] a_out [0:N-1][0:N-1],  // attention weights, Q1.15 (NOT Q8.8):
                                                          // unsigned [0, 0.99997], clamped to 0x7FFF.
                                                          // av_multiply must consume as Q1.15.

    output logic                  done
);

    localparam int LUT_ADDR_WIDTH = $clog2(LUT_DEPTH);
    localparam int IDX_WIDTH      = $clog2(N);
    localparam logic [DATA_WIDTH-1:0] NEG_SENTINEL = {1'b1, {(DATA_WIDTH-1){1'b0}}}; // most-negative Q8.8

    // Internal exp()/row_sum precision: EXP_WIDTH holds an unsigned value in
    // [0, 2^EXP_FRAC_BITS] (i.e. Q1.EXP_FRAC_BITS -- the "+1" bit exists only
    // to represent exactly 1.0 at delta=0). ROW_SUM_WIDTH adds enough extra
    // integer headroom to sum up to N such values without overflow.
    localparam int EXP_WIDTH     = EXP_FRAC_BITS + 1;
    localparam int ROW_SUM_WIDTH = EXP_WIDTH + $clog2(N);

    // Reciprocal + attention-weight output precision. The reciprocal
    // 1/row_sum is a small value (~0.0156 .. 1.0); storing it (and the
    // output attention weights) at Q8.8 caps end-to-end accuracy at
    // ~0.07 max error. Both must be wider to hit the 0.01 target:
    //   RECIP_FRAC=16  -> reciprocal fractional bits (was 8)
    //   A_FRAC_BITS=15 -> attention weights emitted as Q1.15, not Q8.8.
    // A_FRAC_BITS=15 (not 16) leaves the top bit clear so av_multiply's
    // SIGNED mac_pe reads weights as positive; weights are clamped to
    // 0x7FFF (0.99997) for the same reason.
    localparam int RECIP_FRAC   = 16;
    localparam int RECIP_WIDTH  = RECIP_FRAC + 2;   // holds 1/d up to 1.0 (=2^16) + margin
    localparam int A_FRAC_BITS  = DATA_WIDTH - 1;   // Q1.15
    localparam int NORM_FRAC    = EXP_FRAC_BITS + RECIP_FRAC;  // frac bits of exp*recip product

    // -------------------------------------------------------------------------
    // exp() lookup table: LUT[delta] = round(2^EXP_FRAC_BITS * exp(-delta/256)).
    // Stored wider than Q8.8 on purpose -- see scripts/gen_exp_lut.py header:
    // exp() values here are often much smaller than Q8.8's smallest step
    // (1/256), so storing them at Q8.8 precision was rounding many small
    // contributions up by 50%+ each, which compounded into a measurable bias
    // in row_sum for peaked (low-entropy) rows. See the file header above for
    // the BRAM-vs-LUTRAM inference caveat, which still applies.
    // -------------------------------------------------------------------------
    logic [EXP_WIDTH-1:0] exp_lut [0:LUT_DEPTH-1];
    initial $readmemh(EXP_LUT_FILE, exp_lut);

    // -------------------------------------------------------------------------
    // Reciprocal submodule, reused once per row.
    // -------------------------------------------------------------------------
    logic                   recip_start, recip_done;
    logic [DATA_WIDTH-1:0]  recip_d;                  // Q8.8 in (row_sum)
    logic [RECIP_WIDTH-1:0] recip_r;                  // Q?.RECIP_FRAC out

    reciprocal_nr #(
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .OUT_WIDTH(RECIP_WIDTH), .OUT_FRAC(RECIP_FRAC)
    ) recip_inst (
        .clk(clk), .rst_n(rst_n),
        .start(recip_start), .d_in(recip_d),
        .r_out(recip_r), .done(recip_done)
    );

    // -------------------------------------------------------------------------
    // FSM state
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        IDLE, MAX_SCAN, EXP_SCAN, RECIP_KICK, RECIP_WAIT, NORM_SCAN, ROW_NEXT, ALL_DONE
    } state_t;
    state_t state;

    logic [IDX_WIDTH-1:0]    row, col;
    logic [DATA_WIDTH-1:0]   row_max;         // signed Q8.8, this row's max score
    logic [ROW_SUM_WIDTH-1:0] row_sum;        // unsigned Q(ROW_SUM_WIDTH-EXP_FRAC_BITS).EXP_FRAC_BITS,
                                               // sum of this row's high-precision exp() values
    logic [EXP_WIDTH-1:0]    exp_buf [0:N-1];  // this row's exp() values, buffered until row_sum is final
    logic [RECIP_WIDTH-1:0]  recip_result;     // latched 1/row_sum for this row, Q?.RECIP_FRAC

    assign recip_start = (state == RECIP_KICK);

    // row_sum (Q?.EXP_FRAC_BITS) -> recip_d (Q8.8): reciprocal_nr only ever
    // operates in Q8.8, so this is the one place row_sum gets truncated down
    // from its wider internal precision. Dropping (EXP_FRAC_BITS-FRAC_BITS)
    // fractional bits here is a single truncation on an already-accurate sum
    // -- a world apart from truncating on every one of N individual LUT
    // entries before summing, which is what caused the original bias.
    assign recip_d = {1'b0, row_sum[(EXP_FRAC_BITS-FRAC_BITS) +: (DATA_WIDTH-1)]};

    // ---- delta = row_max - S[row][col], sign-extended subtraction ----
    // Always >= 0 by construction: row_max is the true max of the row,
    // computed in the prior MAX_SCAN pass, so no negative-delta case exists.
    logic signed [DATA_WIDTH:0] row_max_ext, s_val_ext, delta_s;
    assign row_max_ext = {row_max[DATA_WIDTH-1], row_max};
    assign s_val_ext   = {s_in[row][col][DATA_WIDTH-1], s_in[row][col]};
    assign delta_s      = row_max_ext - s_val_ext;

    // ---- scale delta by 1/sqrt(D) -- see file header for why this one
    // multiply (instead of scaling row_max/s_in separately) is sufficient
    // and mathematically equivalent to the doc's "divide scores by
    // sqrt(d_k)" step. SCALE_CONST is a Q8.8 constant, computed once at
    // elaboration time.
    // NOTE: int'() (a real-to-int CAST) already rounds to nearest per the
    // SV LRM -- unlike $rtoi(), which truncates. An earlier version of this
    // line added "+ 0.5" here out of habit from $rtoi() use elsewhere, which
    // double-rounded: 0.125*256=32.0 exactly became 32.5, which then rounded
    // UP to 33, a silent 3.125% scaling error on every delta value (found by
    // tb_softmax_debug.sv: row_sum was 74041 instead of the hand-derived
    // 75742). No "+ 0.5" needed here.
    localparam real SCALE_REAL  = 1.0 / $sqrt(real'(D));
    localparam int  SCALE_CONST = int'(SCALE_REAL * real'(1 << FRAC_BITS));

    logic [DATA_WIDTH:0] delta_u;   // delta_s reinterpreted unsigned (always >= 0)
    assign delta_u = delta_s[DATA_WIDTH:0];

    logic [2*DATA_WIDTH:0] delta_scaled_prod;   // (DATA_WIDTH+1) + DATA_WIDTH bits
    assign delta_scaled_prod = delta_u * DATA_WIDTH'(SCALE_CONST);

    logic [DATA_WIDTH:0] delta_scaled;          // Q8.8, same width as delta_u
    assign delta_scaled = delta_scaled_prod[FRAC_BITS +: (DATA_WIDTH+1)];

    logic [LUT_ADDR_WIDTH-1:0] delta_idx;
    assign delta_idx = (delta_scaled > (DATA_WIDTH+1)'(LUT_DEPTH-1))
                        ? (LUT_ADDR_WIDTH)'(LUT_DEPTH-1)
                        : delta_scaled[LUT_ADDR_WIDTH-1:0];

    logic [EXP_WIDTH-1:0] exp_val;
    assign exp_val = exp_lut[delta_idx];

    // ---- normalize multiply: A[row][col] = exp_buf[col] * recip_result ----
    // exp_buf is Q1.EXP_FRAC_BITS, recip_result is Q?.RECIP_FRAC -- product
    // has NORM_FRAC (= EXP_FRAC_BITS + RECIP_FRAC) fractional bits. To emit
    // the weight as Q1.A_FRAC_BITS, drop (NORM_FRAC - A_FRAC_BITS) frac bits.
    logic [EXP_WIDTH+RECIP_WIDTH-1:0] norm_prod;
    assign norm_prod = exp_buf[col] * recip_result;

    // Weight in Q1.A_FRAC_BITS, plus an overflow flag for weights that reach
    // >= 1.0 (norm_prod bits at/above NORM_FRAC set). Such weights are
    // clamped to 0x7FFF so the value stays in signed-positive range for
    // av_multiply's mac_pe (which treats a_in as signed).
    logic [DATA_WIDTH-1:0] a_val;
    logic                  a_ovf;
    assign a_val  = norm_prod[(NORM_FRAC - A_FRAC_BITS) +: DATA_WIDTH];
    assign a_ovf  = |norm_prod[EXP_WIDTH+RECIP_WIDTH-1 : NORM_FRAC];  // weight >= 1.0

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            row   <= '0;
            col   <= '0;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    if (start) begin
                        row     <= '0;
                        col     <= '0;
                        row_max <= NEG_SENTINEL;
                        state   <= MAX_SCAN;
                    end
                end

                MAX_SCAN: begin
                    if ($signed(s_in[row][col]) > $signed(row_max))
                        row_max <= s_in[row][col];
                    if (col == IDX_WIDTH'(N-1)) begin
                        col     <= '0;
                        row_sum <= '0;
                        state   <= EXP_SCAN;
                    end else begin
                        col <= col + 1'b1;
                    end
                end

                EXP_SCAN: begin
                    exp_buf[col] <= exp_val;
                    row_sum      <= row_sum + exp_val;
                    if (col == IDX_WIDTH'(N-1)) begin
                        state <= RECIP_KICK;
                    end else begin
                        col <= col + 1'b1;
                    end
                end

                RECIP_KICK: begin
                    state <= RECIP_WAIT;
                end

                RECIP_WAIT: begin
                    if (recip_done) begin
                        recip_result <= recip_r;
                        col          <= '0;
                        state        <= NORM_SCAN;
                    end
                end

                NORM_SCAN: begin
                    a_out[row][col] <= a_ovf ? {1'b0, {(DATA_WIDTH-1){1'b1}}}  // 0x7FFF
                                             : a_val;
                    if (col == IDX_WIDTH'(N-1)) begin
                        state <= ROW_NEXT;
                    end else begin
                        col <= col + 1'b1;
                    end
                end

                ROW_NEXT: begin
                    if (row == IDX_WIDTH'(N-1)) begin
                        state <= ALL_DONE;
                    end else begin
                        row     <= row + 1'b1;
                        col     <= '0;
                        row_max <= NEG_SENTINEL;
                        state   <= MAX_SCAN;
                    end
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
