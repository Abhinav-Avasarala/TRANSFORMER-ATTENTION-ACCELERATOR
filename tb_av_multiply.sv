// tb_av_multiply.sv
//
// Testbench for av_multiply (O = A * V in Q8.8 fixed-point).
//
// Structure (same shape as tb_qk_systolic.sv, same reasoning per layer):
//   Layer 1  All zeros                 - catches wiring/reset bugs first
//   Layer 2  Identity-A x random V     - A=I means O should equal V exactly;
//                                        this is the real test of the
//                                        A-row-fed / V-column-fed skew logic,
//                                        since it's driven by real (nonzero,
//                                        varied) V data instead of zeros.
//   Layer 3  Known small (hand-checked)- one case you can verify by hand
//   Layer 4  Constrained random (CRT)  - the real bug catcher, many iterations
//   Layer 5  Back-to-back              - verifies the auto-clear-between-runs contract
//
// Golden model: a software blend that mirrors the RTL's fixed-point
// arithmetic EXACTLY (same 32-bit accumulate, same >>FRAC_BITS truncation).
// Every layer compares the DUT against this same golden model.
//
// NOTE: same non-negative-only constraint as tb_qk_systolic.sv (mac_pe's
// ports are actually signed, but this testbench sticks to non-negative
// Q8.8 for consistency with the existing regression-test convention).
// A signed cross-check against scripts/gen_golden.py, the way
// tb_qk_systolic_golden.sv cross-checks qk_systolic, is a natural follow-up
// once this passes.

`timescale 1ns/1ps

module tb_av_multiply;

    // ---- Parameters (same shape as target; N=D=64 here, kept distinct
    //      in the code even though they're numerically equal today) ----
    localparam int N          = 64;   // sequence length: rows of A/V/O, contraction length
    localparam int D          = 64;   // head dimension: columns of V/O
    localparam int DATA_WIDTH = 16;
    localparam int ACC_WIDTH  = 32;
    localparam int FRAC_BITS  = 8;
    localparam int CLK_PERIOD = 10;

    // Feeding window: A's row-feed needs up to (N-1)+(N-1); V's column-feed
    // needs up to (D-1)+(N-1). Total feed cycles must cover the later of
    // the two edges' last valid feed.
    localparam int MAX_ND      = (N > D) ? N : D;
    localparam int FEED_CYCLES = N + MAX_ND - 1;

    // ---- DUT signals ----
    logic                  clk;
    logic                  rst_n;
    logic                  start;
    logic [DATA_WIDTH-1:0] a_in  [0:N-1];
    logic [DATA_WIDTH-1:0] v_in  [0:D-1];
    logic [DATA_WIDTH-1:0] o_out [0:N-1][0:D-1];
    logic                  done;

    // ---- Test data (full matrices held here, fed in skewed) ----
    logic [DATA_WIDTH-1:0] A [0:N-1][0:N-1];
    logic [DATA_WIDTH-1:0] V [0:N-1][0:D-1];

    // ---- Scoreboard ----
    int tests_run    = 0;
    int tests_passed = 0;
    int tests_failed = 0;

    // ---- DUT ----
    av_multiply #(
        .N(N), .D(D), .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .a_in(a_in), .v_in(v_in), .o_out(o_out), .done(done)
    );

    // ---- Clock ----
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // =========================================================================
    // Golden model: must match RTL arithmetic bit-for-bit.
    //   raw = sum_j A[i][j]*V[j][c]   (32-bit accumulate, wraps like hardware)
    //   O   = raw[FRAC_BITS +: DATA_WIDTH]   (same >>8 truncation as the RTL)
    // =========================================================================
    function automatic logic [DATA_WIDTH-1:0] golden_o(input int i, input int c);
        logic [ACC_WIDTH-1:0] acc;
        acc = '0;
        for (int j = 0; j < N; j++)
            acc += A[i][j] * V[j][c];
        return acc[FRAC_BITS +: DATA_WIDTH];
    endfunction

    // =========================================================================
    // Matrix fill helpers
    // =========================================================================
    task automatic fill_zero();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                A[i][j] = '0;
        for (int j = 0; j < N; j++)
            for (int c = 0; c < D; c++)
                V[j][c] = '0;
    endtask

    // Q8.8 identity: 1.0 is stored as (1 << FRAC_BITS) = 256.
    task automatic fill_identity_A();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                A[i][j] = (i == j) ? (1 << FRAC_BITS) : '0;
    endtask

    task automatic fill_random_A();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                A[i][j] = $urandom_range(0, 255);
    endtask

    task automatic fill_random_V();
        for (int j = 0; j < N; j++)
            for (int c = 0; c < D; c++)
                V[j][c] = $urandom_range(0, 255);
    endtask

    // =========================================================================
    // Reset: active-low, synchronous. Hold a few cycles, release on a negedge.
    // =========================================================================
    task automatic apply_reset();
        @(negedge clk);
        rst_n = 0; start = 0;
        for (int i = 0; i < N; i++) a_in[i] = '0;
        for (int c = 0; c < D; c++) v_in[c] = '0;
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
    endtask

    // =========================================================================
    // run_av: pulse start, feed skewed A/V, then wait for done.
    // A is fed ROW-wise (a_in[i] = A[i][t-i]), same pattern as Q.
    // V is fed COLUMN-wise (v_in[c] = V[t-c][c]) -- see av_multiply.sv header
    // for why this differs from qk_systolic's K feed.
    // All driving happens on negedge so data is stable for every posedge.
    // =========================================================================
    task automatic run_av();
        int guard;

        // 1-cycle start pulse (sampled by the posedge between these negedges)
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        // Feed iterations t = 0 .. FEED_CYCLES-1 with the skew pattern.
        // Outside its window each input is zero so idle PEs see 0*0 = 0.
        for (int t = 0; t < FEED_CYCLES; t++) begin
            for (int i = 0; i < N; i++)
                a_in[i] = (t >= i && (t - i) < N) ? A[i][t - i] : '0;
            for (int c = 0; c < D; c++)
                v_in[c] = (t >= c && (t - c) < N) ? V[t - c][c] : '0;
            @(negedge clk);
        end

        // Stop feeding; hold zeros until the pipeline drains and done fires.
        for (int i = 0; i < N; i++) a_in[i] = '0;
        for (int c = 0; c < D; c++) v_in[c] = '0;

        guard = 0;
        while (!done && guard < 4*(N + D)) begin
            @(posedge clk);
            guard++;
        end
        if (!done) $fatal(1, "TIMEOUT: done never asserted");
        // o_out is valid now (latched on the same edge done went high).
    endtask

    // =========================================================================
    // check: compare DUT o_out against the golden model.
    // =========================================================================
    task automatic check(input string name);
        int errors;
        logic [DATA_WIDTH-1:0] exp;
        errors = 0;
        tests_run++;
        for (int i = 0; i < N; i++)
            for (int c = 0; c < D; c++) begin
                exp = golden_o(i, c);
                if (o_out[i][c] !== exp) begin
                    if (errors < 5)
                        $display("    O[%0d][%0d]: got %0d, expected %0d",
                                 i, c, o_out[i][c], exp);
                    errors++;
                end
            end
        if (errors == 0) begin
            tests_passed++;
            $display("  [PASS] %s", name);
        end else begin
            tests_failed++;
            $display("  [FAIL] %s  (%0d/%0d mismatches)", name, errors, N*D);
        end
    endtask

    // run one blend end-to-end, then check it
    task automatic run_and_check(input string name);
        run_av();
        check(name);
    endtask

    // =========================================================================
    // Test sequence
    // =========================================================================
    initial begin
        $dumpfile("tb_av_multiply.vcd");
        $dumpvars(0, tb_av_multiply);

        $display("==================================================");
        $display(" av_multiply testbench  (N=%0d D=%0d Q8.8)", N, D);
        $display("==================================================");

        // ---- Layer 1: all zeros ----
        $display("\n-- Layer 1: All zeros --");
        apply_reset();
        fill_zero();
        run_and_check("All zeros -> O = 0");

        // ---- Layer 2: identity A x random V ----
        // O = I * V = V exactly (no truncation error -- multiplying by
        // 1.0 = 256 then shifting right by 8 recovers V bit-for-bit).
        // This exercises the full row/column skew with real nonzero data.
        $display("\n-- Layer 2: Identity-A x random V --");
        apply_reset();
        fill_identity_A();
        fill_random_V();
        run_and_check("Identity-A x random V -> O = V");

        // ---- Layer 3: known small, hand-checkable ----
        // A[0] = [0.5, 0.5, 0...]  (two attention weights summing to 1.0)
        // V[0] = [2.0, ...]  V[1] = [4.0, ...]
        // O[0][0] = 0.5*2.0 + 0.5*4.0 = 3.0  -> stored 3*256 = 768.
        $display("\n-- Layer 3: Known small (hand-checked) --");
        apply_reset();
        fill_zero();
        A[0][0] = 16'd128; A[0][1] = 16'd128;   // 0.5, 0.5
        V[0][0] = 16'd512;                       // 2.0
        V[1][0] = 16'd1024;                      // 4.0
        run_and_check("Known small (O[0][0] should be 768)");
        $display("    -> O[0][0] = %0d (expect 768)", o_out[0][0]);

        // ---- Layer 4: constrained random (CRT) ----
        $display("\n-- Layer 4: Constrained random --");
        for (int t = 0; t < 15; t++) begin
            apply_reset();
            fill_random_A();
            fill_random_V();
            run_and_check($sformatf("Random set %0d", t));
        end

        // ---- Layer 5: back-to-back (no manual reset between) ----
        // The module auto-clears on !busy, so two starts in a row must each
        // produce a correct, independent result.
        $display("\n-- Layer 5: Back-to-back --");
        apply_reset();
        fill_random_A(); fill_random_V();
        run_and_check("Back-to-back: first");
        fill_random_A(); fill_random_V();    // new data, NO apply_reset()
        run_and_check("Back-to-back: second (no reset between)");

        // ---- Summary ----
        $display("\n==================================================");
        $display(" SUMMARY: %0d/%0d passed  (%0d failed)",
                 tests_passed, tests_run, tests_failed);
        $display(" RESULT : %s",
                 (tests_failed == 0) ? "ALL TESTS PASSED" : "FAILED");
        $display("==================================================");
        $finish;
    end

    // ---- Watchdog ----
    initial begin
        #2_000_000;
        $display("ERROR: global simulation timeout");
        $finish;
    end

endmodule
