// tb_qk_systolic.sv
//
// Testbench for qk_systolic (S = Q * K^T in Q8.8 fixed-point).
//
// Structure (each layer builds confidence before the next):
//   Layer 1  All zeros                 - catches wiring/reset bugs first
//   Layer 2  Identity x Identity       - simplest non-trivial structure
//   Layer 3  Known small (hand-checked)- one case you can verify by hand
//   Layer 4  Constrained random (CRT)  - the real bug catcher, many iterations
//   Layer 5  Back-to-back              - verifies the auto-clear-between-runs contract
//
// Golden model: a software dot product that mirrors the RTL's fixed-point
// arithmetic EXACTLY (same 32-bit accumulate, same >>FRAC_BITS truncation).
// Every layer compares the DUT against this same golden model.
//
// NOTE: the current mac_pe multiplier is UNSIGNED, so this testbench uses
// non-negative Q8.8 inputs. Real attention needs signed scores; making the
// datapath signed is a separate change (flagged after this file).

`timescale 1ns/1ps

module tb_qk_systolic;

    // ---- Parameters (small enough to simulate fast; same shape as target) ----
    localparam int N          = 64;
    localparam int D          = 64;
    localparam int DATA_WIDTH = 16;
    localparam int ACC_WIDTH  = 32;
    localparam int FRAC_BITS  = 8;
    localparam int CLK_PERIOD = 10;

    // Feeding window: element Q[i][k] is presented at feed-cycle t = i+k,
    // so the last nonzero feed is at t = (N-1)+(D-1). Total feed cycles:
    localparam int FEED_CYCLES = N + D - 1;

    // ---- DUT signals ----
    logic                  clk;
    logic                  rst_n;
    logic                  start;
    logic [DATA_WIDTH-1:0] q_in  [0:N-1];
    logic [DATA_WIDTH-1:0] k_in  [0:N-1];
    logic [DATA_WIDTH-1:0] s_out [0:N-1][0:N-1];
    logic                  done;

    // ---- Test data (full matrices held here, fed in skewed) ----
    logic [DATA_WIDTH-1:0] Q [0:N-1][0:D-1];
    logic [DATA_WIDTH-1:0] K [0:N-1][0:D-1];

    // ---- Scoreboard ----
    int tests_run    = 0;
    int tests_passed = 0;
    int tests_failed = 0;

    // ---- DUT ----
    qk_systolic #(
        .N(N), .D(D), .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .q_in(q_in), .k_in(k_in), .s_out(s_out), .done(done)
    );

    // ---- Clock ----
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // =========================================================================
    // Golden model: must match RTL arithmetic bit-for-bit.
    //   raw = sum_k Q[i][k]*K[j][k]   (32-bit accumulate, wraps like hardware)
    //   S   = raw[FRAC_BITS +: DATA_WIDTH]   (same >>8 truncation as the RTL)
    // =========================================================================
    function automatic logic [DATA_WIDTH-1:0] golden_s(input int i, input int j);
        logic [ACC_WIDTH-1:0] acc;
        acc = '0;
        for (int k = 0; k < D; k++)
            acc += Q[i][k] * K[j][k];
        return acc[FRAC_BITS +: DATA_WIDTH];
    endfunction

    // =========================================================================
    // Matrix fill helpers
    // =========================================================================
    task automatic fill_zero();
        for (int i = 0; i < N; i++)
            for (int k = 0; k < D; k++) begin
                Q[i][k] = '0;
                K[i][k] = '0;
            end
    endtask

    // Q8.8 identity: 1.0 is stored as (1 << FRAC_BITS) = 256.
    task automatic fill_identity();
        fill_zero();
        for (int i = 0; i < N; i++) begin
            Q[i][i] = (1 << FRAC_BITS);
            K[i][i] = (1 << FRAC_BITS);
        end
    endtask

    // Constrained random: 0..255 keeps results in range and far from overflow.
    task automatic fill_random();
        for (int i = 0; i < N; i++)
            for (int k = 0; k < D; k++) begin
                Q[i][k] = $urandom_range(0, 255);
                K[i][k] = $urandom_range(0, 255);
            end
    endtask

    // =========================================================================
    // Reset: active-low, synchronous. Hold a few cycles, release on a negedge.
    // =========================================================================
    task automatic apply_reset();
        @(negedge clk);
        rst_n = 0; start = 0;
        for (int i = 0; i < N; i++) begin q_in[i] = '0; k_in[i] = '0; end
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
    endtask

    // =========================================================================
    // run_qk: pulse start, feed skewed Q/K, then wait for done.
    // All driving happens on negedge so data is stable for every posedge.
    // =========================================================================
    task automatic run_qk();
        int guard;

        // 1-cycle start pulse (sampled by the posedge between these negedges)
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        // Feed iterations t = 0 .. FEED_CYCLES-1 with the skew pattern.
        // Outside its window each input is zero so idle PEs see 0*0 = 0.
        for (int t = 0; t < FEED_CYCLES; t++) begin
            for (int i = 0; i < N; i++)
                q_in[i] = (t >= i && (t - i) < D) ? Q[i][t - i] : '0;
            for (int j = 0; j < N; j++)
                k_in[j] = (t >= j && (t - j) < D) ? K[j][t - j] : '0;
            @(negedge clk);
        end

        // Stop feeding; hold zeros until the pipeline drains and done fires.
        for (int i = 0; i < N; i++) begin q_in[i] = '0; k_in[i] = '0; end

        guard = 0;
        while (!done && guard < 4*(N + D)) begin
            @(posedge clk);
            guard++;
        end
        if (!done) $fatal(1, "TIMEOUT: done never asserted");
        // s_out is valid now (latched on the same edge done went high).
    endtask

    // =========================================================================
    // check: compare DUT s_out against the golden model.
    // =========================================================================
    task automatic check(input string name);
        int errors;
        logic [DATA_WIDTH-1:0] exp;
        errors = 0;
        tests_run++;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) begin
                exp = golden_s(i, j);
                if (s_out[i][j] !== exp) begin
                    if (errors < 5)
                        $display("    S[%0d][%0d]: got %0d, expected %0d",
                                 i, j, s_out[i][j], exp);
                    errors++;
                end
            end
        if (errors == 0) begin
            tests_passed++;
            $display("  [PASS] %s", name);
        end else begin
            tests_failed++;
            $display("  [FAIL] %s  (%0d/%0d mismatches)", name, errors, N*N);
        end
    endtask

    // run one matmul end-to-end, then check it
    task automatic run_and_check(input string name);
        run_qk();
        check(name);
    endtask

    // =========================================================================
    // Test sequence
    // =========================================================================
    initial begin
        $dumpfile("tb_qk_systolic.vcd");
        $dumpvars(0, tb_qk_systolic);

        $display("==================================================");
        $display(" qk_systolic testbench  (N=%0d D=%0d Q8.8)", N, D);
        $display("==================================================");

        // ---- Layer 1: all zeros ----
        $display("\n-- Layer 1: All zeros --");
        apply_reset();
        fill_zero();
        run_and_check("All zeros -> S = 0");

        // ---- Layer 2: identity x identity ----
        // S = I * I^T = I. In Q8.8, diagonal = 256 (1.0), off-diagonal = 0.
        $display("\n-- Layer 2: Identity x Identity --");
        apply_reset();
        fill_identity();
        run_and_check("Identity x Identity -> S = I");

        // ---- Layer 3: known small, hand-checkable ----
        // Q row0 = [1.0, 2.0, 0...]  K row0 = [1.0, 1.0, 0...]
        // S[0][0] = 1*1 + 2*1 = 3.0  -> stored 3*256 = 768. All else 0.
        $display("\n-- Layer 3: Known small (hand-checked) --");
        apply_reset();
        fill_zero();
        Q[0][0] = 16'd256; Q[0][1] = 16'd512;   // 1.0, 2.0
        K[0][0] = 16'd256; K[0][1] = 16'd256;   // 1.0, 1.0
        run_and_check("Known small (S[0][0] should be 768)");
        $display("    -> S[0][0] = %0d (expect 768)", s_out[0][0]);

        // ---- Layer 4: constrained random (CRT) ----
        $display("\n-- Layer 4: Constrained random --");
        for (int t = 0; t < 15; t++) begin
            apply_reset();
            fill_random();
            run_and_check($sformatf("Random set %0d", t));
        end

        // ---- Layer 5: back-to-back (no manual reset between) ----
        // The module auto-clears on !busy, so two starts in a row must each
        // produce a correct, independent result.
        $display("\n-- Layer 5: Back-to-back --");
        apply_reset();
        fill_random();
        run_and_check("Back-to-back: first");
        fill_random();               // new data, NO apply_reset()
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
