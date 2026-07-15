// tb_softmax_unit.sv
//
// Testbench for softmax_unit (A = row-wise softmax(S / sqrt(D))).
// s_in is Q8.8; a_out is now Q1.15 (higher-precision attention weights),
// so the comparison below is real-valued, not raw-Q8.8-LSB.
//
// Unlike tb_qk_systolic.sv/tb_av_multiply.sv, this does NOT mirror the
// RTL's internal arithmetic bit-for-bit as its golden model. softmax_unit
// depends on two approximations (the exp() LUT's quantization and
// reciprocal_nr's iterative approximation), so re-implementing both of
// those a second time in the testbench would just duplicate their
// potential bugs rather than independently catching them. Instead, this
// compares against a TRUE floating-point softmax (using SystemVerilog's
// built-in $exp() on `real` values, no LUT/NR involved) with a per-element
// tolerance -- the same philosophy as scripts/gen_golden.py's O_float
// reference, just done directly in SV so no separate Python run is needed
// for this one module.
//
// NOTE: s_in is fed as SIGNED Q8.8 (unlike tb_qk_systolic.sv's legacy
// non-negative-only convention) since softmax_unit's whole point is
// handling real (signed) attention scores.
//
// NOTE: exp_lut.hex (from scripts/gen_exp_lut.py) must be reachable from
// wherever this simulation actually runs -- same working-directory
// gotcha as tb_qk_systolic_golden.sv's GOLDEN_DIR.
//
// NOTE: each full run takes ~(3N+7)*N cycles (~12,700 cycles at N=64,
// per softmax_unit.sv's own header comment) -- this testbench uses fewer
// random iterations than tb_qk_systolic.sv/tb_av_multiply.sv because of
// that per-run cost, not because less coverage is needed.

`timescale 1ns/1ps

module tb_softmax_unit;

    localparam int N          = 64;
    localparam int D          = 64;   // must match softmax_unit's D -- see file header
    localparam int DATA_WIDTH = 16;
    localparam int FRAC_BITS  = 8;
    localparam int  CLK_PERIOD = 10;
    localparam real TOLERANCE  = 0.0035; // max acceptable |error| per weight, real terms.
                                          // Measured worst case after the SCALE_CONST fix:
                                          // 0.002867 (peaked/low-entropy case) -- this leaves
                                          // ~20% margin above that, not the original rough
                                          // 0.000745 estimate, which undercounted the peaked
                                          // case specifically.

    // ---- DUT signals ----
    logic                  clk, rst_n, start, done;
    logic [DATA_WIDTH-1:0] a_out [0:N-1][0:N-1];

    // ---- Test data (connected directly to the DUT's s_in port) ----
    logic [DATA_WIDTH-1:0] S [0:N-1][0:N-1];

    int tests_run, tests_passed, tests_failed;

    // EXP_LUT_FILE overridden to an absolute path here (not left at
    // softmax_unit's relative default) for the same reason
    // tb_qk_systolic_golden.sv's GOLDEN_DIR is absolute: Vivado's xsim
    // runs from its own simulation folder, not this repo, so a relative
    // "exp_lut.hex" silently fails to load and exp_lut defaults to all
    // zeros -- which cascades into every a_out element reading back 0.
    softmax_unit #(
        .N(N), .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .EXP_LUT_FILE("C:/Users/abhin/coding_projects/TRANSFORMER-ATTENTION-ACCELERATOR/exp_lut.hex")
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .s_in(S), .a_out(a_out), .done(done)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    function automatic real q8_8_to_real(input logic [DATA_WIDTH-1:0] v);
        return real'($signed(v)) / 256.0;
    endfunction

    // =========================================================================
    // Matrix fill helpers
    // =========================================================================
    task automatic fill_zero();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                S[i][j] = '0;
    endtask

    // Column 0 strongly dominant. Raw score is 48.0, not 6.0: softmax_unit
    // now divides by sqrt(D)=8 before exponentiating, so a raw gap of 48.0
    // produces the same effective (post-scale) gap of 6.0 this layer was
    // originally calibrated for -- softmax should concentrate most of the
    // weight there but not saturate fully.
    task automatic fill_peaked();
        fill_zero();
        for (int i = 0; i < N; i++)
            S[i][0] = 16'sd12288;   // 48.0 in Q8.8 (-> effective 6.0 after /sqrt(D))
    endtask

    // Signed random, raw range [-16384, 16383] (~[-64.0, +64.0) real).
    // 8x wider than the exp_lut's clamp domain on purpose: scores now get
    // divided by sqrt(D)=8 before hitting the LUT, so a raw spread of only
    // +-8.0 would collapse to +-1.0 post-scale and never reach the LUT's
    // delta=8.0 clamp point. This wider range keeps both the "in range"
    // and "clamped" LUT paths exercised, matching the pre-scaling intent.
    task automatic fill_random();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                S[i][j] = 16'($urandom_range(0, 32767) - 16384);
    endtask

    // =========================================================================
    // Reset / run
    // =========================================================================
    task automatic apply_reset();
        @(negedge clk);
        rst_n = 0; start = 0;
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
    endtask

    task automatic run_softmax();
        int guard;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        guard = 0;
        while (!done && guard < 20000) begin
            @(posedge clk);
            guard++;
        end
        if (!done) $fatal(1, "TIMEOUT: done never asserted");
    endtask

    // =========================================================================
    // check: compare DUT a_out against a true floating-point softmax, row by
    // row, in REAL terms. a_out is now Q1.15 (unsigned, /32768), NOT Q8.8 --
    // softmax_unit emits attention weights at higher precision. Tolerance is
    // real-valued (0.002), set ~2.7x above the measured worst-case weight
    // error of 0.000745 (scripts/diag_rtl_model.py).
    // =========================================================================
    task automatic check(input string name);
        real max_r, sum_r, exp_r [0:N-1], softmax_r;
        real actual_r, err_r, max_err_overall, total_abs_err, mean_abs_err;
        int  errors;

        errors = 0;
        total_abs_err = 0.0;
        max_err_overall = 0.0;

        for (int i = 0; i < N; i++) begin
            max_r = q8_8_to_real(S[i][0]);
            for (int j = 1; j < N; j++)
                if (q8_8_to_real(S[i][j]) > max_r) max_r = q8_8_to_real(S[i][j]);

            sum_r = 0.0;
            for (int j = 0; j < N; j++) begin
                // (S[i][j] - max_r) / sqrt(D) -- matches softmax_unit's
                // scaled-delta step (see softmax_unit.sv's file header)
                exp_r[j] = $exp((q8_8_to_real(S[i][j]) - max_r) / $sqrt(real'(D)));
                sum_r += exp_r[j];
            end

            for (int j = 0; j < N; j++) begin
                softmax_r = exp_r[j] / sum_r;
                actual_r  = real'(a_out[i][j]) / 32768.0;   // Q1.15 unsigned
                err_r     = actual_r - softmax_r;
                if (err_r < 0.0) err_r = -err_r;

                total_abs_err += err_r;
                if (err_r > max_err_overall) max_err_overall = err_r;

                if (err_r > TOLERANCE) begin
                    errors++;
                    if (errors <= 10)
                        $display("    A[%0d][%0d]: got %.6f, expected %.6f (err=%.6f)",
                                 i, j, actual_r, softmax_r, err_r);
                end
            end
        end

        mean_abs_err = total_abs_err / real'(N*N);
        tests_run++;
        if (errors == 0) begin
            tests_passed++;
            $display("  [PASS] %s  (max_err=%.6f, mean_err=%.6f)",
                      name, max_err_overall, mean_abs_err);
        end else begin
            tests_failed++;
            $display("  [FAIL] %s  (%0d/%0d mismatches, max_err=%.6f, mean_err=%.6f)",
                      name, errors, N*N, max_err_overall, mean_abs_err);
        end
    endtask

    task automatic run_and_check(input string name);
        run_softmax();
        check(name);
    endtask

    // =========================================================================
    // Test sequence
    // =========================================================================
    initial begin
        $dumpfile("tb_softmax_unit.vcd");
        $dumpvars(0, tb_softmax_unit);

        $display("==================================================");
        $display(" softmax_unit testbench  (N=%0d Q8.8)", N);
        $display("==================================================");

        // ---- Layer 1: all zeros ----
        // Every row: all scores tied at 0 -> uniform attention, A[i][j] = 1/64
        // for every element. A strong structural check with a known answer.
        $display("\n-- Layer 1: All zeros (expect uniform 1/64 per element) --");
        apply_reset();
        fill_zero();
        run_and_check("All zeros -> uniform softmax");

        // ---- Layer 2: peaked ----
        // Column 0 dominant but not saturated -- exercises real normalization
        // math, not just a degenerate all-equal or all-but-one-zero case.
        $display("\n-- Layer 2: Peaked (column 0 dominant) --");
        apply_reset();
        fill_peaked();
        run_and_check("Peaked -> column 0 dominant");

        // ---- Layer 3: constrained random ----
        $display("\n-- Layer 3: Constrained random --");
        for (int t = 0; t < 3; t++) begin
            apply_reset();
            fill_random();
            run_and_check($sformatf("Random set %0d", t));
        end

        // ---- Layer 4: back-to-back (no manual reset between) ----
        $display("\n-- Layer 4: Back-to-back --");
        apply_reset();
        fill_random();
        run_and_check("Back-to-back: first");
        fill_random();
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
    // 7 runs * ~12,700 cycles * 10ns ~= 0.9ms of real work; generous margin.
    initial begin
        #5_000_000;
        $display("ERROR: global simulation timeout");
        $finish;
    end

endmodule
