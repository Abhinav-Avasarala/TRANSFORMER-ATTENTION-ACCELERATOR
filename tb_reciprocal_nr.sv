// tb_reciprocal_nr.sv
//
// Sweeps the ENTIRE valid input range of reciprocal_nr (D = 256..65535,
// per its documented precondition D >= 256) and checks the result
// against a true floating-point 1/D reference.
//
// Unlike tb_qk_systolic.sv/tb_av_multiply.sv, this is NOT a bit-exact
// check -- reciprocal_nr is an ITERATIVE APPROXIMATION (seed + 3 Newton-
// Raphson steps), so a small amount of error vs. the true mathematical
// answer is expected. Instead this checks that error stays within
// ERROR_TOLERANCE for every single D, and reports the worst case found.
//
// This is an exhaustive sweep, not a handful of directed cases, on
// purpose: the seed bug fixed earlier (coarse msb_pos-only seed causing
// collapse to near-zero for D near the top of a power-of-two bucket)
// only showed up at specific D values. A few directed test cases would
// not have caught it -- a full sweep would have caught it immediately.

`timescale 1ns/1ps

module tb_reciprocal_nr;

    localparam int DATA_WIDTH      = 16;
    localparam int FRAC_BITS       = 8;
    localparam int CLK_PERIOD      = 10;
    localparam int ERROR_TOLERANCE = 2;   // max acceptable |raw error|, in Q8.8 LSBs

    logic                  clk, rst_n, start, done;
    logic [DATA_WIDTH-1:0] d_in, r_out;

    reciprocal_nr #(
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .d_in(d_in), .r_out(r_out), .done(done)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    int tests_run, tests_passed, tests_failed;
    int max_abs_error, max_abs_error_d;

    task automatic apply_reset();
        @(negedge clk);
        rst_n = 0; start = 0; d_in = '0;
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
    endtask

    // Run reciprocal_nr on one D value, return r_out.
    task automatic run_recip(input logic [DATA_WIDTH-1:0] d_val,
                              output logic [DATA_WIDTH-1:0] r_val);
        int guard;
        @(negedge clk);
        d_in  = d_val;
        start = 1;
        @(negedge clk);
        start = 0;
        guard = 0;
        while (!done && guard < 50) begin
            @(posedge clk);
            guard++;
        end
        if (!done) $fatal(1, "TIMEOUT: done never asserted for D=%0d", d_val);
        r_val = r_out;
    endtask

    initial begin
        real                    d_real, true_recip_real;
        int                     expected_raw, actual_raw, err;
        logic [DATA_WIDTH-1:0]  r_val;

        $display("==================================================");
        $display(" reciprocal_nr testbench -- full sweep D=256..65535");
        $display("==================================================");

        apply_reset();
        tests_run = 0; tests_passed = 0; tests_failed = 0;
        max_abs_error = 0; max_abs_error_d = 0;

        for (int d = 256; d <= 65535; d++) begin
            run_recip(d[DATA_WIDTH-1:0], r_val);

            d_real          = real'(d) / 256.0;
            true_recip_real = 1.0 / d_real;
            expected_raw    = $rtoi(true_recip_real * 256.0 + 0.5);  // round to nearest

            actual_raw = int'(r_val);
            err        = actual_raw - expected_raw;
            if (err < 0) err = -err;

            tests_run++;
            if (err > max_abs_error) begin
                max_abs_error   = err;
                max_abs_error_d = d;
            end

            if (err <= ERROR_TOLERANCE) begin
                tests_passed++;
            end else begin
                tests_failed++;
                if (tests_failed <= 10)
                    $display("  [FAIL] D=%0d (%.4f): got %0d, expected %0d (err=%0d)",
                             d, d_real, actual_raw, expected_raw, err);
            end
        end

        $display("\n==================================================");
        $display(" SUMMARY: %0d/%0d passed  (%0d failed)",
                 tests_passed, tests_run, tests_failed);
        $display(" Worst-case error: %0d raw units, at D=%0d (real D=%.4f)",
                 max_abs_error, max_abs_error_d, real'(max_abs_error_d) / 256.0);
        $display(" RESULT : %s",
                 (tests_failed == 0) ? "ALL TESTS PASSED" : "FAILED");
        $display("==================================================");
        $finish;
    end

    // ---- Watchdog ----
    // 65280 values * ~6 cycles/run * 10ns ~= 4ms of real work; generous margin.
    initial begin
        #50_000_000;
        $display("ERROR: global simulation timeout");
        $finish;
    end

endmodule
