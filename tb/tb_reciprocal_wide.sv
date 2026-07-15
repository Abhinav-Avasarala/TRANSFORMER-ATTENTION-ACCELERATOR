// tb_reciprocal_wide.sv
//
// Tests reciprocal_nr in the EXACT configuration softmax_unit uses
// (OUT_WIDTH=18, OUT_FRAC=16) -- which tb_reciprocal_nr does NOT cover,
// since that one only exercises the default Q8.8 output config. That gap
// is why a wide-config bug reached the end-to-end test undetected.
//
// Part 1: directed cases, printed raw, so we can see exactly what the
//         module returns for the values softmax actually feeds it.
// Part 2: exhaustive sweep of the row_sum input range (d real in [1,64],
//         i.e. Q8.8 raw [256, 16384]) against a true 1/d reference.

`timescale 1ns/1ps

module tb_reciprocal_wide;

    localparam int DATA_WIDTH = 16;   // d_in: Q8.8
    localparam int FRAC_BITS  = 8;
    localparam int OUT_WIDTH  = 18;   // r_out: Q?.16  <-- softmax's config
    localparam int OUT_FRAC   = 16;
    localparam int CLK_PERIOD = 10;

    logic                  clk, rst_n, start, done;
    logic [DATA_WIDTH-1:0] d_in;
    logic [OUT_WIDTH-1:0]  r_out;

    reciprocal_nr #(
        .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .OUT_WIDTH(OUT_WIDTH),   .OUT_FRAC(OUT_FRAC)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .d_in(d_in), .r_out(r_out), .done(done)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    task automatic apply_reset();
        @(negedge clk);
        rst_n = 0; start = 0; d_in = '0;
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
    endtask

    task automatic run_recip(input logic [DATA_WIDTH-1:0] d_val,
                              output logic [OUT_WIDTH-1:0] r_val);
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
        if (!done) $fatal(1, "TIMEOUT: done never asserted for d=%0d", d_val);
        r_val = r_out;
    endtask

    initial begin
        logic [OUT_WIDTH-1:0] r;
        real  d_real, r_real, true_r;
        int   expected;
        int   fails;
        real  max_rel;
        int   worst_d;

        $display("==================================================");
        $display(" reciprocal_nr WIDE config (OUT_WIDTH=%0d, OUT_FRAC=%0d)", OUT_WIDTH, OUT_FRAC);
        $display("==================================================");

        apply_reset();

        // ---- Part 1: directed cases from the failing softmax rows ----
        $display("\n-- Directed cases (raw values) --");
        $display("   d_raw  d_real    r_out   expected   r_real    true");
        foreach_dir : begin
            int dvals [4];
            dvals = '{295, 16384, 256, 512};
            for (int i = 0; i < 4; i++) begin
                run_recip(DATA_WIDTH'(dvals[i]), r);
                d_real   = real'(dvals[i]) / 256.0;
                true_r   = 1.0 / d_real;
                expected = $rtoi(true_r * real'(1 << OUT_FRAC) + 0.5);
                r_real   = real'(r) / real'(1 << OUT_FRAC);
                $display("   %5d  %7.4f  %6d  %6d   %8.6f  %8.6f  %s",
                         dvals[i], d_real, r, expected, r_real, true_r,
                         (r == OUT_WIDTH'(expected)) ? "" : "  <-- MISMATCH");
            end
        end

        // ---- Part 2: exhaustive sweep over softmax's actual d range ----
        $display("\n-- Exhaustive sweep, d_raw = 256 .. 16384 (row_sum 1.0 .. 64.0) --");
        fails   = 0;
        max_rel = 0.0;
        worst_d = 0;
        for (int d = 256; d <= 16384; d++) begin
            run_recip(DATA_WIDTH'(d), r);
            d_real = real'(d) / 256.0;
            true_r = 1.0 / d_real;
            r_real = real'(r) / real'(1 << OUT_FRAC);
            if (((r_real - true_r) / true_r) > max_rel ||
                ((true_r - r_real) / true_r) > max_rel) begin
                max_rel = ((r_real > true_r) ? (r_real - true_r) : (true_r - r_real)) / true_r;
                worst_d = d;
            end
            // flag anything worse than ~2 LSB relative at the small end
            if (((r_real > true_r) ? (r_real - true_r) : (true_r - r_real)) >
                 (3.0 / real'(1 << OUT_FRAC))) begin
                fails++;
                if (fails <= 10)
                    $display("   [BAD] d=%0d (%.4f): got %.6f, true %.6f",
                             d, d_real, r_real, true_r);
            end
        end

        $display("\n==================================================");
        $display(" Worst relative error: %.3e at d_raw=%0d", max_rel, worst_d);
        $display(" Values off by >3 LSB: %0d / %0d", fails, 16384-256+1);
        $display(" RESULT: %s", (fails == 0) ? "PASS" : "FAIL");
        $display("==================================================");
        $finish;
    end

    initial begin
        #200_000_000;
        $display("ERROR: global simulation timeout");
        $finish;
    end

endmodule
