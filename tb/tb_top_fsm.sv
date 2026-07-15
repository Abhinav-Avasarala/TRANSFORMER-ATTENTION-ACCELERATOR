// tb_top_fsm.sv
//
// End-to-end integration test for top_fsm: Q,K,V -> qk_systolic ->
// softmax_unit -> av_multiply -> O, driven by top_fsm's own sequencing
// and skew-feed generation (no testbench-side feeding needed -- that's
// the whole point of top_fsm existing).
//
// This is the FIRST testbench in this project that checks the actual
// project pass/fail criterion, in the project doc's own terms: "hardware
// output O must match a NumPy golden reference (float32) within a max
// absolute error of 0.01 per element." Every earlier testbench validated
// one stage in isolation, either bit-exact against an internal model or
// with a raw-Q8.8-LSB tolerance; this one compares real (dequantized) O
// against scripts/gen_golden.py's true float32 O_float, in the same real-
// valued absolute-error terms the doc states the bar in.
//
// Data source: reuses the EXISTING golden/ directory (q.hex, k.hex,
// v.hex, o_float_expected.txt) from earlier gen_golden.py runs -- no
// regeneration needed, since N=D=64 here match those defaults and
// float_attention() in gen_golden.py already included the /sqrt(d_k)
// scaling from the start (it was the RTL that was missing it, not the
// golden model).
//
// o_float_expected.txt is read directly as real numbers via $fscanf
// (gen_golden.py wrote it with np.savetxt, plain space/newline-separated
// floats, row-major) rather than converting it to a hex/fixed-point file
// first -- comparing against the true float reference directly is a more
// faithful match to the doc's own stated criterion than quantizing it
// first would be.
//
// GOLDEN_DIR is an absolute path for the same reason as every other
// golden-data-reading testbench in this project: Vivado's xsim runs from
// its own simulation folder, not this repo.
//
// NOTE: a single full run costs ~13,100 cycles (per top_fsm.sv's own
// latency estimate: ~194 QK + ~12,700 softmax + ~194 AV) -- this
// testbench runs the existing golden dataset ONCE, not multiple random
// iterations, given that cost and that only one golden dataset currently
// exists on disk.

`timescale 1ns/1ps

module tb_top_fsm;

    localparam int N          = 64;
    localparam int D          = 64;
    localparam int DATA_WIDTH = 16;
    localparam int ACC_WIDTH  = 32;
    localparam int FRAC_BITS  = 8;
    localparam int CLK_PERIOD = 10;
    localparam real MAX_ERR_TOLERANCE = 0.01;   // doc's stated correctness bar

    localparam string GOLDEN_DIR =
        "C:/Users/abhin/coding_projects/TRANSFORMER-ATTENTION-ACCELERATOR/golden";

    // ---- DUT signals ----
    logic                  clk, rst_n, start, done;
    logic [DATA_WIDTH-1:0] Q [0:N-1][0:D-1];
    logic [DATA_WIDTH-1:0] K [0:N-1][0:D-1];
    logic [DATA_WIDTH-1:0] V [0:N-1][0:D-1];
    logic [DATA_WIDTH-1:0] O [0:N-1][0:D-1];

    // ---- Golden data ----
    logic [DATA_WIDTH-1:0] Q_flat [0:N*D-1];
    logic [DATA_WIDTH-1:0] K_flat [0:N*D-1];
    logic [DATA_WIDTH-1:0] V_flat [0:N*D-1];
    real                   O_expected [0:N-1][0:D-1];

    top_fsm #(
        .N(N), .D(D), .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH), .FRAC_BITS(FRAC_BITS),
        // Absolute path: xsim runs from its own sim folder, so the relative
        // default would silently load an all-zero exp LUT (the WARNING +
        // all-zero O this testbench first produced before this override).
        .EXP_LUT_FILE("C:/Users/abhin/coding_projects/TRANSFORMER-ATTENTION-ACCELERATOR/exp_lut.hex")
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .Q(Q), .K(K), .V(V), .O(O), .done(done)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    function automatic real q8_8_to_real(input logic [DATA_WIDTH-1:0] v);
        return real'($signed(v)) / 256.0;
    endfunction

    // =========================================================================
    // Load Q, K, V (hex, from gen_golden.py) and O_expected (plain text
    // floats, from gen_golden.py's np.savetxt output).
    // =========================================================================
    task automatic load_golden_data();
        int fd, r;

        fd = $fopen({GOLDEN_DIR, "/q.hex"}, "r");
        if (fd == 0)
            $fatal(1, "Could not open %s/q.hex -- run: python scripts/gen_golden.py --out golden",
                   GOLDEN_DIR);
        $fclose(fd);

        $readmemh({GOLDEN_DIR, "/q.hex"}, Q_flat);
        $readmemh({GOLDEN_DIR, "/k.hex"}, K_flat);
        $readmemh({GOLDEN_DIR, "/v.hex"}, V_flat);

        for (int i = 0; i < N; i++)
            for (int k = 0; k < D; k++) begin
                Q[i][k] = Q_flat[i*D + k];
                K[i][k] = K_flat[i*D + k];
                V[i][k] = V_flat[i*D + k];
            end

        fd = $fopen({GOLDEN_DIR, "/o_float_expected.txt"}, "r");
        if (fd == 0)
            $fatal(1, "Could not open %s/o_float_expected.txt", GOLDEN_DIR);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
                r = $fscanf(fd, "%f", O_expected[i][j]);
        $fclose(fd);
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

    task automatic run_top();
        int guard;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        guard = 0;
        while (!done && guard < 30000) begin
            @(posedge clk);
            guard++;
        end
        if (!done) $fatal(1, "TIMEOUT: done never asserted");
    endtask

    // =========================================================================
    // check: compare DUT O against gen_golden.py's true float32 O_expected,
    // in real (dequantized) terms -- matching the doc's own stated
    // correctness bar directly, not a raw-Q8.8-LSB proxy for it.
    // =========================================================================
    task automatic check();
        real actual_r, err, max_err, sum_err;
        int  mismatches;

        max_err    = 0.0;
        sum_err    = 0.0;
        mismatches = 0;

        for (int i = 0; i < N; i++) begin
            for (int j = 0; j < D; j++) begin
                actual_r = q8_8_to_real(O[i][j]);
                err      = actual_r - O_expected[i][j];
                if (err < 0.0) err = -err;

                sum_err += err;
                if (err > max_err) max_err = err;

                if (err > MAX_ERR_TOLERANCE) begin
                    mismatches++;
                    if (mismatches <= 10)
                        $display("    O[%0d][%0d]: got %.6f, expected %.6f (err=%.6f)",
                                 i, j, actual_r, O_expected[i][j], err);
                end
            end
        end

        $display("\n==================================================");
        $display(" Max Absolute Error : %.6f  (doc target: < %.4f)", max_err, MAX_ERR_TOLERANCE);
        $display(" Mean Absolute Error: %.6f  (doc target: < 0.0050)", sum_err / real'(N*D));
        $display(" Elements exceeding tolerance: %0d / %0d", mismatches, N*D);
        $display(" RESULT: %s", (mismatches == 0) ? "PASS" : "FAIL");
        $display("==================================================");
    endtask

    // =========================================================================
    // Test sequence
    // =========================================================================
    initial begin
        $display("==================================================");
        $display(" top_fsm end-to-end testbench  (N=%0d D=%0d Q8.8)", N, D);
        $display(" comparing against scripts/gen_golden.py's true O_float");
        $display("==================================================");

        load_golden_data();
        apply_reset();
        run_top();
        check();

        $finish;
    end

    // ---- Watchdog ----
    // ~13,100 cycles * 10ns ~= 131us of real work; generous margin.
    initial begin
        #2_000_000;
        $display("ERROR: global simulation timeout");
        $finish;
    end

endmodule
