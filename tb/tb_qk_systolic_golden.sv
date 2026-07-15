// tb_qk_systolic_golden.sv
//
// Golden-vector cross-check for qk_systolic against an INDEPENDENT
// reference: scripts/gen_golden.py (NumPy, not this repo's SystemVerilog).
//
// Unlike tb_qk_systolic.sv's internal golden_s() -- which mirrors the
// RTL's own arithmetic and therefore can never disagree with a
// correctly-wired-but-conceptually-wrong RTL -- this compares against
// a from-scratch Python reimplementation (fixed_qkt() in gen_golden.py).
// Passing THIS test is a much stronger correctness signal.
//
// Workflow:
//   1. python scripts/gen_golden.py --n 64 --d 64 --out golden
//   2. Run this testbench.
//
// GOLDEN_DIR is an ABSOLUTE path on purpose: the Vivado project that
// simulates this file lives in a different folder than this repo, and
// Vivado's xsim runs with its working directory set to its own
// simulation run folder (e.g. <project>.sim/sim_1/behav/xsim/), not
// this repo. An absolute path means "run gen_golden.py, then just
// simulate" with no copying of .hex files required.

`timescale 1ns/1ps

module tb_qk_systolic_golden;

    localparam int    N          = 64;
    localparam int    D          = 64;
    localparam int    DATA_WIDTH = 16;
    localparam int    ACC_WIDTH  = 32;
    localparam int    FRAC_BITS  = 8;
    localparam int    CLK_PERIOD = 10;
    localparam string GOLDEN_DIR =
        "C:/Users/abhin/coding_projects/TRANSFORMER-ATTENTION-ACCELERATOR/golden";

    localparam int FEED_CYCLES = N + D - 1;

    // ---- DUT signals ----
    logic                  clk;
    logic                  rst_n;
    logic                  start;
    logic [DATA_WIDTH-1:0] q_in  [0:N-1];
    logic [DATA_WIDTH-1:0] k_in  [0:N-1];
    logic [DATA_WIDTH-1:0] s_out [0:N-1][0:N-1];
    logic                  done;

    // ---- Golden vectors, flat (row-major, matches numpy .flatten()) ----
    logic [DATA_WIDTH-1:0] Q_flat [0:N*D-1];
    logic [DATA_WIDTH-1:0] K_flat [0:N*D-1];
    logic [DATA_WIDTH-1:0] S_flat [0:N*N-1];

    function automatic logic [DATA_WIDTH-1:0] Qv(input int i, input int k);
        return Q_flat[i*D + k];
    endfunction

    function automatic logic [DATA_WIDTH-1:0] Kv(input int j, input int k);
        return K_flat[j*D + k];
    endfunction

    function automatic logic [DATA_WIDTH-1:0] Sexp(input int i, input int j);
        return S_flat[i*N + j];
    endfunction

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

    // ---- Reset (same contract as tb_qk_systolic.sv) ----
    task automatic apply_reset();
        @(negedge clk);
        rst_n = 0; start = 0;
        for (int i = 0; i < N; i++) begin q_in[i] = '0; k_in[i] = '0; end
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
    endtask

    // ---- Run one matmul, feeding from the loaded golden Q/K ----
    task automatic run_qk();
        int guard;

        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        for (int t = 0; t < FEED_CYCLES; t++) begin
            for (int i = 0; i < N; i++)
                q_in[i] = (t >= i && (t - i) < D) ? Qv(i, t - i) : '0;
            for (int j = 0; j < N; j++)
                k_in[j] = (t >= j && (t - j) < D) ? Kv(j, t - j) : '0;
            @(negedge clk);
        end

        for (int i = 0; i < N; i++) begin q_in[i] = '0; k_in[i] = '0; end

        guard = 0;
        while (!done && guard < 4*(N + D)) begin
            @(posedge clk);
            guard++;
        end
        if (!done) $fatal(1, "TIMEOUT: done never asserted");
    endtask

    initial begin
        int fd;
        int errors;

        $display("==================================================");
        $display(" qk_systolic GOLDEN cross-check (vs scripts/gen_golden.py)");
        $display("==================================================");

        fd = $fopen({GOLDEN_DIR, "/q.hex"}, "r");
        if (fd == 0)
            $fatal(1, "Could not open %s/q.hex -- run: python scripts/gen_golden.py --out %s  (see GOTCHA comment re: xsim working directory)",
                   GOLDEN_DIR, GOLDEN_DIR);
        $fclose(fd);

        $readmemh({GOLDEN_DIR, "/q.hex"}, Q_flat);
        $readmemh({GOLDEN_DIR, "/k.hex"}, K_flat);
        $readmemh({GOLDEN_DIR, "/s_fixed_expected.hex"}, S_flat);

        apply_reset();
        run_qk();

        errors = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) begin
                if (s_out[i][j] !== Sexp(i, j)) begin
                    if (errors < 10)
                        $display("  S[%0d][%0d]: got %0d, golden %0d",
                                 i, j, $signed(s_out[i][j]), $signed(Sexp(i, j)));
                    errors++;
                end
            end

        $display("\n==================================================");
        if (errors == 0)
            $display(" RESULT: ALL %0d ELEMENTS MATCH NUMPY GOLDEN MODEL", N*N);
        else
            $display(" RESULT: FAILED (%0d/%0d mismatches)", errors, N*N);
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
