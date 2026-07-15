// tb_bram_if.sv
//
// Testbench for bram_if (serial load of Q/K/V, full-array handoff to
// top_fsm's interface; full-array capture of O, serial drain back out).
//
// Structure:
//   Layer 1  Sequential load             - simple, hand-checkable pattern
//   Layer 2  Sequential drain            - same, for the O path
//   Layer 3  Random load (stress)        - catches indexing bugs statistically
//   Layer 4  Random drain (stress)
//   Layer 5  Back-to-back round trips    - load -> drain -> load -> drain,
//                                          no reset between, verifying the
//                                          state machine cycles correctly
//                                          on its own (realistic multi-pass
//                                          usage)
//
// This is a pure data-integrity test (does every element land in the
// right place, do load_done/out_done fire on the right cycle) -- there's
// no numeric/attention semantics involved, so test data is arbitrary
// 16-bit patterns, not Q8.8-meaningful values.

`timescale 1ns/1ps

module tb_bram_if;

    localparam int N          = 64;
    localparam int D          = 64;
    localparam int DATA_WIDTH = 16;
    localparam int CLK_PERIOD = 10;
    localparam int TOTAL      = N * D;

    logic                  clk, rst_n, start;
    logic [DATA_WIDTH-1:0] q_data, k_data, v_data;
    logic                  data_valid;
    logic [DATA_WIDTH-1:0] Q_out [0:N-1][0:D-1];
    logic [DATA_WIDTH-1:0] K_out [0:N-1][0:D-1];
    logic [DATA_WIDTH-1:0] V_out [0:N-1][0:D-1];
    logic                  load_done;
    logic                  capture_o;
    logic [DATA_WIDTH-1:0] O_in [0:N-1][0:D-1];
    logic [DATA_WIDTH-1:0] out_data;
    logic                  out_valid;
    logic                  out_done;

    logic [DATA_WIDTH-1:0] Q_test [0:N-1][0:D-1];
    logic [DATA_WIDTH-1:0] K_test [0:N-1][0:D-1];
    logic [DATA_WIDTH-1:0] V_test [0:N-1][0:D-1];
    logic [DATA_WIDTH-1:0] O_test [0:N-1][0:D-1];

    int tests_run, tests_passed, tests_failed;

    bram_if #(
        .N(N), .D(D), .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .q_data(q_data), .k_data(k_data), .v_data(v_data), .data_valid(data_valid),
        .Q_out(Q_out), .K_out(K_out), .V_out(V_out), .load_done(load_done),
        .capture_o(capture_o), .O_in(O_in),
        .out_data(out_data), .out_valid(out_valid), .out_done(out_done)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // =========================================================================
    // Fill helpers
    // =========================================================================
    task automatic fill_sequential_qkv();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++) begin
                Q_test[i][j] = 16'(i*D + j);
                K_test[i][j] = 16'(i*D + j) + 16'd1000;
                V_test[i][j] = 16'(i*D + j) + 16'd2000;
            end
    endtask

    task automatic fill_random_qkv();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++) begin
                Q_test[i][j] = $urandom_range(0, 65535);
                K_test[i][j] = $urandom_range(0, 65535);
                V_test[i][j] = $urandom_range(0, 65535);
            end
    endtask

    task automatic fill_sequential_o();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
                O_test[i][j] = 16'(i*D + j) + 16'd3000;
    endtask

    task automatic fill_random_o();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
                O_test[i][j] = $urandom_range(0, 65535);
    endtask

    // =========================================================================
    // Reset
    // =========================================================================
    task automatic apply_reset();
        @(negedge clk);
        rst_n = 0; start = 0; data_valid = 0; capture_o = 0;
        q_data = '0; k_data = '0; v_data = '0;
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
    endtask

    // =========================================================================
    // Load: pulse start, stream Q_test/K_test/V_test in row-major order,
    // wait for load_done. All driving on negedge, matching this project's
    // established convention.
    // =========================================================================
    task automatic load_matrices();
        int guard;
        int r, c;
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        for (int idx = 0; idx < TOTAL; idx++) begin
            r = idx / D;
            c = idx % D;
            data_valid = 1;
            q_data = Q_test[r][c];
            k_data = K_test[r][c];
            v_data = V_test[r][c];
            @(negedge clk);
        end
        data_valid = 0;

        guard = 0;
        while (!load_done && guard < 10) begin
            @(posedge clk);
            guard++;
        end
        if (!load_done) $fatal(1, "TIMEOUT: load_done never asserted");
    endtask

    task automatic check_load(input string name);
        int errors;
        errors = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++) begin
                if (Q_out[i][j] !== Q_test[i][j]) begin
                    if (errors < 5) $display("    Q_out[%0d][%0d]: got %0d, expected %0d",
                                              i, j, Q_out[i][j], Q_test[i][j]);
                    errors++;
                end
                if (K_out[i][j] !== K_test[i][j]) begin
                    if (errors < 5) $display("    K_out[%0d][%0d]: got %0d, expected %0d",
                                              i, j, K_out[i][j], K_test[i][j]);
                    errors++;
                end
                if (V_out[i][j] !== V_test[i][j]) begin
                    if (errors < 5) $display("    V_out[%0d][%0d]: got %0d, expected %0d",
                                              i, j, V_out[i][j], V_test[i][j]);
                    errors++;
                end
            end
        tests_run++;
        if (errors == 0) begin
            tests_passed++;
            $display("  [PASS] %s", name);
        end else begin
            tests_failed++;
            $display("  [FAIL] %s  (%0d mismatches)", name, errors);
        end
    endtask

    // =========================================================================
    // Drain: drive O_in statically, pulse capture_o, then sample out_data/
    // out_valid on each subsequent negedge (half a cycle after the DUT's
    // registered update, so the sampled value is settled -- same discipline
    // as every other testbench in this project).
    // =========================================================================
    task automatic drain_and_check(input string name);
        logic [DATA_WIDTH-1:0] captured [0:N-1][0:D-1];
        int errors;
        int r, c;

        O_in = O_test;
        @(negedge clk); capture_o = 1;
        @(negedge clk); capture_o = 0;

        for (int idx = 0; idx < TOTAL; idx++) begin
            @(negedge clk);
            if (!out_valid) $fatal(1, "out_valid not high during expected drain cycle idx=%0d", idx);
            r = idx / D;
            c = idx % D;
            captured[r][c] = out_data;
        end
        if (!out_done) $fatal(1, "out_done not asserted on final drain cycle");

        errors = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < D; j++)
                if (captured[i][j] !== O_test[i][j]) begin
                    if (errors < 5) $display("    O[%0d][%0d]: got %0d, expected %0d",
                                              i, j, captured[i][j], O_test[i][j]);
                    errors++;
                end

        tests_run++;
        if (errors == 0) begin
            tests_passed++;
            $display("  [PASS] %s", name);
        end else begin
            tests_failed++;
            $display("  [FAIL] %s  (%0d mismatches)", name, errors);
        end
    endtask

    // =========================================================================
    // Test sequence
    // =========================================================================
    initial begin
        $dumpfile("tb_bram_if.vcd");
        $dumpvars(0, tb_bram_if);

        $display("==================================================");
        $display(" bram_if testbench  (N=%0d D=%0d, TOTAL=%0d words)", N, D, TOTAL);
        $display("==================================================");

        // ---- Layer 1: sequential load ----
        $display("\n-- Layer 1: Sequential load --");
        apply_reset();
        fill_sequential_qkv();
        load_matrices();
        check_load("Sequential load -> Q_out/K_out/V_out match");

        // ---- Layer 2: sequential drain ----
        $display("\n-- Layer 2: Sequential drain --");
        fill_sequential_o();
        drain_and_check("Sequential drain -> out stream matches");

        // ---- Layer 3: random load (stress) ----
        $display("\n-- Layer 3: Random load --");
        for (int t = 0; t < 3; t++) begin
            fill_random_qkv();
            load_matrices();
            check_load($sformatf("Random load %0d", t));
        end

        // ---- Layer 4: random drain (stress) ----
        $display("\n-- Layer 4: Random drain --");
        for (int t = 0; t < 3; t++) begin
            fill_random_o();
            drain_and_check($sformatf("Random drain %0d", t));
        end

        // ---- Layer 5: back-to-back round trips, no reset between ----
        $display("\n-- Layer 5: Back-to-back load/drain round trips --");
        for (int t = 0; t < 2; t++) begin
            fill_random_qkv();
            load_matrices();
            check_load($sformatf("Round trip %0d: load", t));
            fill_random_o();
            drain_and_check($sformatf("Round trip %0d: drain", t));
        end

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
        #5_000_000;
        $display("ERROR: global simulation timeout");
        $finish;
    end

endmodule
