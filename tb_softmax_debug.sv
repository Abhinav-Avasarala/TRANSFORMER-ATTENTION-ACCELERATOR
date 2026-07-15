// tb_softmax_debug.sv
//
// Diagnostic-only testbench: runs the exact "Peaked" case from
// tb_softmax_unit.sv's Layer 2 (column 0 = 48.0, rest 0.0), and prints
// softmax_unit's INTERNAL signals at the moment they matter, via
// hierarchical reference into `dut`. Purpose: reciprocal_nr is proven
// correct in isolation (tb_reciprocal_wide: d=295 -> 56872, exact), and
// exp_buf[0] should be exactly 65536 (delta=0 -> exp_lut[0]=65536), so
// hand-derivation says a_out[*][0] should be ~0.8678 -- but the real RTL
// produces ~0.8858. This prints row_sum, recip_d (what's actually fed to
// reciprocal_nr), recip_result (what's actually latched back), and
// exp_buf[0], to see directly where the real values diverge from the
// hand-derived ones instead of guessing further.

`timescale 1ns/1ps

module tb_softmax_debug;

    localparam int N          = 64;
    localparam int D          = 64;
    localparam int DATA_WIDTH = 16;
    localparam int FRAC_BITS  = 8;
    localparam int CLK_PERIOD = 10;

    logic                  clk, rst_n, start, done;
    logic [DATA_WIDTH-1:0] S [0:N-1][0:N-1];
    logic [DATA_WIDTH-1:0] a_out [0:N-1][0:N-1];

    softmax_unit #(
        .N(N), .D(D), .DATA_WIDTH(DATA_WIDTH), .FRAC_BITS(FRAC_BITS),
        .EXP_LUT_FILE("C:/Users/abhin/coding_projects/TRANSFORMER-ATTENTION-ACCELERATOR/exp_lut.hex")
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .s_in(S), .a_out(a_out), .done(done)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ---- watch RECIP_KICK / RECIP_WAIT / the norm multiply directly ----
    always @(posedge clk) begin
        if (dut.state == dut.RECIP_KICK)
            $display("[t=%0t] RECIP_KICK   row_sum=%0d (real=%.6f)  recip_d=%0d (real=%.6f)",
                     $time, dut.row_sum, real'(dut.row_sum)/65536.0,
                     dut.recip_d, real'(dut.recip_d)/256.0);

        if (dut.recip_done)
            $display("[t=%0t] recip_done   recip_r=%0d (real=%.6f)",
                     $time, dut.recip_r, real'(dut.recip_r)/65536.0);

        if (dut.state == dut.RECIP_WAIT && dut.recip_done)
            $display("[t=%0t] latching recip_result <= recip_r = %0d", $time, dut.recip_r);

        if (dut.state == dut.NORM_SCAN && dut.col == 0)
            $display("[t=%0t] NORM_SCAN col=0  exp_buf[0]=%0d  recip_result=%0d (real=%.6f)  norm_prod=%0d  a_val=%0d",
                     $time, dut.exp_buf[0], dut.recip_result, real'(dut.recip_result)/65536.0,
                     dut.norm_prod, dut.a_val);
    end

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
        if (!done) $fatal(1, "TIMEOUT");
    endtask

    initial begin
        $display("==================================================");
        $display(" softmax_unit internal-signal debug -- Peaked case");
        $display("==================================================");

        apply_reset();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                S[i][j] = '0;
        for (int i = 0; i < N; i++)
            S[i][0] = 16'sd12288;   // 48.0 in Q8.8

        run_softmax();

        $display("\nFinal a_out[0][0] = %0d (real=%.6f)", a_out[0][0], real'(a_out[0][0])/32768.0);
        $display("Expected ~0.864931 (see tb_softmax_unit Layer 2)");
        $finish;
    end

    initial begin
        #2_000_000;
        $display("ERROR: timeout");
        $finish;
    end

endmodule
