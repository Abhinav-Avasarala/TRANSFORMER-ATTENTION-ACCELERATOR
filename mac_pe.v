module mac_pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 16
)(
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  valid_in,
    input  wire [DATA_WIDTH-1:0] a_in,
    input  wire [DATA_WIDTH-1:0] b_in,
    output reg  [ACC_WIDTH-1:0]  sum_out,
    output wire                  valid_out
);

    // Stage 1: latch inputs
    reg [DATA_WIDTH-1:0] reg_a;
    reg [DATA_WIDTH-1:0] reg_b;

    // Stage 2: registered product (2*DATA_WIDTH to hold full product without overflow)
    reg [2*DATA_WIDTH-1:0] reg_product;

    // Internal accumulator — persists across cycles to build the dot product
    reg [ACC_WIDTH-1:0] acc;

    // 3-stage shift register to pipeline valid_in through the same latency as data
    reg [2:0] valid_pipe;

    always @(posedge clk) begin
        if (reset) begin
            reg_a       <= 0;
            reg_b       <= 0;
            reg_product <= 0;
            acc         <= 0;
            sum_out     <= 0;
            valid_pipe  <= 3'b0;
        end else begin
            // Stage 1: latch inputs
            reg_a <= a_in;
            reg_b <= b_in;

            // Stage 2: multiply
            reg_product <= reg_a * reg_b;

            // Stage 3: accumulate into internal register and expose on output
            acc     <= acc + reg_product;
            sum_out <= acc + reg_product;

            // Pipeline valid_in through 3 stages to match data latency
            valid_pipe <= {valid_pipe[1:0], valid_in};
        end
    end

    assign valid_out = valid_pipe[2];

endmodule
