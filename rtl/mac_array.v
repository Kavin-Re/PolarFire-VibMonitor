// LANES parallel MACs sharing one broadcast activation x (output-stationary dense layer
// engine: lane = neuron).  Per cycle with en=1:  acc <= (clear ? bias : acc) + x*w[lane]
// 'clear' marks the first element of a dot product.  Result valid the cycle after the last
// en pulse.  Accumulator is ACC_W-bit signed (no saturation: 24 b holds 448 x 127^2 worst case).
module mac_array #(
    parameter LANES = 8,
    parameter MODE  = 0,
    parameter K     = 4,
    parameter ACC_W = 24
) (
    input                         clk,
    input                         rst,
    input                         en,
    input                         clear,
    input  signed [7:0]           x,
    input  [LANES*8-1:0]          w_flat,
    input  [LANES*ACC_W-1:0]      bias_flat,
    output [LANES*ACC_W-1:0]      acc_flat
);
    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : lane
            wire signed [15:0]       p;
            reg  signed [ACC_W-1:0]  acc;
            wire signed [ACC_W-1:0]  bias = bias_flat[g*ACC_W +: ACC_W];
            approx_mul #(.MODE(MODE), .K(K)) u_mul (.a(x), .w(w_flat[g*8 +: 8]), .p(p));
            always @(posedge clk) begin
                if (rst)      acc <= {ACC_W{1'b0}};
                else if (en)  acc <= (clear ? bias : acc) + p;
            end
            assign acc_flat[g*ACC_W +: ACC_W] = acc;
        end
    endgenerate
endmodule
