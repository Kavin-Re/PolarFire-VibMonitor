// Approximate signed 8x8 multiplier (sign-magnitude, 7-bit magnitudes).
// Operands MUST be in [-127,127] (-128 is not representable in sign-magnitude;
// the Python quantiser clips to +-127).
//   MODE 0 : exact magnitude product
//   MODE 1 : truncated (partial-product bits with column weight < 2^K dropped) + gated constant
//   MODE 2 : DRUM-K (K-bit segment from leading one, LSB forced to 1)
// Bit-exact twin of approx.py (mult_exact / mult_trunc / mult_drum, n=7).
module approx_mul #(parameter MODE = 0, parameter K = 4) (
    input  signed [7:0]  a,
    input  signed [7:0]  w,
    output signed [15:0] p
);
    wire        neg = a[7] ^ w[7];
    wire [7:0]  na  = -a;
    wire [7:0]  nw  = -w;
    wire [6:0]  ma  = a[7] ? na[6:0] : a[6:0];
    wire [6:0]  mw  = w[7] ? nw[6:0] : w[6:0];

    // compensation constant, identical to approx.trunc_comp_const(7, K)
    function integer comp_const;
        input integer k;
        integer i, j, tot;
        begin
            tot = 0;
            for (i = 0; i < 7; i = i + 1)
                for (j = 0; j < 7; j = j + 1)
                    if (i + j < k) tot = tot + (1 << (i + j));
            comp_const = (tot + 2) / 4;
        end
    endfunction

    function [3:0] lod7;
        input [6:0] x;
        integer i;
        begin
            lod7 = 0;
            for (i = 1; i < 7; i = i + 1) if (x[i]) lod7 = i;
        end
    endfunction

    reg [15:0] mag;
    reg [15:0] t;
    reg [6:0]  bm, sega, segb;
    reg [3:0]  la, lb, sha, shb;
    integer    i, s;

    always @* begin
        mag = 16'd0;
        if (MODE == 0) begin
            mag = ma * mw;
        end else if (MODE == 1) begin
            for (i = 0; i < 7; i = i + 1) begin
                s  = (K > i) ? (K - i) : 0;
                bm = (mw >> s) << s;
                t  = {9'b0, bm};
                t  = t << i;
                if (ma[i]) mag = mag + t;
            end
            if (K > 0 && ma != 0 && mw != 0) mag = mag + comp_const(K);
        end else begin
            la   = lod7(ma);
            lb   = lod7(mw);
            sha  = (la > K - 1) ? (la - (K - 1)) : 4'd0;
            shb  = (lb > K - 1) ? (lb - (K - 1)) : 4'd0;
            sega = (sha != 0) ? ((ma >> sha) | 7'd1) : ma;
            segb = (shb != 0) ? ((mw >> shb) | 7'd1) : mw;
            mag  = (sega * segb) << (sha + shb);
        end
    end

    assign p = (neg && mag != 0) ? -mag : mag;
endmodule
