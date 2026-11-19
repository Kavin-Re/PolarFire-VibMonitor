// Self-checking testbench. Run from the rtl/ directory (vectors/ is read relative to cwd).
//   python3 gen_vectors.py
//   iverilog -g2005 -o sim approx_mul.v mac_array.v tb_mac_array.v && vvp sim
// 1) exhaustive 255x255 multiplier check for exact / trunc4 / trunc6 / drum4 / drum3
// 2) dot-product (mac_array, 8 lanes, 64 elements) trials for exact / trunc4 / drum4
`timescale 1ns/1ps
module tb_mac_array;
    localparam NV = 65025, ND = 12, LEN = 64, LANES = 8, AW = 24;

    reg  [7:0]  a, w;
    wire [15:0] pe, pt4, pt6, pd4, pd3;
    approx_mul #(.MODE(0), .K(0)) u_e  (.a(a), .w(w), .p(pe));
    approx_mul #(.MODE(1), .K(4)) u_t4 (.a(a), .w(w), .p(pt4));
    approx_mul #(.MODE(1), .K(6)) u_t6 (.a(a), .w(w), .p(pt6));
    approx_mul #(.MODE(2), .K(4)) u_d4 (.a(a), .w(w), .p(pd4));
    approx_mul #(.MODE(2), .K(3)) u_d3 (.a(a), .w(w), .p(pd3));

    reg [95:0] mv [0:NV-1];
    reg [7:0]  xm [0:ND*LEN-1];
    reg [LANES*8-1:0]  wm [0:ND*LEN-1];
    reg [LANES*AW-1:0] bm [0:ND-1];
    reg [LANES*AW-1:0] ee [0:ND-1];
    reg [LANES*AW-1:0] et [0:ND-1];
    reg [LANES*AW-1:0] ed [0:ND-1];

    reg clk = 0, rst = 1, en = 0, clear = 0;
    reg  [7:0] x;
    reg  [LANES*8-1:0]  w_flat;
    reg  [LANES*AW-1:0] b_flat;
    wire [LANES*AW-1:0] acc_e, acc_t, acc_d;
    mac_array #(.LANES(LANES), .MODE(0), .K(0)) m_e (clk, rst, en, clear, x, w_flat, b_flat, acc_e);
    mac_array #(.LANES(LANES), .MODE(1), .K(4)) m_t (clk, rst, en, clear, x, w_flat, b_flat, acc_t);
    mac_array #(.LANES(LANES), .MODE(2), .K(4)) m_d (clk, rst, en, clear, x, w_flat, b_flat, acc_d);
    always #5 clk = ~clk;

    integer i, t, k, errs, checks;
    reg [95:0] v;
    initial begin
        errs = 0; checks = 0;
        $readmemh("vectors/mul_vectors.hex", mv);
        $readmemh("vectors/dot_x.hex", xm);
        $readmemh("vectors/dot_w.hex", wm);
        $readmemh("vectors/dot_b.hex", bm);
        $readmemh("vectors/dot_e_exact.hex", ee);
        $readmemh("vectors/dot_e_t4.hex", et);
        $readmemh("vectors/dot_e_d4.hex", ed);
        for (i = 0; i < NV; i = i + 1) begin
            v = mv[i];
            a = v[95:88]; w = v[87:80];
            #1;
            checks = checks + 5;
            if (pe  !== v[79:64]) begin errs = errs + 1; if (errs < 10) $display("MISMATCH exact a=%0d w=%0d got %0d exp %0d", $signed(a), $signed(w), $signed(pe), $signed(v[79:64])); end
            if (pt4 !== v[63:48]) begin errs = errs + 1; if (errs < 10) $display("MISMATCH trunc4 a=%0d w=%0d got %0d exp %0d", $signed(a), $signed(w), $signed(pt4), $signed(v[63:48])); end
            if (pt6 !== v[47:32]) begin errs = errs + 1; if (errs < 10) $display("MISMATCH trunc6 a=%0d w=%0d got %0d exp %0d", $signed(a), $signed(w), $signed(pt6), $signed(v[47:32])); end
            if (pd4 !== v[31:16]) begin errs = errs + 1; if (errs < 10) $display("MISMATCH drum4 a=%0d w=%0d got %0d exp %0d", $signed(a), $signed(w), $signed(pd4), $signed(v[31:16])); end
            if (pd3 !== v[15:0])  begin errs = errs + 1; if (errs < 10) $display("MISMATCH drum3 a=%0d w=%0d got %0d exp %0d", $signed(a), $signed(w), $signed(pd3), $signed(v[15:0])); end
        end
        $display("multiplier section: %0d checks, %0d errors so far", checks, errs);
        // dot-product trials
        repeat (2) @(negedge clk);
        rst = 0;
        for (t = 0; t < ND; t = t + 1) begin
            b_flat = bm[t];
            for (k = 0; k < LEN; k = k + 1) begin
                @(negedge clk);
                x = xm[t*LEN+k]; w_flat = wm[t*LEN+k];
                clear = (k == 0); en = 1;
            end
            @(negedge clk);
            en = 0; clear = 0;
            @(negedge clk);
            checks = checks + 3;
            if (acc_e !== ee[t]) begin errs = errs + 1; $display("DOT MISMATCH exact trial %0d", t); end
            if (acc_t !== et[t]) begin errs = errs + 1; $display("DOT MISMATCH trunc4 trial %0d", t); end
            if (acc_d !== ed[t]) begin errs = errs + 1; $display("DOT MISMATCH drum4 trial %0d", t); end
        end
        if (errs == 0) $display("TB PASS: %0d checks, 0 errors", checks);
        else           $display("TB FAIL: %0d checks, %0d errors", checks, errs);
        $finish;
    end
endmodule
