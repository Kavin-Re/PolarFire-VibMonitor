"""Cross-check of the committed golden vectors, runnable with the standard library only (no numpy,
no Verilog simulator).

Two independent pure-integer implementations are compared with rtl/vectors/*.hex:

  * `model_*`  - the algorithms of approx.py (mult_exact / mult_trunc / mult_drum) rewritten for
                 Python ints, i.e. "what the software model means";
  * `rtl_*`    - a statement-by-statement transliteration of rtl/approx_mul.v and rtl/mac_array.v
                 with every register masked to its declared width, i.e. "what the Verilog says".

If both agree with the vectors for all 65,025 operand pairs x 5 designs and for the 12 dot-product
trials, then the RTL source, the Python model and the committed vectors describe the same function.
This is NOT a substitute for simulating the Verilog (rtl/run_sim.sh does that with Icarus Verilog);
it guards against the three drifting apart and runs anywhere Python runs.
"""
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
VEC = os.path.join(HERE, "..", "rtl", "vectors")
N = 7                      # magnitude bits
DESIGNS = [("exact", 0, 0), ("trunc4", 1, 4), ("trunc6", 1, 6), ("drum4", 2, 4), ("drum3", 2, 3)]
LANES, LEN, AW, ND = 8, 64, 24, 12


# ----------------------------------------------------------------- software model (approx.py semantics)
def trunc_comp_const(n, k):
    tot = sum(2 ** (i + j) for i in range(n) for j in range(n) if i + j < k)
    return (tot + 2) // 4


def model_mag(mode, k, a, b):
    if mode == 0:
        return a * b
    if mode == 1:
        out = 0
        for i in range(N):
            s = max(k - i, 0)
            out += ((a >> i) & 1) * (((b >> s) << s) << i)
        if k > 0 and a > 0 and b > 0:
            out += trunc_comp_const(N, k)
        return out

    def seg(x):
        sh = max(x.bit_length() - 1 - (k - 1), 0) if x else 0
        return ((x >> sh) | 1 if sh > 0 else x), sh
    (sa, ha), (sb, hb) = seg(a), seg(b)
    return (sa * sb) << (ha + hb)


def model_signed(mode, k, a, w):
    m = model_mag(mode, k, abs(a), abs(w))
    return -m if (a < 0) != (w < 0) else m


# ----------------------------------------------------------------- transliteration of rtl/approx_mul.v
def rtl_approx_mul(mode, K, a8, w8):
    """a8, w8: raw 8-bit two's-complement operands (0..255). Returns the raw 16-bit output p."""
    neg = ((a8 >> 7) ^ (w8 >> 7)) & 1
    na, nw = (-a8) & 0xFF, (-w8) & 0xFF
    ma = (na if a8 >> 7 else a8) & 0x7F
    mw = (nw if w8 >> 7 else w8) & 0x7F

    def lod7(x):
        r = 0
        for i in range(1, 7):
            if (x >> i) & 1:
                r = i
        return r

    mag = 0
    if mode == 0:
        mag = (ma * mw) & 0xFFFF
    elif mode == 1:
        for i in range(7):
            s = K - i if K > i else 0
            bm = ((mw >> s) << s) & 0x7F              # reg [6:0] bm
            t = (bm << i) & 0xFFFF                    # reg [15:0] t
            if (ma >> i) & 1:
                mag = (mag + t) & 0xFFFF
        if K > 0 and ma != 0 and mw != 0:
            mag = (mag + trunc_comp_const(7, K)) & 0xFFFF
    else:
        la, lb = lod7(ma), lod7(mw)
        sha = (la - (K - 1)) & 0xF if la > K - 1 else 0   # reg [3:0]
        shb = (lb - (K - 1)) & 0xF if lb > K - 1 else 0
        sega = ((ma >> sha) | 1) & 0x7F if sha else ma
        segb = ((mw >> shb) | 1) & 0x7F if shb else mw
        mag = ((sega * segb) << ((sha + shb) & 0xF)) & 0xFFFF   # shift amount is a 4-bit sum
    return (-mag) & 0xFFFF if (neg and mag != 0) else mag


def s(v, bits):
    return v - (1 << bits) if v >> (bits - 1) else v


def rtl_mac(mode, K, xs, ws, bias_raw):
    """One lane of rtl/mac_array.v: acc <= (clear ? bias : acc) + p, AW-bit wrap-around."""
    acc = 0
    for k, (x8, w8) in enumerate(zip(xs, ws)):
        base = bias_raw if k == 0 else acc
        acc = (base + s(rtl_approx_mul(mode, K, x8, w8), 16)) & ((1 << AW) - 1)
    return acc


def _lines(name):
    with open(os.path.join(VEC, name)) as f:
        return [ln.strip() for ln in f if ln.strip()]


class TestMultiplierVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [int(ln, 16) for ln in _lines("mul_vectors.hex")]

    def test_vector_file_is_exhaustive(self):
        self.assertEqual(len(self.rows), 255 * 255)
        pairs = {(s((v >> 88) & 0xFF, 8), s((v >> 80) & 0xFF, 8)) for v in self.rows}
        self.assertEqual(len(pairs), 255 * 255)
        self.assertTrue(all(-127 <= a <= 127 and -127 <= w <= 127 for a, w in pairs))

    def test_model_and_rtl_transliteration_match_vectors(self):
        bad = []
        for v in self.rows:
            a8, w8 = (v >> 88) & 0xFF, (v >> 80) & 0xFF
            for j, (name, mode, k) in enumerate(DESIGNS):
                exp = (v >> (16 * (4 - j))) & 0xFFFF
                got_model = model_signed(mode, k, s(a8, 8), s(w8, 8)) & 0xFFFF
                got_rtl = rtl_approx_mul(mode, k, a8, w8)
                if got_model != exp or got_rtl != exp:
                    bad.append((name, s(a8, 8), s(w8, 8), s(exp, 16), s(got_model, 16), s(got_rtl, 16)))
        self.assertEqual(bad[:5], [], "%d mismatches (design, a, w, expected, model, rtl)" % len(bad))

    def test_exact_column_is_true_product(self):
        for v in self.rows[::97]:
            a, w = s((v >> 88) & 0xFF, 8), s((v >> 80) & 0xFF, 8)
            self.assertEqual(s((v >> 64) & 0xFFFF, 16), a * w)

    def test_no_internal_overflow_for_any_supported_k(self):
        """The 16-bit `mag` and the signed 16-bit output never wrap for any parameter the sweep uses."""
        for mode, ks in ((1, range(0, 9)), (2, range(1, 7))):
            for k in ks:
                worst = max(model_mag(mode, k, a, b) for a in range(128) for b in range(128))
                self.assertLess(worst, 1 << 15, (mode, k))
                for a, b in ((127, 127), (127, 1), (64, 64), (85, 106), (1, 1), (0, 99)):
                    self.assertEqual(rtl_approx_mul(mode, k, a, b), model_mag(mode, k, a, b), (mode, k, a, b))

    def test_minus_128_is_outside_the_contract(self):
        """Documented limitation: -128 has no sign-magnitude form, the RTL treats its magnitude as 0."""
        self.assertEqual(rtl_approx_mul(0, 0, 0x80, 5), 0)


class TestDotProductVectors(unittest.TestCase):
    def test_mac_array_trials(self):
        xs = [int(v, 16) for v in _lines("dot_x.hex")]
        ws = [int(v, 16) for v in _lines("dot_w.hex")]
        bs = [int(v, 16) for v in _lines("dot_b.hex")]
        self.assertEqual((len(xs), len(ws), len(bs)), (ND * LEN, ND * LEN, ND))
        for fname, mode, k in (("dot_e_exact.hex", 0, 0), ("dot_e_t4.hex", 1, 4), ("dot_e_d4.hex", 2, 4)):
            exp = [int(v, 16) for v in _lines(fname)]
            self.assertEqual(len(exp), ND)
            for t in range(ND):
                for lane in range(LANES):
                    x = xs[t * LEN:(t + 1) * LEN]
                    w = [(ws[t * LEN + i] >> (8 * lane)) & 0xFF for i in range(LEN)]
                    got = rtl_mac(mode, k, x, w, (bs[t] >> (AW * lane)) & ((1 << AW) - 1))
                    self.assertEqual(got, (exp[t] >> (AW * lane)) & ((1 << AW) - 1), (fname, t, lane))

    def test_operands_stay_in_sign_magnitude_range(self):
        for name in ("dot_x.hex", "dot_w.hex"):
            for v in _lines(name):
                val = int(v, 16)
                for i in range(len(v) // 2):
                    self.assertNotEqual((val >> (8 * i)) & 0xFF, 0x80, name)


class TestAccumulatorHeadroom(unittest.TestCase):
    def test_24_bit_accumulator_worst_case(self):
        """mac_array.v claims 24 bits hold 448 x 127^2. True for exact/truncated; record where it is not."""
        lim = (1 << (AW - 1)) - 1
        fan_in = 448
        self.assertLess(fan_in * 127 * 127, lim)
        for mode, k in ((1, 4), (1, 6), (1, 8), (2, 6), (2, 4), (2, 3), (2, 2), (2, 1)):
            worst = max(model_mag(mode, k, a, b) for a in range(128) for b in range(128))
            self.assertLess(fan_in * worst, lim, (mode, k, worst))


if __name__ == "__main__":
    unittest.main()
