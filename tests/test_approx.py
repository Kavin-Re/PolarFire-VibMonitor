import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import approx as A

R = np.arange(128)
AA, BB = np.meshgrid(R, R, indexing="ij")


class TestApprox(unittest.TestCase):
    def test_exact(self):
        self.assertTrue(np.array_equal(A.mult_exact(AA, BB), AA * BB))

    def test_trunc_k0_exact_and_underestimates_without_comp(self):
        self.assertTrue(np.array_equal(A.mult_trunc(AA, BB, 7, 0), AA * BB))
        p = A.mult_trunc(AA, BB, 7, 5, comp=False)
        self.assertTrue((p <= AA * BB).all())
        self.assertTrue((AA * BB - p).max() < 2 ** 5 * 7)

    def test_trunc_zero_gating(self):
        p = A.mult_trunc(AA, BB, 7, 6)
        self.assertTrue((p[0, :] == 0).all() and (p[:, 0] == 0).all())

    def test_drum_exact_for_small_operands(self):
        for k in (3, 4, 5):
            small = np.arange(1 << (k))
            a, b = np.meshgrid(small, small, indexing="ij")
            self.assertTrue(np.array_equal(A.mult_drum(a, b, 7, k), a * b))

    def test_drum_error_bounded(self):
        e = A.error_metrics(dict(kind="drum", k=4))
        self.assertLess(e["MRED"], 0.07)
        self.assertLess(A.error_metrics(dict(kind="drum", k=6))["MRED"], e["MRED"])

    def test_mitchell_powers_of_two_exact_and_underestimates(self):
        pw = 2 ** np.arange(7)
        a, b = np.meshgrid(pw, pw, indexing="ij")
        self.assertTrue(np.array_equal(A.mult_mitchell(a, b, 7, 6), a * b))
        p = A.mult_mitchell(AA, BB, 7, 6)
        self.assertTrue((p <= AA * BB).all())
        nz = (AA * BB) > 0
        self.assertLess(((AA * BB - p)[nz] / (AA * BB)[nz]).max(), 0.1112)  # classical 11.1% bound
        self.assertTrue((p[0] == 0).all())

    def test_error_monotonic_with_level(self):
        m = [A.error_metrics(dict(kind="trunc", k=k))["MRED"] for k in (2, 4, 6)]
        self.assertTrue(m[0] < m[1] < m[2])
        m = [A.error_metrics(dict(kind="mitchell", F=f))["MRED"] for f in (6, 4, 2)]
        self.assertTrue(m[0] < m[1] < m[2])

    def test_lut_matches_function(self):
        sp = dict(kind="drum", k=4)
        self.assertTrue(np.array_equal(A.make_lut(sp), A.mult_drum(AA, BB, 7, 4)))

    def test_cost_proxy(self):
        ex = A.mult_ge(dict(kind="exact"))
        self.assertGreater(ex, 0)
        c = [A.mult_ge(dict(kind="trunc", k=k)) for k in (0, 2, 4, 6)]
        self.assertTrue(all(c[i] > c[i + 1] for i in range(3)))
        self.assertGreater(A.mac_ge(dict(kind="exact")), ex)


if __name__ == "__main__":
    unittest.main()
