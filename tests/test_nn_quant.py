import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nn import MLP, quantize, quantize_input, qforward, dot_lut, Standardizer
import approx as A


def toy(n=900, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    c = rng.standard_normal((3, 10)) * 2
    return c[y] + rng.standard_normal((n, 10)), y


class TestNN(unittest.TestCase):
    def setUp(self):
        X, y = toy()
        self.std = Standardizer().fit(X[:600])
        self.Z = self.std(X)
        self.y = y
        self.m = MLP([10, 16, 8, 3], 0)
        self.m.fit(self.Z[:600], y[:600], self.Z[600:750], y[600:750], epochs=40)

    def test_trains(self):
        self.assertGreater((self.m.predict(self.Z[750:]) == self.y[750:]).mean(), 0.9)

    def test_quant_close_to_float(self):
        f = self.m.predict(self.Z[750:])
        for bits, tol in ((8, 0.05), (16, 0.01)):
            q = quantize(self.m, self.Z[:600], bits)
            p = qforward(quantize_input(self.Z[750:], q), q).argmax(1)
            self.assertGreater((p == f).mean(), 1 - tol)

    def test_exact_lut_equals_matmul(self):
        q = quantize(self.m, self.Z[:600], 8)
        x = quantize_input(self.Z[750:], q)
        a = qforward(x, q)
        b = qforward(x, q, A.make_lut(dict(kind="exact")))
        self.assertTrue(np.array_equal(a, b))

    def test_dot_lut_signs(self):
        rng = np.random.default_rng(1)
        x = rng.integers(-127, 128, (5, 20))
        W = rng.integers(-127, 128, (20, 4))
        self.assertTrue(np.array_equal(dot_lut(x, W, A.make_lut(dict(kind="exact"))), x @ W))

    def test_approx_changes_but_stays_sane(self):
        q = quantize(self.m, self.Z[:600], 8)
        x = quantize_input(self.Z[750:], q)
        base = (qforward(x, q).argmax(1) == self.y[750:]).mean()
        ap = (qforward(x, q, A.make_lut(dict(kind="drum", k=4))).argmax(1) == self.y[750:]).mean()
        self.assertGreater(ap, base - 0.1)


if __name__ == "__main__":
    unittest.main()
