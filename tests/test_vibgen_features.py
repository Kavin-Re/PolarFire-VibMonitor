import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vibgen as V
import features as F


class TestVibgen(unittest.TestCase):
    def test_shape_determinism(self):
        a, ma = V.gen_window(np.random.default_rng(5), 3)
        b, mb = V.gen_window(np.random.default_rng(5), 3)
        self.assertEqual(a.shape, (32, 4096))
        self.assertEqual(a.dtype, np.int16)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(a.min() >= -2048 and a.max() <= 2047)

    def test_fault_frequencies(self):
        self.assertAlmostEqual(V.BPFO, 3.5848, places=3)
        self.assertAlmostEqual(V.BPFI, 5.4152, places=3)
        self.assertAlmostEqual(V.BPFO + V.BPFI, 9.0, places=6)  # n_balls identity

    def _spec(self, label, seed=1):
        cfg = V.VibConfig(p_drop=0, p_drift=0, noise_scale=0.3, sev_range=(1, 1))
        x, m = V.gen_window(np.random.default_rng(seed), label, cfg)
        return x, m

    def test_imbalance_1x_peak(self):
        x, m = self._spec(1)
        c = int(np.argmin(np.abs(V.CH_POS - m["fault_pos"])))
        s = np.abs(np.fft.rfft(x[c] - x[c].mean() * 1.0)) ** 2
        f = np.fft.rfftfreq(V.N, 1 / V.FS)
        lo = (f > 5) & (f < 400)
        self.assertLess(abs(f[lo][np.argmax(s[lo])] - m["fr_mean"]), 0.12 * m["fr_mean"])

    def test_outer_race_envelope_peak_at_bpfo(self):
        x, m = self._spec(3, seed=2)
        c = int(np.argmin(np.abs(V.CH_POS - m["fault_pos"])))
        Fe = F.extract_features(x, m["fr_est"], fixed=False)
        # BPFO envelope band energy should exceed the BPFI band on the channel nearest the fault
        self.assertGreater(Fe[c, 9], Fe[c, 11])

    def test_dropout_flatline(self):
        cfg = V.VibConfig(p_drop=1.0, p_drift=0)
        hit = 0
        for s in range(3):
            x, m = V.gen_window(np.random.default_rng(s), 0, cfg)
            hit += int((x.std(axis=1) < 1.0).sum())
        self.assertGreater(hit, 20)  # most channels flat-lined or heavily blanked (bursts keep some variance)


class TestFeatures(unittest.TestCase):
    def test_fft_fixed_vs_float(self):
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((4, 1024)) * 800).astype(np.int64)
        re, im = F.fft_fx(x, np.zeros_like(x))
        ref = np.fft.fft(x.astype(float), axis=-1) / 1024
        err = np.abs((re + 1j * im) - ref)
        self.assertLess(err.max(), 4.0)
        sqnr = 10 * np.log10((np.abs(ref) ** 2).sum() / (err ** 2).sum())
        self.assertGreater(sqnr, 25)  # measured ~27.7 dB (per-stage scaling noise)

    def test_log2_mitchell_error(self):
        v = np.arange(1, 100000)
        e = F.log2_feat(v, True) - np.log2(v)
        self.assertLess(np.abs(e).max(), 0.0861 + 1 / 16)

    def test_timedomain_features_known(self):
        n = np.arange(V.N)
        x = np.round(1000 * np.sin(2 * np.pi * 100 * n / V.FS)).astype(np.int16)
        adc = np.tile(x, (32, 1))
        Fx = F.extract_features(adc, 30.0, fixed=True)
        self.assertAlmostEqual(Fx[0, 2], np.sqrt(2), places=1)   # crest of sine
        self.assertAlmostEqual(Fx[0, 1], 1.5, places=1)          # kurtosis of sine
        self.assertAlmostEqual(Fx[0, 0], np.log2(1000 ** 2 / 2 * V.N), delta=0.15)

    def test_flatline_features_zero(self):
        adc = np.zeros((32, V.N), np.int16) + 17
        Fx = F.extract_features(adc, 30.0, fixed=True)
        self.assertEqual(Fx[0, 1], 0.0)
        self.assertEqual(Fx[0, 2], 0.0)

    def test_fixed_close_to_ideal(self):
        x, m = V.gen_window(np.random.default_rng(3), 4)
        a = F.extract_features(x, m["fr_est"], True)
        b = F.extract_features(x, m["fr_est"], False)
        ok = ~m["drop"]
        self.assertLess(np.abs(a - b)[ok].mean(), 0.5)


if __name__ == "__main__":
    unittest.main()
