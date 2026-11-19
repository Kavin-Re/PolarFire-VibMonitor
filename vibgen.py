"""Synthetic 32-channel rotating-machinery vibration generator (SIMULATION ONLY).

Assumptions (see RESULTS.md for the full list):
  * 32 accelerometers along a 3 m machine train, +-8 g full scale, 12-bit ADC, fs=8192 Hz,
    window N=4096 samples (0.5 s).
  * Shaft speed 20-38 Hz (1200-2280 rpm) with slow speed variation inside the window.
  * Bearing geometry like a 6205 deep-groove bearing (9 balls, d=7.94, D=39.04 mm, 0 deg):
    BPFO=3.585X, BPFI=5.415X, BSF=2.357X, FTF=0.398X (orders of shaft speed).
  * Fault sources reach each channel through a distance-based transmissibility
    (exp(-dist/lambda)) with random per-channel gain and a few samples of delay =>
    cross-channel coupling; plus small electrical adjacent-channel crosstalk.
  * Per-channel sensor faults: gain error, DC offset, slow drift (ramp), dropout
    (flatline or burst).
  * Not validated against any real machine.
"""
from dataclasses import dataclass
import numpy as np

FS = 8192
N = 4096
NCH = 32
FSR_G = 8.0
ADC_BITS = 12
CLASSES = ["healthy", "imbalance", "misalignment", "bearing_outer", "bearing_inner", "looseness"]

_nb, _d, _D = 9, 7.94, 39.04
BPFO = _nb / 2 * (1 - _d / _D)
BPFI = _nb / 2 * (1 + _d / _D)
BSF = _D / (2 * _d) * (1 - (_d / _D) ** 2)
FTF = 0.5 * (1 - _d / _D)
CH_POS = np.linspace(0.0, 3.0, NCH)  # metres along the machine train
_T = np.arange(N) / FS


@dataclass
class VibConfig:
    noise_scale: float = 1.0     # multiplies broadband noise sigma
    p_drop: float = 0.02         # per-channel dropout probability per window
    p_drift: float = 0.03        # per-channel strong-drift probability per window
    xtalk_max: float = 0.03      # max adjacent-channel electrical crosstalk fraction
    sev_range: tuple = (0.3, 1.0)


def _res_kernel(f_res, tau, L=48):
    t = np.arange(L) / FS
    return np.exp(-t / tau) * np.sin(2 * np.pi * f_res * t)


def _impulses(revs, order, jitter, amp_fn, rng):
    """Impulse train at `order` x shaft rate (with slip jitter), float sample positions."""
    ph = revs * order + rng.uniform()
    ev = np.nonzero(np.diff(np.floor(ph)) > 0)[0] + 1
    if ev.size == 0:
        return np.zeros(N)
    period = N / max(ph[-1] - ph[0], 1e-9)  # samples per event
    pos = ev - 0.5 + rng.normal(0, jitter * period, ev.size)
    amp = amp_fn(ev) * (1 + 0.1 * rng.standard_normal(ev.size))
    imp = np.zeros(N + 2)
    i0 = np.floor(pos).astype(int)
    w = pos - i0
    ok = (i0 >= 0) & (i0 < N)
    np.add.at(imp, i0[ok], amp[ok] * (1 - w[ok]))
    np.add.at(imp, i0[ok] + 1, amp[ok] * w[ok])
    return imp[:N]


def gen_window(rng, label, cfg=None):
    """Return (adc int16 [32,N], meta dict). label: index into CLASSES."""
    cfg = cfg or VibConfig()
    # ---- speed profile
    fr0 = rng.uniform(20, 38)
    a_m, f_m = rng.uniform(0.002, 0.02), rng.uniform(0.2, 2.0)
    ramp = rng.uniform(-0.02, 0.02)
    fr_t = fr0 * (1 + ramp * (_T / _T[-1] - 0.5) + a_m * np.sin(2 * np.pi * f_m * _T + rng.uniform(0, 6.28)))
    revs = np.cumsum(fr_t) / FS               # shaft revolutions vs sample
    ph = 2 * np.pi * revs
    fr_mean = float(fr_t.mean())
    fr_est = fr_mean * (1 + 0.003 * rng.standard_normal())  # tacho estimate
    sp = (fr0 / 30.0)
    sev = rng.uniform(*cfg.sev_range) if label > 0 else 0.0

    def tone(h, a, phi=None):
        return a * np.sin(h * ph + (rng.uniform(0, 6.28) if phi is None else phi))

    # ---- sources: (signal, position, lambda)
    srcs = []
    base = tone(1, rng.uniform(0.03, 0.10) * sp ** 2) + tone(2, rng.uniform(0.005, 0.03)) + tone(3, rng.uniform(0.002, 0.015))
    srcs.append((base, rng.uniform(0.5, 2.5), 1.5))
    if rng.random() < 0.3:  # non-synchronous (fan / gear-mesh like) tone
        srcs.append((tone(1, 0.0) + rng.uniform(0.01, 0.05) * np.sin(2 * np.pi * rng.uniform(150, 400) * _T + 1.0), rng.uniform(0, 3), 1.0))
    if rng.random() < 0.15:  # benign random impacts (raises kurtosis w/o a bearing fault)
        imp = np.zeros(N)
        k = rng.integers(1, 4)
        imp[rng.integers(0, N - 60, k)] = rng.uniform(0.2, 0.8, k)
        srcs.append((np.convolve(imp, _res_kernel(rng.uniform(1500, 3000), 0.0006))[:N], rng.uniform(0, 3), 1.0))
    fault_pos = -1.0
    if label == 1:   # imbalance: 1X dominant
        fault_pos = rng.uniform(0.8, 2.2)
        A = (0.12 + 0.7 * sev) * sp ** 2
        srcs.append((tone(1, A) + tone(2, 0.08 * A), fault_pos, 1.8))
    elif label == 2:  # misalignment: 2X dominant (+1X, 3X)
        fault_pos = rng.uniform(1.0, 2.0)
        A = (0.08 + 0.45 * sev) * sp
        srcs.append((tone(2, A) + tone(1, 0.35 * A) + tone(3, 0.25 * A) + tone(4, 0.08 * A), fault_pos, 1.8))
    elif label in (3, 4):  # bearing outer / inner race fault
        fault_pos = float(rng.choice([0.0, 1.0, 2.0, 3.0]))
        A = (0.15 + 1.2 * sev) * sp
        kern = _res_kernel(rng.uniform(2300, 3100), rng.uniform(0.0004, 0.001))
        if label == 3:
            imp = _impulses(revs, BPFO, 0.01, lambda e: np.full(e.size, A), rng)
        else:
            m, p0 = rng.uniform(0.5, 1.0), rng.uniform(0, 6.28)
            imp = _impulses(revs, BPFI, 0.01, lambda e: A * np.maximum(1 + m * np.cos(2 * np.pi * revs[e] + p0), 0.0), rng)
        srcs.append((np.convolve(imp, kern)[:N], fault_pos, 0.7))
    elif label == 5:  # looseness: sub-harmonics + many harmonics + truncation
        fault_pos = rng.uniform(0.0, 3.0)
        A = (0.1 + 0.5 * sev) * sp
        x = tone(1, 1.0, 0.0)
        s = tone(0.5, 0.45) + tone(1.5, 0.3) + tone(2.5, 0.15)
        for h in range(2, 9):
            s += tone(h, rng.uniform(0.1, 0.4) / h * 2)
        clip = np.clip(1.5 * x, -1, 1) - x
        srcs.append((A * (s + 0.6 * clip), fault_pos, 1.5))

    # ---- mix into channels with transmissibility and delays
    X = np.zeros((NCH, N))
    idx = np.arange(N)[None, :]
    for sig, pos, lam in srcs:
        g = np.exp(-np.abs(CH_POS - pos) / lam) * rng.uniform(0.4, 1.0, NCH)
        dly = rng.integers(0, 4, NCH)
        X += g[:, None] * sig[(idx - dly[:, None]) % N]
    # ---- noise: white + pink-ish + 50 Hz pickup
    sig_n = rng.uniform(0.03, 0.12) * cfg.noise_scale
    wn = rng.standard_normal((NCH, N)) * sig_n * rng.uniform(0.8, 1.2, (NCH, 1))
    f = np.fft.rfftfreq(N, 1 / FS)
    pn = np.fft.irfft(np.fft.rfft(rng.standard_normal((NCH, N)), axis=1) / np.sqrt(1 + f / 50.0), n=N, axis=1)
    X += wn + 0.5 * sig_n * pn
    if rng.random() < 0.3:
        X += rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * 50 * _T + rng.uniform(0, 6.28, (NCH, 1)))
    # ---- sensor effects
    xt = rng.uniform(0, cfg.xtalk_max)
    Xs = X.copy()
    Xs[1:] += xt * X[:-1]
    Xs[:-1] += xt * X[1:]
    X = Xs
    X = X * rng.uniform(0.93, 1.07, (NCH, 1)) + rng.uniform(-0.1, 0.1, (NCH, 1))
    drift = rng.random(NCH) < cfg.p_drift
    X += (drift * rng.uniform(-1, 1, NCH) * rng.uniform(0.2, 1.0))[:, None] * (_T / _T[-1])[None, :]
    drop = rng.random(NCH) < cfg.p_drop
    for c in np.nonzero(drop)[0]:
        off = X[c].mean()
        if rng.random() < 0.6:
            X[c] = off
        else:
            L = int(rng.uniform(0.05, 0.4) * N)
            s0 = rng.integers(0, N - L)
            X[c, s0:s0 + L] = off
    # ---- 12-bit ADC
    lsb = 2 * FSR_G / (1 << ADC_BITS)
    adc = np.clip(np.round(X / lsb), -(1 << (ADC_BITS - 1)), (1 << (ADC_BITS - 1)) - 1).astype(np.int16)
    meta = dict(label=label, severity=sev, fr_mean=fr_mean, fr_est=fr_est, fault_pos=fault_pos,
                drift=drift, drop=drop)
    return adc, meta
