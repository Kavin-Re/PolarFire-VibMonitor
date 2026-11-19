"""Fixed-point-friendly feature extraction for the 32-channel monitor.

Per channel (14 features):
  0 log2(sum x^2)   (energy, i.e. 2*log2 RMS + const)
  1 kurtosis        2 crest factor
  3..8  band energies (log2) from a 1024-pt FFT of the /4 decimated signal (2048 Hz):
        0.5X, 1X, 2X, 3X, 4-10X, HF(10.5X..1 kHz)   (bands defined in orders of the tacho speed)
  9..12 envelope-spectrum band energies (log2) at BPFO, 2BPFO, BPFI, 2BPFI
        (2-3.5 kHz band-pass -> |.| -> /8 boxcar -> 512-pt FFT @1024 Hz)
  13    log2 of total envelope-spectrum energy

fixed=True : integer arithmetic (Q15 FIR taps and Hann window, Q15-twiddle radix-2 FFT
             with 1/2 scaling per stage, int64 accumulators, Mitchell-style log2 with 4
             fractional bits).  This is what the deployed pipeline would compute.
fixed=False: floating-point "ideal" reference with identical scaling conventions.
Square-root / division for RMS/kurtosis/crest are evaluated in float from the integer
sums (a hardware version needs an iterative divider / integer sqrt, not modelled).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from vibgen import FS, N, NCH, BPFO, BPFI
from approx import lod

NFEAT = 14
DEC = 4
FS_D = FS // DEC              # 2048
N_D = N // DEC                # 1024
ENV_DEC = 8
FS_E = FS // ENV_DEC          # 1024
N_E = N // ENV_DEC            # 512
GUARD_D, GUARD_E = 4, 5       # guard-bit left shifts before the FFTs
ORDER_BANDS = [(0.35, 0.65), (0.85, 1.15), (1.85, 2.15), (2.85, 3.15), (3.5, 10.5)]


def _fir(cut_lo, cut_hi, taps=31):
    n = np.arange(taps) - (taps - 1) / 2
    def sinc_lp(fc):
        return 2 * fc / FS * np.sinc(2 * fc / FS * n)
    h = sinc_lp(cut_hi) - (sinc_lp(cut_lo) if cut_lo > 0 else 0)
    h = h * np.hamming(taps)
    return h


H_LP = _fir(0, 900.0)
H_BP = _fir(2000.0, 3500.0)
H_LP_Q = np.round(H_LP * 32768).astype(np.int64)
H_BP_Q = np.round(H_BP * 32768).astype(np.int64)


def hann(n, fixed):
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    return np.round(w * 32767).astype(np.int64) if fixed else w


def _bitrev(n):
    b = int(np.log2(n))
    r = np.zeros(n, dtype=np.int64)
    for i in range(n):
        r[i] = int(format(i, f"0{b}b")[::-1], 2)
    return r


def fft_fx(re, im, saturate=True):
    """Radix-2 DIT FFT, Q15 twiddles, >>1 per stage (overall 1/n), int16 saturation."""
    n = re.shape[-1]
    sh = re.shape
    rev = _bitrev(n)
    re = re[..., rev].astype(np.int64)
    im = im[..., rev].astype(np.int64)
    for s in range(int(np.log2(n))):
        m = 1 << (s + 1)
        h = m >> 1
        ang = -2 * np.pi * np.arange(h) / m
        wr = np.round(np.cos(ang) * 32767).astype(np.int64)
        wi = np.round(np.sin(ang) * 32767).astype(np.int64)
        r = re.reshape(sh[:-1] + (n // m, m))
        i = im.reshape(sh[:-1] + (n // m, m))
        ar, ai, br, bi = r[..., :h], i[..., :h], r[..., h:], i[..., h:]
        tr = (br * wr - bi * wi + 16384) >> 15
        ti = (br * wi + bi * wr + 16384) >> 15
        nr = np.concatenate([(ar + tr + 1) >> 1, (ar - tr + 1) >> 1], axis=-1)
        ni = np.concatenate([(ai + ti + 1) >> 1, (ai - ti + 1) >> 1], axis=-1)
        if saturate:
            nr = np.clip(nr, -32768, 32767)
            ni = np.clip(ni, -32768, 32767)
        re, im = nr.reshape(sh), ni.reshape(sh)
    return re, im


def power_spectrum(x, fixed):
    """x: [C, n] (int or float). Returns one-sided power [C, n/2+1], scaled as |FFT/n|^2."""
    n = x.shape[-1]
    if fixed:
        re, im = fft_fx(x.astype(np.int64), np.zeros_like(x, dtype=np.int64))
        p = re * re + im * im
    else:
        X = np.fft.fft(x.astype(np.float64), axis=-1) / n
        p = X.real ** 2 + X.imag ** 2
    return p[..., : n // 2 + 1]


def log2_feat(v, fixed):
    """Mitchell-style log2 with 4 fractional bits (fixed) or exact log2 (ideal)."""
    if not fixed:
        return np.log2(np.maximum(v, 1.0))
    v = np.maximum(np.asarray(v, np.int64), 1)
    k = lod(v)
    mant = ((v - (1 << k)) << 4) >> k
    return (k * 16 + mant) / 16.0


def _fir_same(x, hq, h, fixed):
    T = len(h)
    xp = np.pad(x, ((0, 0), (T // 2, T // 2)))
    win = sliding_window_view(xp, T, axis=1)
    if fixed:
        return (win.astype(np.int64) @ hq[::-1].copy() + 16384) >> 15
    return win.astype(np.float64) @ h[::-1]


def band_sum(p, lo_hz, hi_hz, df):
    """Sum p over bins whose centre lies in [lo,hi] Hz. lo/hi arrays-or-scalars."""
    k = np.arange(p.shape[-1])
    m = (k * df >= lo_hz) & (k * df <= hi_hz) & (k >= 1)
    return (p * m).sum(axis=-1)


def extract_features(adc, fr_est, fixed=True):
    """adc: int16 [32,N]; fr_est: tacho speed (Hz). Returns float [32, NFEAT]."""
    C = adc.shape[0]
    x = adc.astype(np.int64) if fixed else adc.astype(np.float64)
    if fixed:
        x0 = x - ((x.sum(axis=1, keepdims=True) + N // 2) >> 12)
        sumsq = (x0 * x0).sum(axis=1)
        sum4 = ((x0 * x0) ** 2).sum(axis=1)
        sumsq_f, sum4_f = sumsq.astype(np.float64), sum4.astype(np.float64)
    else:
        x0 = x - x.mean(axis=1, keepdims=True)
        sumsq_f = (x0 ** 2).sum(axis=1)
        sum4_f = (x0 ** 4).sum(axis=1)
        sumsq = sumsq_f
    peak = np.abs(x0).max(axis=1)
    rms = np.sqrt(sumsq_f / N)
    live = sumsq_f > 0
    kurt = np.where(live, N * sum4_f / np.maximum(sumsq_f, 1) ** 2, 0.0)
    crest = np.where(live, peak / np.maximum(rms, 1e-9), 0.0)
    F = np.zeros((C, NFEAT))
    F[:, 0] = log2_feat(sumsq, fixed)
    F[:, 1] = kurt
    F[:, 2] = crest
    # ---- low-rate branch: order-band energies
    xd = _fir_same(x0, H_LP_Q, H_LP, fixed)[:, ::DEC]
    xd = (xd << GUARD_D) if fixed else xd * (1 << GUARD_D)
    w = hann(N_D, fixed)
    xw = np.clip((xd * w + 16384) >> 15, -32768, 32767) if fixed else xd * w
    p = power_spectrum(xw, fixed)
    df = FS_D / N_D
    for j, (a, b) in enumerate(ORDER_BANDS):
        F[:, 3 + j] = log2_feat(band_sum(p, a * fr_est, b * fr_est, df), fixed)
    F[:, 8] = log2_feat(band_sum(p, 10.5 * fr_est, 1000.0, df), fixed)
    # ---- envelope branch
    xb = _fir_same(x0, H_BP_Q, H_BP, fixed)
    env = np.abs(xb)
    env = env.reshape(C, N_E, ENV_DEC).sum(axis=2)
    env = (env >> 3) if fixed else env / 8.0
    env = env - ((env.sum(axis=1, keepdims=True) + N_E // 2) // N_E if fixed else env.mean(axis=1, keepdims=True))
    env = (env << GUARD_E) if fixed else env * (1 << GUARD_E)
    w = hann(N_E, fixed)
    ew = np.clip((env * w + 16384) >> 15, -32768, 32767) if fixed else env * w
    pe = power_spectrum(ew, fixed)
    dfe = FS_E / N_E
    for j, o in enumerate([BPFO, 2 * BPFO, BPFI, 2 * BPFI]):
        f0 = o * fr_est
        hw = max(3.0, 0.05 * f0)
        F[:, 9 + j] = log2_feat(band_sum(pe, f0 - hw, f0 + hw, dfe), fixed)
    F[:, 13] = log2_feat(band_sum(pe, 2.0, FS_E / 2, dfe), fixed)
    return F
