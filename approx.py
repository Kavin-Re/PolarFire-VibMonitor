"""Self-contained approximate multipliers + gate-equivalent (GE) cost PROXY.

All multipliers operate on UNSIGNED MAGNITUDES of `n` bits (default n=7: int8
operands in sign-magnitude, |x|<=127).  Sign = XOR of operand signs, applied
outside (identical for every design, so excluded from the cost proxy).

Designs
  exact            : n x n array multiplier
  trunc(k)         : partial-product bits with column weight < 2^k are dropped;
                     optional constant compensation (expected value of the dropped
                     bits assuming independent uniform bits), gated so 0*x == 0.
  drum(k)          : DRUM-k (Hashemi et al. 2015): keep k bits starting at the
                     leading one, force segment LSB to 1 (unbiasing), exact k x k.
  mitchell(F)      : Mitchell log multiplier (1962) with F fractional bits kept in
                     the log domain (F=n-1 is the classic full Mitchell).
"""
import numpy as np
from functools import lru_cache
from math import ceil, log2

NBITS = 7  # magnitude bits of int8 sign-magnitude operands


def lod(x):
    """Leading-one position of non-negative ints (0 for x==0)."""
    x = np.asarray(x, dtype=np.int64)
    k = np.zeros_like(x)
    for b in range(1, 40):
        k = np.where((x >> b) > 0, b, k)
    return k


def mult_exact(a, b, n=NBITS):
    return np.asarray(a, np.int64) * np.asarray(b, np.int64)


def trunc_comp_const(n, k):
    """Compensation constant: expected value of the dropped partial-product bits
    (each bit ~ P=1/4), i.e. floor((sum 2^(i+j) + 2)/4) over dropped (i,j). Integer-only
    so the RTL can compute the identical value."""
    tot = sum(2 ** (i + j) for i in range(n) for j in range(n) if i + j < k)
    return (tot + 2) // 4


def mult_trunc(a, b, n=NBITS, k=4, comp=True):
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64)
    out = np.zeros(np.broadcast(a, b).shape, np.int64)
    for i in range(n):
        s = max(k - i, 0)
        out = out + (((a >> i) & 1) * (((b >> s) << s) << i))
    if comp and k > 0:
        out = out + trunc_comp_const(n, k) * ((a > 0) & (b > 0))
    return out


def drum_segment(x, k):
    x = np.asarray(x, np.int64)
    sh = np.maximum(lod(x) - (k - 1), 0)
    seg = np.where(sh > 0, (x >> sh) | 1, x)
    return seg, sh


def mult_drum(a, b, n=NBITS, k=4):
    sa, ha = drum_segment(a, k)
    sb, hb = drum_segment(b, k)
    return (sa * sb) << (ha + hb)


def mult_mitchell(a, b, n=NBITS, F=6):
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64)
    ka, kb = lod(a), lod(b)
    fa = np.where(a > 0, ((a - (1 << ka)) << F) >> ka, 0)
    fb = np.where(b > 0, ((b - (1 << kb)) << F) >> kb, 0)
    s = ((ka + kb) << F) + fa + fb
    K = s >> F
    f = s & ((1 << F) - 1)
    res = (((1 << F) + f) << K) >> F
    return np.where((a > 0) & (b > 0), res, 0)


# ---------------------------------------------------------------- configs
def spec_list(n=NBITS):
    specs = [dict(name="exact", kind="exact")]
    for k in (2, 3, 4, 5, 6, 7, 8):
        specs.append(dict(name=f"trunc{k}", kind="trunc", k=k))
    for k in (6, 5, 4, 3, 2, 1):
        specs.append(dict(name=f"drum{k}", kind="drum", k=k))
    for F in (6, 4, 3, 2, 1, 0):
        specs.append(dict(name=f"mitch{F}", kind="mitchell", F=F))
    return specs


def apply_spec(spec, a, b, n=NBITS):
    kd = spec["kind"]
    if kd == "exact":
        return mult_exact(a, b, n)
    if kd == "trunc":
        return mult_trunc(a, b, n, spec["k"])
    if kd == "drum":
        return mult_drum(a, b, n, spec["k"])
    if kd == "mitchell":
        return mult_mitchell(a, b, n, spec["F"])
    raise ValueError(kd)


@lru_cache(maxsize=None)
def _lut(kind, n, k, F):
    A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n), indexing="ij")
    return apply_spec(dict(kind=kind, k=k, F=F), A, B, n).astype(np.int32)


def make_lut(spec, n=NBITS):
    """(2^n x 2^n) int32 table of approximate magnitude products."""
    return _lut(spec["kind"], n, spec.get("k", 0), spec.get("F", 0))


def error_metrics(spec, n=NBITS):
    """Exhaustive over all operand pairs (uniform operand distribution)."""
    lut = make_lut(spec, n).astype(np.float64)
    A, B = np.meshgrid(np.arange(1 << n), np.arange(1 << n), indexing="ij")
    ex = (A * B).astype(np.float64)
    err = lut - ex
    nz = ex > 0
    return dict(
        MED=float(np.abs(err).mean()),
        NMED=float(np.abs(err).mean() / ex.max()),
        MRED=float((np.abs(err[nz]) / ex[nz]).mean()),
        bias=float(err.mean()),
        max_abs_err=float(np.abs(err).max()),
        max_rel_err=float((np.abs(err[nz]) / ex[nz]).max()),
        exact_frac=float((err == 0).mean()),
    )


# ---------------------------------------------------------------- cost PROXY
# Unit costs in gate equivalents (GE, 1 GE = one 2-input NAND). These are
# textbook-order-of-magnitude ASSUMPTIONS, not measured and not PolarFire LUT
# counts.  Only the *relative* ordering is meant to be informative.
GE = dict(AND=1.0, FA=5.0, MUX=2.5, FF=6.0, LOD_PER_BIT=2.0)
ACC_BITS = 24


def shifter_ge(in_w, max_shift, out_cap):
    """Log barrel shifter: stage s shifts by 2^s; output width grows then caps."""
    stages = ceil(log2(max_shift + 1)) if max_shift > 0 else 0
    return sum(min(in_w + (2 ** (s + 1) - 1), out_cap) * GE["MUX"] for s in range(stages))


def array_ge(n, k_trunc=0):
    pp = sum(n - max(k_trunc - i, 0) for i in range(n))
    cols = 2 * n - k_trunc
    fa = max(pp - cols, 0)  # each 3:2 compression removes one bit
    return pp * GE["AND"] + fa * GE["FA"]


def mult_ge(spec, n=NBITS):
    kd = spec["kind"]
    if kd == "exact":
        return array_ge(n)
    if kd == "trunc":
        return array_ge(n, spec["k"])
    lodg = 2 * n * GE["LOD_PER_BIT"]
    if kd == "drum":
        k = spec["k"]
        ext = 2 * shifter_ge(n, n - k, k)  # extract k-bit window
        core = array_ge(k)
        amt = ceil(log2(n)) * GE["FA"]  # add the two shift amounts
        outsh = shifter_ge(2 * k, 2 * (n - k), 2 * n)
        return lodg + ext + core + amt + outsh
    if kd == "mitchell":
        F = spec["F"]
        norm = 2 * shifter_ge(n - 1, n - 1, n - 1)
        add = (ceil(log2(n)) + F) * GE["FA"]
        anti = shifter_ge(F + 1, 2 * n, 2 * n)
        return lodg + norm + add + anti
    raise ValueError(kd)


def mac_ge(spec, n=NBITS):
    """Multiplier + 24-bit accumulator adder + accumulator register."""
    return mult_ge(spec, n) + ACC_BITS * (GE["FA"] + GE["FF"])
