"""Generate golden vectors for tb_mac_array.v from the Python models (approx.py).
Run:  python3 gen_vectors.py   (from rtl/ or anywhere)  -> rtl/vectors/*.hex
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from approx import make_lut

LANES, LEN, AW = 8, 64, 24
SPECS = dict(
    exact=dict(kind="exact"),
    t4=dict(kind="trunc", k=4), t6=dict(kind="trunc", k=6),
    d4=dict(kind="drum", k=4), d3=dict(kind="drum", k=3),
)


def smul(spec, a, w):
    lut = make_lut(spec).astype(np.int64)
    m = lut[np.abs(a), np.abs(w)]
    return np.sign(a) * np.sign(w) * m


def hexw(v, bits):
    return format(int(v) & ((1 << bits) - 1), f"0{bits // 4}x")


def main():
    out = os.path.join(HERE, "vectors")
    os.makedirs(out, exist_ok=True)
    # exhaustive multiplier vectors over [-127,127]^2
    A, W = np.meshgrid(np.arange(-127, 128), np.arange(-127, 128), indexing="ij")
    A, W = A.ravel(), W.ravel()
    cols = [smul(SPECS[k], A, W) for k in ("exact", "t4", "t6", "d4", "d3")]
    with open(os.path.join(out, "mul_vectors.hex"), "w") as f:
        for i in range(len(A)):
            f.write(hexw(A[i], 8) + hexw(W[i], 8) + "".join(hexw(c[i], 16) for c in cols) + "\n")
    # dot-product trials
    rng = np.random.default_rng(7)
    trials = []
    qpath = os.path.join(HERE, "..", "data", "qmodel_int8.npz")
    if os.path.exists(qpath):  # trial 0: real quantised layer-0 slice + a real input row
        q = np.load(qpath)
        x = q["x_example"][:LEN].astype(np.int64)
        Wm = q["W0"][:LEN, :LANES].astype(np.int64)
        b = q["b0"][:LANES].astype(np.int64)
        trials.append((x, Wm, b))
    while len(trials) < 12:
        x = rng.integers(0, 128, LEN) * (rng.random(LEN) > 0.3)
        Wm = rng.integers(-127, 128, (LEN, LANES))
        b = rng.integers(-5000, 5000, LANES)
        trials.append((x.astype(np.int64), Wm.astype(np.int64), b.astype(np.int64)))
    fx = open(os.path.join(out, "dot_x.hex"), "w")
    fw = open(os.path.join(out, "dot_w.hex"), "w")
    fb = open(os.path.join(out, "dot_b.hex"), "w")
    fe = {k: open(os.path.join(out, f"dot_e_{k}.hex"), "w") for k in ("exact", "t4", "d4")}
    for x, Wm, b in trials:
        for k in range(LEN):
            fx.write(hexw(x[k], 8) + "\n")
            fw.write("".join(hexw(Wm[k, l], 8) for l in reversed(range(LANES))) + "\n")
        fb.write("".join(hexw(b[l], AW) for l in reversed(range(LANES))) + "\n")
        for key in fe:
            acc = np.array([(smul(SPECS[key], x, Wm[:, l])).sum() + b[l] for l in range(LANES)])
            fe[key].write("".join(hexw(acc[l], AW) for l in reversed(range(LANES))) + "\n")
    for f in [fx, fw, fb] + list(fe.values()):
        f.close()
    print(f"wrote vectors: {len(A)} mult vectors, {len(trials)} dot trials (real-slice trial: {os.path.exists(qpath)})")


if __name__ == "__main__":
    main()
