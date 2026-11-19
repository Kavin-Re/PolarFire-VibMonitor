"""Build/cache feature datasets from the synthetic generator."""
import os, sys, time
import numpy as np
from multiprocessing import Pool
from vibgen import gen_window, VibConfig, CLASSES, NCH
from features import extract_features, NFEAT

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _one(args):
    seed, label, cfg, ideal = args
    rng = np.random.default_rng(seed)
    adc, meta = gen_window(rng, label, cfg)
    f = extract_features(adc, meta["fr_est"], True)
    fi = extract_features(adc, meta["fr_est"], False) if ideal else np.zeros(1)
    return f, fi, label, meta["drop"], meta["drift"], meta["severity"]


def build(n, seed0, cfg=None, ideal=False, procs=2):
    cfg = cfg or VibConfig()
    jobs = [(seed0 * 1_000_003 + i, i % len(CLASSES), cfg, ideal) for i in range(n)]
    with Pool(procs) as p:
        res = p.map(_one, jobs, chunksize=16)
    out = dict(
        X=np.stack([r[0] for r in res]).astype(np.float32),
        y=np.array([r[2] for r in res]),
        drop=np.stack([r[3] for r in res]),
        drift=np.stack([r[4] for r in res]),
        sev=np.array([r[5] for r in res]),
    )
    if ideal:
        out["X_ideal"] = np.stack([r[1] for r in res]).astype(np.float32)
    return out


def get(name, n, seed0, cfg=None, ideal=False, force=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.npz")
    if os.path.exists(path) and not force:
        z = np.load(path)
        return {k: z[k] for k in z.files}
    t = time.time()
    d = build(n, seed0, cfg, ideal)
    np.savez_compressed(path, **d)
    print(f"built {name}: {n} windows in {time.time()-t:.0f}s", flush=True)
    return d
