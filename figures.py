"""Illustrative figures of the synthetic generator (one window per class)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vibgen as V
import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(11)
cfg = V.VibConfig(p_drop=0, p_drift=0, sev_range=(0.8, 1.0))
fig, axs = plt.subplots(6, 3, figsize=(13, 13))
rmsmap = []
for l, name in enumerate(V.CLASSES):
    x, m = V.gen_window(rng, l, cfg)
    c = int(np.argmin(np.abs(V.CH_POS - (m["fault_pos"] if m["fault_pos"] >= 0 else 1.5))))
    s = x[c].astype(float) * 16 / 4096
    t = np.arange(V.N) / V.FS
    axs[l, 0].plot(t[:2048], s[:2048], lw=.5); axs[l, 0].set_ylabel(f"{name}\nch{c} [g]")
    f = np.fft.rfftfreq(V.N, 1 / V.FS)
    P = np.abs(np.fft.rfft((s - s.mean()) * np.hanning(V.N))) / V.N * 4
    axs[l, 1].semilogy(f, P + 1e-6, lw=.6); axs[l, 1].set_xlim(0, 400)
    fr = m["fr_mean"]
    for h, st in [(0.5, ":"), (1, "-"), (2, "--"), (3, "-.")]:
        axs[l, 1].axvline(h * fr, color="r", ls=st, lw=.6, alpha=.6)
    xb = F._fir_same((x[c:c + 1] - x[c].mean()).astype(float), None, F.H_BP, False)
    env = np.abs(xb).reshape(1, F.N_E, F.ENV_DEC).mean(2)[0]
    E = np.abs(np.fft.rfft((env - env.mean()) * np.hanning(F.N_E))) / F.N_E
    fe = np.fft.rfftfreq(F.N_E, 1 / F.FS_E)
    axs[l, 2].plot(fe, E, lw=.7); axs[l, 2].set_xlim(0, 450)
    for o, col in [(V.BPFO, "g"), (2 * V.BPFO, "g"), (V.BPFI, "m"), (2 * V.BPFI, "m")]:
        axs[l, 2].axvline(o * m["fr_mean"], color=col, ls="--", lw=.7)
    rmsmap.append(x.astype(float).std(axis=1) * 16 / 4096)
axs[0, 0].set_title("time [s], first 0.25 s"); axs[0, 1].set_title("spectrum [Hz] (red: 0.5X,1X,2X,3X)")
axs[0, 2].set_title("envelope spectrum [Hz] (green BPFO,2x; magenta BPFI,2x)")
axs[-1, 0].set_xlabel("t [s]"); axs[-1, 1].set_xlabel("Hz"); axs[-1, 2].set_xlabel("Hz")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "generator_examples.png"), dpi=100); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 3.2))
im = ax.imshow(np.array(rmsmap), aspect="auto", cmap="viridis")
ax.set_yticks(range(6)); ax.set_yticklabels(V.CLASSES); ax.set_xlabel("channel"); plt.colorbar(im, label="AC RMS [g]")
ax.set_title("Per-channel RMS (cross-channel coupling by distance)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "channel_rms.png"), dpi=110)
print("ok")
