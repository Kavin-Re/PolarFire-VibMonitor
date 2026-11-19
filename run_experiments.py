"""Train -> quantise -> approximate-MAC sweep. Writes results.json, figures/, data/qmodel_int8.npz"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dataset
from vibgen import CLASSES
from features import NFEAT
from nn import MLP, Standardizer, quantize, quantize_input, qforward, count_macs
import approx

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SIZES = [32 * NFEAT, 64, 32, len(CLASSES)]
TESTS = ["test_nominal", "test_highnoise", "test_heavydrop", "test_weakfault"]
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)


def ci95(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)


tr = dataset.get("train", 3000, 1)
te = {n: dataset.get(n, 0, 0) for n in TESTS}
Xall = tr["X"].reshape(len(tr["y"]), -1)
yall = tr["y"]
perm = np.random.default_rng(0).permutation(len(yall))
vi, ti = perm[:450], perm[450:]
std = Standardizer().fit(Xall[ti])
Ztr, Zva = std(Xall[ti]), std(Xall[vi])
ytr, yva = yall[ti], yall[vi]
Zte = {n: std(te[n]["X"].reshape(len(te[n]["y"]), -1)) for n in TESTS}
yte = {n: te[n]["y"] for n in TESTS}
R = dict(sizes=SIZES, n_train=len(ti), n_val=len(vi), n_test={n: int(len(yte[n])) for n in TESTS})
R["macs_per_layer"] = count_macs(SIZES)
R["macs_total"] = int(sum(R["macs_per_layer"]))
R["params"] = int(sum(a * b + b for a, b in zip(SIZES[:-1], SIZES[1:])))

# ---------------------------------------------------------------- train 3 seeds
models, q8, q16 = [], [], []
R["float"], R["int8_exact"], R["int16_exact"] = {}, {}, {}
for seed in range(3):
    m = MLP(SIZES, seed)
    va = m.fit(Ztr, ytr, Zva, yva, epochs=80, seed=seed)
    models.append(m)
    q8.append(quantize(m, Ztr, 8))
    q16.append(quantize(m, Ztr, 16))
    log(f"seed {seed}: val acc {va:.4f}")
for name, fn in [("float", lambda i, n: models[i].predict(Zte[n])),
                 ("int8_exact", lambda i, n: qforward(quantize_input(Zte[n], q8[i]), q8[i]).argmax(1)),
                 ("int16_exact", lambda i, n: qforward(quantize_input(Zte[n], q16[i]), q16[i]).argmax(1))]:
    for n in TESTS:
        accs = [float((fn(i, n) == yte[n]).mean()) for i in range(3)]
        R[name][n] = dict(per_seed=accs, mean=float(np.mean(accs)), sd=float(np.std(accs)))
    log(name, {n: round(R[name][n]["mean"], 4) for n in TESTS})

# confusion matrix (seed 0, int8 exact, nominal)
pred = qforward(quantize_input(Zte["test_nominal"], q8[0]), q8[0]).argmax(1)
cm = np.zeros((6, 6), int)
for a, b in zip(yte["test_nominal"], pred):
    cm[a, b] += 1
R["confusion_int8_nominal_seed0"] = cm.tolist()
R["per_class_acc_int8_nominal_seed0"] = (np.diag(cm) / cm.sum(1)).tolist()
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
for i in range(6):
    for j in range(6):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="w" if cm[i, j] > cm.max() / 2 else "k", fontsize=8)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("int8 exact MAC, nominal test (seed 0)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "confusion_int8.png"), dpi=130); plt.close(fig)

# export int8 model (seed 0) + one real input row for the RTL vector generator
xq = quantize_input(Zte["test_nominal"][:1], q8[0])[0]
np.savez(os.path.join(HERE, "data", "qmodel_int8.npz"), W0=q8[0]["layers"][0]["Wq"], b0=q8[0]["layers"][0]["bq"],
         W1=q8[0]["layers"][1]["Wq"], b1=q8[0]["layers"][1]["bq"], W2=q8[0]["layers"][2]["Wq"],
         b2=q8[0]["layers"][2]["bq"], x_example=xq, mu=std.mu, sd=std.sd)
R["weight_range_int8"] = [int(min(l["Wq"].min() for l in q8[0]["layers"])), int(max(l["Wq"].max() for l in q8[0]["layers"]))]
R["input_range_int8"] = [int(quantize_input(Zte["test_nominal"], q8[0]).min()), int(quantize_input(Zte["test_nominal"], q8[0]).max())]

# ---------------------------------------------------------------- feature fidelity (fixed vs ideal)
te0 = dataset.get("test_nominal", 0, 0)
Xf, Xi = te0["X"], te0["X_ideal"]
live = ~te0["drop"]
R["feat_fidelity"] = dict(
    mean_abs_diff_per_feature=[float(np.abs(Xf[:, :, j] - Xi[:, :, j])[live].mean()) for j in range(NFEAT)],
    feature_std_per_feature=[float(Xi[:, :, j][live].std()) for j in range(NFEAT)],
)
Zi = std(Xi.reshape(len(Xi), -1))
R["acc_ideal_features_into_fixedtrained_int8"] = float((qforward(quantize_input(Zi, q8[0]), q8[0]).argmax(1) == yte["test_nominal"]).mean())
log("ideal-feature mismatch acc", R["acc_ideal_features_into_fixedtrained_int8"])

# ---------------------------------------------------------------- approximate sweep
specs = approx.spec_list()
R["sweep"] = []
base_cost = approx.mult_ge(specs[0])
base_mac = approx.mac_ge(specs[0])
for sp in specs:
    lut = approx.make_lut(sp)
    row = dict(name=sp["name"], spec=sp, **{f"err_{k}": v for k, v in approx.error_metrics(sp).items()})
    row["mult_ge"] = approx.mult_ge(sp)
    row["mac_ge"] = approx.mac_ge(sp)
    row["rel_mult_ge"] = row["mult_ge"] / base_cost
    row["rel_mac_ge"] = row["mac_ge"] / base_mac
    for n in TESTS:
        accs = [float((qforward(quantize_input(Zte[n], q8[i]), q8[i], lut).argmax(1) == yte[n]).mean()) for i in range(3)] if n == "test_nominal" else \
               [float((qforward(quantize_input(Zte[n], q8[0]), q8[0], lut).argmax(1) == yte[n]).mean())]
        row[n] = dict(mean=float(np.mean(accs)), sd=float(np.std(accs)), per_seed=accs)
    row["L1_only_nominal"] = float((qforward(quantize_input(Zte["test_nominal"], q8[0]), q8[0], lut, {0}).argmax(1) == yte["test_nominal"]).mean())
    R["sweep"].append(row)
    log(f"{sp['name']:8s} multGE={row['mult_ge']:.0f} MRED={row['err_MRED']:.4f} acc={row['test_nominal']['mean']:.4f}")

# ---------------------------------------------------------------- Pareto
def pareto(pts):  # pts: list of (cost, acc); maximise acc, minimise cost
    idx = []
    for i, (c, a) in enumerate(pts):
        if not any((c2 <= c and a2 >= a and (c2 < c or a2 > a)) for j, (c2, a2) in enumerate(pts) if j != i):
            idx.append(i)
    return idx

fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
mk = dict(exact="*", trunc="s", drum="o", mitchell="^")
col = dict(exact="k", trunc="#1b6ca8", drum="#d1495b", mitchell="#2e8b57")
for ax, key, lab in [(axs[0], "rel_mult_ge", "multiplier GE proxy / exact"), (axs[1], "rel_mac_ge", "MAC GE proxy / exact (mult + 24b acc + reg)")]:
    pts = [(r[key], r["test_nominal"]["mean"]) for r in R["sweep"]]
    pf = pareto(pts)
    for r in R["sweep"]:
        k = r["spec"]["kind"]
        ax.errorbar(r[key], r["test_nominal"]["mean"] * 100, yerr=r["test_nominal"]["sd"] * 100, fmt=mk[k], ms=7, capsize=2, label=k)
        ax.annotate(r["name"], (r[key], r["test_nominal"]["mean"] * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ps = sorted(pf, key=lambda i: pts[i][0])
    ax.step([pts[i][0] for i in ps], [pts[i][1] * 100 for i in ps], where="post", color="gray", ls="--", lw=1)
    ax.set_xlabel(lab + "  (PROXY, not synthesised)"); ax.grid(alpha=.3)
    R.setdefault("pareto_" + key, [R["sweep"][i]["name"] for i in ps])
h, l = axs[0].get_legend_handles_labels()
u = dict(zip(l, h)); axs[0].legend(u.values(), u.keys(), loc="lower right", fontsize=8)
axs[0].set_ylabel("nominal test accuracy [%] (int8, mean+-sd of 3 seeds)")
fig.suptitle("Accuracy vs approximate-multiplier cost proxy (synthetic data)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "pareto.png"), dpi=130); plt.close(fig)

# accuracy vs error figure
fig, ax = plt.subplots(figsize=(5.5, 4))
for r in R["sweep"]:
    k = r["spec"]["kind"]
    ax.semilogx(max(r["err_MRED"], 1e-4), r["test_nominal"]["mean"] * 100, mk[k], color=col[k])
    ax.annotate(r["name"], (max(r["err_MRED"], 1e-4), r["test_nominal"]["mean"] * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("MRED of multiplier (exhaustive, uniform operands)"); ax.set_ylabel("nominal test accuracy [%]"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "acc_vs_mred.png"), dpi=130); plt.close(fig)

# ---------------------------------------------------------------- workload counts (raw)
from features import FS_D, N_D, FS_E, N_E, H_LP, H_BP
from vibgen import N, NCH
fir = NCH * (N_D * len(H_LP) + N * len(H_BP))
bfly = lambda n: (n // 2) * int(np.log2(n))
fft = NCH * (bfly(N_D) + bfly(N_E))
R["workload"] = dict(fir_macs_per_window=int(fir), fft_butterflies_per_window=int(fft),
                     fft_real_mults_per_window=int(fft * 4), nn_macs_per_window=R["macs_total"],
                     nn_share_of_mults=R["macs_total"] / (fir + fft * 4 + R["macs_total"]),
                     nn_cycles_32_lanes=int(sum(int(np.ceil(o / 32)) * i for i, o in zip(SIZES[:-1], SIZES[1:]))))
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1)
log("done")
