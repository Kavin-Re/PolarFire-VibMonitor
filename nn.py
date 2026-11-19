"""From-scratch numpy MLP, post-training quantisation and integer inference
with a pluggable (approximate) multiplier look-up table."""
import numpy as np


class Standardizer:
    def __init__(self, clip=6.0):
        self.clip = clip

    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-3
        return self

    def __call__(self, X):
        return np.clip((X - self.mu) / self.sd, -self.clip, self.clip)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X, cache=False):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(z if i == len(self.W) - 1 else np.maximum(z, 0))
        return (acts[-1], acts) if cache else acts[-1]

    def predict(self, X):
        return self.forward(X).argmax(1)

    def fit(self, X, y, Xv, yv, epochs=80, lr=2e-3, bs=64, wd=1e-4, patience=12, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        P = self.W + self.b
        m = [np.zeros_like(p) for p in P]
        v = [np.zeros_like(p) for p in P]
        t = 0
        best, best_state, bad = -1, None, 0
        nL = len(self.W)
        for ep in range(epochs):
            perm = rng.permutation(len(X))
            for s in range(0, len(X), bs):
                idx = perm[s:s + bs]
                logits, acts = self.forward(X[idx], True)
                g = softmax(logits)
                g[np.arange(len(idx)), y[idx]] -= 1
                g /= len(idx)
                gW, gb = [None] * nL, [None] * nL
                for l in range(nL - 1, -1, -1):
                    gW[l] = acts[l].T @ g + wd * self.W[l]
                    gb[l] = g.sum(0)
                    if l > 0:
                        g = (g @ self.W[l].T) * (acts[l] > 0)
                t += 1
                for k, (p, gr) in enumerate(zip(P, gW + gb)):
                    m[k] = 0.9 * m[k] + 0.1 * gr
                    v[k] = 0.999 * v[k] + 0.001 * gr * gr
                    p -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            acc = (self.predict(Xv) == yv).mean()
            if verbose:
                print(ep, round(acc, 4))
            if acc > best:
                best, bad = acc, 0
                best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            else:
                bad += 1
                if bad >= patience:
                    break
        self.W, self.b = best_state
        return best


# --------------------------------------------------------------- quantisation
def quantize(model, Zcal, bits=8, pct=99.9):
    qmax = 2 ** (bits - 1) - 1
    s_in = np.percentile(np.abs(Zcal), pct) / qmax
    s_x, a = s_in, Zcal
    layers = []
    L = len(model.W)
    for l, (W, b) in enumerate(zip(model.W, model.b)):
        s_w = np.abs(W).max() / qmax
        Wq = np.clip(np.round(W / s_w), -qmax, qmax).astype(np.int64)
        bq = np.round(b / (s_x * s_w)).astype(np.int64)
        z = a @ W + b
        d = dict(Wq=Wq, bq=bq, s_w=s_w, s_x=s_x)
        if l < L - 1:
            a = np.maximum(z, 0)
            s_y = np.percentile(a, pct) / qmax
            scale = s_x * s_w / s_y
            e = int(np.floor(np.log2(scale)))
            S = 14 - e
            d.update(M=int(round(scale * 2 ** S)), S=S, s_y=s_y)
            s_x = s_y
        layers.append(d)
    return dict(bits=bits, qmax=qmax, s_in=s_in, layers=layers)


def quantize_input(Z, qm):
    return np.clip(np.round(Z / qm["s_in"]), -qm["qmax"], qm["qmax"]).astype(np.int64)


def dot_lut(x, W, lut, chunk_elems=4_000_000):
    """sum_i sign * LUT[|x_i|,|w_ij|] using sign-magnitude operands."""
    n = lut.shape[1]
    sh = int(np.log2(n))
    flat = lut.ravel().astype(np.int64)
    ax, sx = np.abs(x), np.sign(x)
    aw, sw = np.abs(W), np.sign(W)
    B, I = x.shape
    O = W.shape[1]
    out = np.empty((B, O), np.int64)
    step = max(1, chunk_elems // (I * O))
    for s in range(0, B, step):
        e = min(B, s + step)
        p = flat[(ax[s:e, :, None] << sh) | aw[None, :, :]]
        out[s:e] = (p * (sx[s:e, :, None] * sw[None, :, :])).sum(axis=1)
    return out


def qforward(Xq, qm, lut=None, approx_layers=None):
    """Integer inference; returns int logits. lut=None => exact integer matmul.
    approx_layers: set of layer indices using the LUT multiplier (default all)."""
    x = Xq
    L = len(qm["layers"])
    for l, d in enumerate(qm["layers"]):
        use = lut is not None and (approx_layers is None or l in approx_layers)
        acc = (dot_lut(x, d["Wq"], lut) if use else x @ d["Wq"]) + d["bq"]
        if l < L - 1:
            y = (acc * d["M"] + (1 << (d["S"] - 1))) >> d["S"]
            x = np.clip(y, 0, qm["qmax"])
        else:
            return acc


def count_macs(sizes):
    return [a * b for a, b in zip(sizes[:-1], sizes[1:])]
