"""GPU version of codebook.py (tied codebook tree, exact EM).  Same model, same updates, same
statistics; torch instead of numpy so the E-step's batched matrix products run on a GPU.
Probabilities in float32 with per-message rescaling; log-likelihood and M-step sums in float64.
"""
import copy
import math
import numpy as np
import torch

S = 3
FLOOR = 1e-30
DEV = "cuda" if torch.cuda.is_available() else "cpu"
F32, F64 = torch.float32, torch.float64


class Codebook:
    def __init__(self, M, c, tau):
        self.M = torch.as_tensor(M, dtype=F32, device=DEV)      # (S, v, v): M[i, a', a]
        self.c = torch.as_tensor(c, dtype=F32, device=DEV)      # (v, d)
        self.tau = float(tau)

    @property
    def v(self):
        return self.c.shape[0]

    def stationary(self, iters=500):
        Mb = self.M.mean(0).to(F64)
        mu = torch.full((self.v,), 1.0 / self.v, dtype=F64, device=DEV)
        for _ in range(iters):
            mu = Mb @ mu
        mu = mu + 1e-15
        return (mu / mu.sum()).to(F32)

    def numpy(self):
        return self.M.cpu().numpy(), self.c.cpu().numpy(), self.tau


def _leaves(X):
    if not torch.is_tensor(X):
        X = np.asarray(X, float)
    X = torch.as_tensor(X, dtype=F32, device=DEV)
    return X[..., None] if X.dim() == 2 else X


def _emission(cb, X):
    d = X.shape[1]
    lq = -((X[:, None, :] - cb.c[None]) ** 2).sum(-1) / (2 * cb.tau ** 2)
    mx = lq.max(1, keepdim=True).values
    u = torch.exp(lq - mx)
    z = u.sum(1, keepdim=True)
    ll = (mx.to(F64) + torch.log(z.to(F64))).sum() - X.shape[0] * d * math.log(math.sqrt(2 * math.pi) * cb.tau)
    return u / z, ll


def upward(cb, X):
    X = _leaves(X)
    U, nl, d = X.shape
    L = int(round(math.log(nl) / math.log(S)))
    Xf = X.reshape(-1, d)
    u, ll = _emission(cb, Xf)
    ups, msgs = {0: u}, {}
    for h in range(1, L + 1):
        uc = ups[h - 1].reshape(-1, S, cb.v)
        m = torch.stack([uc[:, i] @ cb.M[i] for i in range(S)], 1)
        mx = m.max(2, keepdim=True).values
        m = m / mx; ll = ll + torch.log(mx.to(F64)).sum()
        b = m.prod(1) + FLOOR
        z = b.sum(1, keepdim=True); ll = ll + torch.log(z.to(F64)).sum()
        msgs[h], ups[h] = m, b / z
    return ups, msgs, ll, L, Xf


def estep(cb, X):
    pi = cb.stationary()
    ups, msgs, ll, L, Xf = upward(cb, X)
    ll = ll + torch.log((ups[L] @ pi).to(F64)).sum()
    v = cb.v
    pair = torch.zeros((S, v, v), dtype=F64, device=DEV)
    o = pi.expand(ups[L].shape[0], v)
    for h in range(L, 0, -1):
        m = msgs[h]
        uc = ups[h - 1].reshape(-1, S, v)
        o_child = torch.empty((m.shape[0], S, v), dtype=F32, device=DEV)
        for i in range(S):
            others = torch.ones_like(o)
            for j in range(S):
                if j != i:
                    others = others * m[:, j]
            r = o * others + FLOOR
            r = r / r.sum(1, keepdim=True)
            raw = uc[:, i] @ cb.M[i]                                   # unscaled message
            Z = (r * raw).sum(1, keepdim=True)
            pair[i] += (cb.M[i].T * ((r / Z).T @ uc[:, i])).to(F64)
            oc = r @ cb.M[i].T
            o_child[:, i] = oc / oc.sum(1, keepdim=True)
        o = o_child.reshape(-1, v)
    P = o * ups[0]
    P = P / P.sum(1, keepdim=True)
    return pair, P, Xf, float(ll)


def mstep(pair, P, Xf, smooth=1e-6):
    pr = pair + smooth
    M = (pr / pr.sum(2, keepdim=True)).transpose(1, 2)
    P64, X64 = P.to(F64), Xf.to(F64)
    w = P64.sum(0) + 1e-12
    c = (P64.T @ X64) / w[:, None]
    sse = (P64 * ((X64[:, None, :] - c[None]) ** 2).sum(-1)).sum()
    tau = math.sqrt(max(float(sse) / (Xf.shape[0] * Xf.shape[1]), 1e-6))
    return Codebook(M.to(F32), c.to(F32), tau)


def random_init(X, v, rng):
    Xf = np.asarray(X, float).reshape(-1, 1) if np.asarray(X).ndim == 2 else np.asarray(X, float).reshape(-1, np.asarray(X).shape[-1])
    qs = np.quantile(Xf, (np.arange(v) + 0.5) / v, axis=0)
    c = qs + 0.05 * Xf.std(0) * rng.normal(size=qs.shape)
    M = rng.dirichlet(np.ones(v), size=(S, v)).transpose(0, 2, 1)
    return Codebook(M, c, 0.5 * Xf.std() / v ** (1 / Xf.shape[1]) + 0.1)


def run_em(cb, X, iters, tol):
    prev = -math.inf
    for _ in range(iters):
        pair, P, Xf, ll = estep(cb, X)
        cb = mstep(pair, P, Xf)
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    return cb, estep(cb, X)[3]


def fit(X, v, rng, restarts=4, iters=250, tol=1e-7):
    best = None
    for _ in range(restarts):
        cb, ll = run_em(random_init(X, v, rng), X, iters, tol)
        if best is None or ll > best[0]:
            best = (ll, cb)
    return best[1], best[0]


def split(cb, rng, jitter=0.05):
    """v -> 2v symbols: each symbol becomes two copies with slightly separated values; a parent's
    probability of a child symbol is shared between its two copies (small random asymmetry).
    Both parent copies inherit the parent's children law.  The split model has the same likelihood
    up to the jitter, so EM continues from a model as good as the coarse one."""
    M, c, tau = cb.numpy()
    v = c.shape[0]
    sd = max(tau, 1e-3) * jitter
    c2 = np.concatenate([c - sd, c + sd])
    w = 0.5 + 0.05 * rng.uniform(-1, 1, size=(3, v, v))
    M2 = np.zeros((3, 2 * v, 2 * v))
    for i in range(3):
        child = np.concatenate([M[i] * w[i], M[i] * (1 - w[i])], 0)      # (2v children, v parents)
        M2[i] = np.concatenate([child, child], 1)                          # both parent copies
    return Codebook(M2, c2, tau)


def fit_coarse_to_fine(X, v_final, rng, v0=4, restarts=4, iters=250, tol=1e-7):
    """Maximum likelihood by EM, initialised coarse-to-fine: best of `restarts` at v0, then split and
    refit until v_final.  Returns the fitted models at every size (dict v -> (codebook, ll))."""
    cb, ll = fit(X, v0, rng, restarts, iters, tol)
    out = {v0: (cb, ll)}
    v = v0
    while v < v_final:
        cb, ll = run_em(split(cb, rng), X, iters, tol)
        v *= 2
        out[v] = (cb, ll)
    return out


# ------------------------------------------------------------------ hard-quantized leaves (Q4)
# Leaf symbol = its quantile cell (no emission noise); symbol value = cell mean.  All noise lives in
# the tied M_i at every level, so leaves and latent children are generated the same way.

class QCodebook(Codebook):
    def __init__(self, M, c, edges):
        super().__init__(M, c, 1.0)
        self.edges = torch.as_tensor(edges, dtype=F32, device=DEV)   # (v-1,) cell boundaries (d = 1)


def quantize(Xf, v):
    x = np.asarray(Xf, float).ravel()
    edges = np.quantile(x, np.arange(1, v) / v)
    idx = np.searchsorted(edges, x)
    c = np.array([x[idx == a].mean() for a in range(v)])[:, None]
    return edges, c


def _q_emission(cb, Xf):
    if getattr(cb, "quant", None) is not None:                      # d > 1: tree-structured VQ
        idx = cb.quant.assign(Xf, cb.depth)
    else:
        idx = torch.searchsorted(cb.edges, Xf[:, 0].contiguous())
    u = torch.zeros((Xf.shape[0], cb.v), dtype=F32, device=DEV)
    u[torch.arange(Xf.shape[0], device=DEV), idx] = 1.0
    return u, torch.zeros((), dtype=F64, device=DEV)     # uniform-in-cell density: constant in M


def _with_q(fn):
    def wrapped(cb, X, *a, **k):
        global _emission
        orig = _emission
        if isinstance(cb, QCodebook):
            _emission = _q_emission
        try:
            return fn(cb, X, *a, **k)
        finally:
            _emission = orig
    return wrapped


q_estep = _with_q(estep)
q_root_value = _with_q(lambda cb, X, chunk=200: root_value(cb, X, chunk))


def q_run_em(cb, X, iters, tol, smooth=1e-6):
    prev = -math.inf
    for _ in range(iters):
        pair, P, Xf, ll = q_estep(cb, X)
        pr = pair + smooth
        new = copy.copy(cb)                                          # keeps quantizer / edges
        new.M = (pr / pr.sum(2, keepdim=True)).transpose(1, 2).to(F32)
        cb = new
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    return cb, q_estep(cb, X)[3]


def q_split(cb, X, rng):
    """v -> 2v with nested quantile cells (median split of every cell); M shared as in split()."""
    v = cb.v
    Xf = np.asarray(X, float).ravel()
    edges2, c2 = quantize(Xf, 2 * v)
    M = cb.M.cpu().numpy()
    w = 0.5 + 0.05 * rng.uniform(-1, 1, size=(3, v, v))
    M2 = np.zeros((3, 2 * v, 2 * v))
    order = np.repeat(np.arange(v), 2)            # new cell 2a, 2a+1 refine old cell a
    for i in range(3):
        child = np.zeros((2 * v, v))
        for a2 in range(2 * v):
            child[a2] = M[i][order[a2]] * (w[i][order[a2]] if a2 % 2 == 0 else 1 - w[i][order[a2]])
        M2[i] = child[:, order]
    return QCodebook(M2, c2, edges2)


def q_fit_coarse_to_fine(X, v_final, rng, v0=4, restarts=4, iters=300, tol=1e-8):
    Xf = np.asarray(X, float).ravel()
    edges, c = quantize(Xf, v0)
    best = None
    for _ in range(restarts):
        M = rng.dirichlet(np.ones(v0), size=(3, v0)).transpose(0, 2, 1)
        cb, ll = q_run_em(QCodebook(M, c, edges), X, iters, tol)
        if best is None or ll > best[1]:
            best = (cb, ll)
    cb, ll = best
    out = {v0: (cb, ll)}
    v = v0
    while v < v_final:
        cb, ll = q_run_em(q_split(cb, X, rng), X, iters, tol)
        v *= 2
        out[v] = (cb, ll)
    return out


def root_value(cb, X, chunk=200):
    X = _leaves(X)
    pi = cb.stationary(); out = []
    for a in range(0, X.shape[0], chunk):
        ups, _, _, L, _ = upward(cb, X[a:a + chunk])
        p = ups[L] * pi
        out.append((p / p.sum(1, keepdim=True)) @ cb.c)
    r = torch.cat(out).cpu().numpy()
    return r[:, 0] if r.shape[1] == 1 else r
