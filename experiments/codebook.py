"""Tied codebook tree (srhm.tex Definition codebook, section 8.7): exact beliefs over v symbols.

Model: every node (leaves included) carries a symbol a in {0..v-1}; symbol a has a value c_a in leaf
coordinates (R^d).  Children's symbols a'_i ~ M_i[:, a] independently, the SAME M_i at every height,
the leaf level included (tied).  A leaf is x = c_{a'} + tau * eps.  Root symbol ~ stationary law of
the average kernel.
Learner: maximum likelihood by exact EM, zero labels.  E-step = one upward + one downward sum-product
pass; the M-step uses only sums over grids of grid-local expected statistics (pair counts for M_i,
weighted means for c and tau).  Root prediction E[c_root | leaves] needs no labels: the codebook
carries the naming.
"""
import numpy as np

S = 3
FLOOR = 1e-300


class Codebook:
    def __init__(self, M, c, tau):
        self.M = np.asarray(M, float)        # (S, v, v): M[i, a', a] = P(child_i = a' | parent = a)
        self.c = np.asarray(c, float)        # (v, d)
        self.tau = float(tau)

    @property
    def v(self):
        return self.c.shape[0]

    def stationary(self, iters=500):
        Mb = self.M.mean(0)
        mu = np.ones(self.v) / self.v
        for _ in range(iters):
            mu = Mb @ mu
        mu = mu + 1e-15
        return mu / mu.sum()


def _emission(cb, X):
    """X (N, d) leaves -> normalised likelihood over leaf symbols (N, v) and total log scale."""
    d = X.shape[1]
    lq = -((X[:, None, :] - cb.c[None]) ** 2).sum(-1) / (2 * cb.tau ** 2) \
         - d * np.log(np.sqrt(2 * np.pi) * cb.tau)
    mx = lq.max(1, keepdims=True)
    u = np.exp(lq - mx)
    z = u.sum(1, keepdims=True)
    return u / z, (mx + np.log(z)).sum()


def _leaves(X):
    X = np.asarray(X, float)
    return X[..., None] if X.ndim == 2 else X            # (U, n_leaves, d)


def upward(cb, X):
    X = _leaves(X)
    U, nl, d = X.shape
    L = int(round(np.log(nl) / np.log(S)))
    u, ll = _emission(cb, X.reshape(-1, d))
    ups, msgs = {0: u}, {}
    for h in range(1, L + 1):
        uc = ups[h - 1].reshape(-1, S, cb.v)
        m = np.stack([uc[:, i] @ cb.M[i] for i in range(S)], 1)      # m[:, i, a] = sum_a' u(a') M_i[a', a]
        mx = m.max(2, keepdims=True); m = m / mx; ll += np.log(mx).sum()
        b = m.prod(1) + FLOOR
        z = b.sum(1, keepdims=True); ll += np.log(z).sum()
        msgs[h], ups[h] = m, b / z
    return ups, msgs, ll, L, X.reshape(-1, d)


def estep(cb, X):
    pi = cb.stationary()
    ups, msgs, ll, L, Xf = upward(cb, X)
    ll += np.log(ups[L] @ pi).sum()
    v = cb.v
    pair = np.zeros((S, v, v))                              # pair[i, a, a'] summed over all grids
    o = np.tile(pi, (ups[L].shape[0], 1))
    for h in range(L, 0, -1):
        m = msgs[h]
        uc = ups[h - 1].reshape(-1, S, v)
        o_child = np.zeros((m.shape[0], S, v))
        for i in range(S):
            r = o * np.prod(np.delete(m, i, axis=1), axis=1) + FLOOR
            r /= r.sum(1, keepdims=True)
            Z = (r * (uc[:, i] @ cb.M[i])).sum(1, keepdims=True)    # raw (unscaled) message
            pair[i] += cb.M[i].T * ((r / Z).T @ uc[:, i])
            oc = r @ cb.M[i].T
            o_child[:, i] = oc / oc.sum(1, keepdims=True)
        o = o_child.reshape(-1, v)
    P = o * ups[0]                                          # leaf-symbol posteriors (n_leaves, v)
    P /= P.sum(1, keepdims=True)
    return pair, P, Xf, ll


def mstep(pair, P, Xf, smooth=1e-6):
    M = np.stack([((pair[i] + smooth) / (pair[i] + smooth).sum(1, keepdims=True)).T for i in range(S)])
    w = P.sum(0) + 1e-12
    c = (P.T @ Xf) / w[:, None]
    sse = (P * ((Xf[:, None, :] - c[None]) ** 2).sum(-1)).sum()
    tau = np.sqrt(max(sse / (Xf.shape[0] * Xf.shape[1]), 1e-6))
    return Codebook(M, c, tau)


def random_init(X, v, rng):
    Xf = _leaves(X).reshape(-1, _leaves(X).shape[-1])
    qs = np.quantile(Xf, (np.arange(v) + 0.5) / v, axis=0)
    c = qs + 0.05 * Xf.std(0) * rng.normal(size=qs.shape)
    M = rng.dirichlet(np.ones(v), size=(S, v)).transpose(0, 2, 1)   # columns = distributions
    return Codebook(M, c, 0.5 * Xf.std() / v ** (1 / Xf.shape[1]) + 0.1)


def fit(X, v, rng, restarts=4, iters=250, tol=1e-7):
    best = None
    for _ in range(restarts):
        cb, prev = random_init(X, v, rng), -np.inf
        for _ in range(iters):
            pair, P, Xf, ll = estep(cb, X)
            cb = mstep(pair, P, Xf)
            if ll - prev < tol * abs(ll):
                break
            prev = ll
        ll = estep(cb, X)[3]
        if best is None or ll > best[0]:
            best = (ll, cb)
    return best[1], best[0]


def root_value(cb, X, chunk=40):
    X = _leaves(X)
    pi = cb.stationary(); out = []
    for a in range(0, X.shape[0], chunk):
        ups, _, _, L, _ = upward(cb, X[a:a + chunk])
        p = ups[L] * pi
        out.append((p / p.sum(1, keepdims=True)) @ cb.c)
    return np.concatenate(out)[:, 0] if X.shape[-1] == 1 else np.concatenate(out)
