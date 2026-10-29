"""Bounded nonlinear Gaussian tied tree (srhm.tex section 8, Theorem ngt) and its pure learner.

Model:  z_root ~ stationary law;  children z_{wi} = phi_i(z_w) + psi * eps,  i = 1..S,  same phi at every
        height;  phi_i(z) = theta_i . basis(sigma(z)),  sigma(z) = z / (1 + |z|);  leaves observed.
Learner: maximum likelihood by EM, zero labels.  Beliefs are exact on a grid; the cell is sum-product.
        E-step = one upward + one downward BP pass.  The M-step uses only sums over grids of grid-local
        expected statistics (Fisher identity): weighted least squares for theta_i, closed form for psi.
"""
import numpy as np

S = 3
G = np.linspace(-4.0, 4.0, 161)                  # latent grid (same type as the leaves)
NB = 5


def sig(z):
    return z / (1.0 + np.abs(z))


def basis(z):
    p = sig(z)
    return np.stack([np.ones_like(p), p, p ** 2, p ** 3, np.tanh(3 * p)], -1)


BG = basis(G)                                     # (n, NB)


class Kernel:
    def __init__(self, theta, psi):
        self.theta = np.asarray(theta, float)     # (S, NB)
        self.psi = float(psi)

    def phi(self, z):
        return basis(z) @ self.theta.T            # (..., S)

    def T(self):
        """Grid transition matrices T_i[g, g'] = P(child_i = G[g'] | parent = G[g])."""
        m = BG @ self.theta.T                     # (n, S)
        out = []
        for i in range(S):
            t = np.exp(-(G[None, :] - m[:, i:i + 1]) ** 2 / (2 * self.psi ** 2))
            out.append(t / t.sum(1, keepdims=True))
        return out

    def stationary(self, T=None, iters=400):
        T = self.T() if T is None else T
        mu = np.ones(len(G)) / len(G)
        for _ in range(iters):
            mu = sum(mu @ t for t in T) / S
        mu = mu + 1e-15
        return mu / mu.sum()


def random_true_kernel(rng, psi=0.35):
    th = np.zeros((S, NB))
    for i in range(S):
        b = rng.uniform(0.6, 1.4) * rng.choice([-1, 1])
        th[i, 0] = rng.uniform(-0.3, 0.3)
        th[i, 1] = b
        th[i, 4] = rng.uniform(-1, 1) * abs(b) / 3 * 0.9
    return Kernel(th, psi)


def sample_trees(ker, U, L, rng, burn=40):
    z = np.zeros(U)
    for _ in range(burn):                         # root from the stationary law
        i = rng.integers(S, size=U)
        z = ker.phi(z)[np.arange(U), i] + ker.psi * rng.normal(size=U)
    root = z.copy()
    z = z[:, None]
    for _ in range(L):
        m = ker.phi(z)                            # (U, nodes, S)
        z = (m + ker.psi * rng.normal(size=m.shape)).reshape(U, -1)
    return z, root


def _leaf_msgs(ker, Xc):
    """Xc (N, S) leaf children of height-1 nodes -> messages (N, S, n) with log scales (N,)."""
    m = BG @ ker.theta.T                                           # (n, S)
    lm = -(Xc[:, :, None] - m.T[None]) ** 2 / (2 * ker.psi ** 2) - np.log(np.sqrt(2 * np.pi) * ker.psi)
    mx = lm.max(2, keepdims=True)
    return np.exp(lm - mx), mx[..., 0].sum(1)


def upward(ker, X, T):
    """Returns per-height lists: msgs[h] (N_h, S, n) messages from children, up[h] (N_h, n) normalised
    beliefs, and the total log-likelihood contribution of the upward normalisers."""
    U, nl = X.shape
    L = int(round(np.log(nl) / np.log(S)))
    msgs, ups, ll = {}, {}, 0.0
    m, lsc = _leaf_msgs(ker, X.reshape(-1, S))
    ll += lsc.sum()
    for h in range(1, L + 1):
        if h > 1:
            uc = ups[h - 1].reshape(-1, S, len(G))                 # children beliefs
            m = np.stack([uc[:, i] @ T[i].T for i in range(S)], 1)
            mx = m.max(2, keepdims=True); m = m / mx; ll += np.log(mx).sum()
        b = m.prod(1) + 1e-300                                    # underflow floor
        z = b.sum(1, keepdims=True)
        ll += np.log(z).sum()
        msgs[h], ups[h] = m, b / z
    return msgs, ups, ll, L


def estep(ker, X):
    """Sufficient statistics for the M-step, summed over all grids of all trees (tied kernel)."""
    T = ker.T()
    pi = ker.stationary(T)
    msgs, ups, ll, L = upward(ker, X, T)
    ll += np.log(ups[L] @ pi).sum()
    n = len(G)
    W = np.zeros((S, n)); Y = np.zeros((S, n)); Y2 = np.zeros((S, n))
    o = np.tile(pi, (ups[L].shape[0], 1))                          # outside messages at height L
    Xc = X.reshape(-1, S)
    for h in range(L, 0, -1):
        m = msgs[h]
        o_child = np.zeros((m.shape[0], S, n))
        for i in range(S):
            others = np.prod(np.delete(m, i, axis=1), axis=1)
            r = o * others + 1e-300                                # underflow floor
            r /= r.sum(1, keepdims=True)
            if h > 1:
                uc = ups[h - 1].reshape(-1, S, n)[:, i]
                Z = (r * (uc @ T[i].T)).sum(1, keepdims=True)         # raw (unscaled) message
                pair = T[i] * ((r / Z).T @ uc)                     # sum_w p_w(g, g')
                W[i] += pair.sum(1); Y[i] += pair @ G; Y2[i] += pair @ G ** 2
                oc = r @ T[i]
                o_child[:, i] = oc / oc.sum(1, keepdims=True)
            else:
                P = r * m[:, i]; P /= P.sum(1, keepdims=True)
                x = Xc[:, i]
                W[i] += P.sum(0); Y[i] += P.T @ x; Y2[i] += P.T @ x ** 2
        o = o_child.reshape(-1, n)
    return W, Y, Y2, ll


def mstep(W, Y, Y2):
    th = np.zeros((S, NB)); sse = 0.0
    for i in range(S):
        A = BG.T @ (W[i][:, None] * BG) + 1e-9 * np.eye(NB)
        th[i] = np.linalg.solve(A, BG.T @ Y[i])
        f = BG @ th[i]
        sse += (Y2[i] - 2 * f * Y[i] + f ** 2 * W[i]).sum()
    return Kernel(th, np.sqrt(max(sse / W.sum(), 1e-4)))


def random_init(rng):
    th = np.zeros((S, NB))
    th[:, 0] = rng.normal(0, 0.2, S)
    th[:, 1] = rng.uniform(0.3, 1.2, S) * rng.choice([-1, 1], S)
    th[:, 2:] = rng.normal(0, 0.1, (S, NB - 2))
    return Kernel(th, 0.8)


def fit(X, rng, restarts=6, iters=300, tol=1e-7):
    best = None
    for _ in range(restarts):
        ker, prev = random_init(rng), -np.inf
        for it in range(iters):
            W, Y, Y2, ll = estep(ker, X)
            ker = mstep(W, Y, Y2)
            if ll - prev < tol * abs(ll):
                break
            prev = ll
        ll = estep(ker, X)[3]
        if best is None or ll > best[0]:
            best = (ll, ker)
    return best[1], best[0]


def root_posterior_mean(ker, X, chunk=40):
    T = ker.T(); pi = ker.stationary(T)
    out = []
    for a in range(0, X.shape[0], chunk):             # chunked: memory, not results
        _, ups, _, L = upward(ker, X[a:a + chunk], T)
        P = ups[L] * pi
        out.append((P / P.sum(1, keepdims=True)) @ G)
    return np.concatenate(out)


def kernel_error(ker, true):
    """RMS over the true stationary law of |phi_hat - phi|, averaged over positions, relative to psi."""
    pi = true.stationary()
    d = (BG @ ker.theta.T - BG @ true.theta.T) ** 2              # (n, S)
    return float(np.sqrt((pi[:, None] * d).sum(0)).mean() / true.psi), abs(ker.psi / true.psi - 1)
