"""Continuous check: the linear-Gaussian self-similar tree.

Model LG(d, n, L): z_root ~ N(0, Sigma0); every node's children z_{wi} = A_i^{(h)} z_w + eps,
eps ~ N(0, Psi).  Tied: A^{(h)} = A for every height h.  Leaves (height 0) observed in R^d.
Task: predict the root vector.

Exact inference = upward Kalman pass on the tree (a linear recursive grid).
Zero-label learner (mirrors the discrete pipeline):
  1. bottom: method of moments on sibling cross-covariances -> B_j = A_j A_1^{-1}   (gauge-fixed
     ratios; the realised latent covariance cancels, like the mixture weights in Thm 5.6)
  2. gauge fixing by self-similarity: level-2 statistics give D_j = A_1 B_j A_1^{-1}, so A_1
     solves the Sylvester system A_1 B_j = D_j A_1 (unique up to scale); the noise covariance
     fixes the scale; a global sign remains (Z_2, the continuous analogue of Aut(D)).
"""
import numpy as np


def random_model(d, n, lam, rng):
    """A_i = lam * Q_i (Q_i Haar orthogonal), Psi = (1 - lam^2) I, Sigma0 = I (stationary)."""
    Q = np.stack([np.linalg.qr(rng.normal(size=(d, d)))[0] for _ in range(n)])
    return lam * Q, (1 - lam ** 2) * np.eye(d), np.eye(d)


class LGTree:
    def __init__(self, maps, Psi, Sigma0, L):
        """maps: dict height h -> (n, d, d) array (map from a height-h node to its children)."""
        self.maps, self.Psi, self.Sigma0, self.L = maps, Psi, Sigma0, L
        self.n, self.d = maps[1].shape[0], maps[1].shape[1]

    def sample(self, P, rng):
        d, n = self.d, self.n
        cS, cP = np.linalg.cholesky(self.Sigma0), np.linalg.cholesky(self.Psi)
        z = (rng.normal(size=(P, 1, d)) @ cS.T)
        root = z[:, 0].copy()
        for h in range(self.L, 0, -1):
            A = self.maps[h]
            kids = [z @ A[i].T + rng.normal(size=z.shape) @ cP.T for i in range(n)]
            z = np.stack(kids, axis=2).reshape(P, -1, d)
        return z, root


def kalman_root(leaves, maps, Psi, Sigma0):
    """Upward Kalman pass. maps: dict h -> (n,d,d). Returns posterior mean of the root,
    the root likelihood precision J and the likelihood means (for Sigma0 estimation)."""
    P, _, d = leaves.shape
    n = maps[1].shape[0]
    mu, Pc = leaves, np.zeros((d, d))
    h = 0
    while mu.shape[1] > 1:
        h += 1
        A = maps[h]
        G = np.linalg.inv(Psi + Pc)
        J = sum(A[i].T @ G @ A[i] for i in range(n))
        grp = mu.reshape(P, -1, n, d)
        info = sum(grp[:, :, i] @ (G @ A[i]) for i in range(n))            # rows: (A^T G mu)^T
        Pc = np.linalg.inv(J)
        mu = info @ Pc                                                     # J^{-1} info
    post = np.linalg.inv(J + np.linalg.inv(Sigma0))
    return (info @ post)[:, 0], J, mu[:, 0]


def _ratio(C, j, n):
    """Least-squares B with C[j,k] = B C[0,k] for all k not in {0, j}."""
    ks = [k for k in range(n) if k not in (0, j)]
    return np.hstack([C[j, k] for k in ks]) @ np.linalg.pinv(np.hstack([C[0, k] for k in ks]))


def _cov(grids):
    N, n, d = grids.shape
    return np.einsum('nid,nje->ijde', grids, grids) / N


def mom_bottom(leaves, n):
    """Step 1: B_j = A_j A_1^{-1}, S' = A_1 S A_1^T, Psi from bottom sibling moments."""
    d = leaves.shape[2]
    X = leaves.reshape(-1, n, d)
    C = _cov(X)
    B = np.stack([np.eye(d)] + [_ratio(C, j, n) for j in range(1, n)])
    Sp = np.mean([C[0, k] @ np.linalg.inv(B[k]).T for k in range(1, n)], axis=0)
    Sp = (Sp + Sp.T) / 2
    Psi = np.mean([C[i, i] - B[i] @ Sp @ B[i].T for i in range(n)], axis=0)
    Psi = _psd((Psi + Psi.T) / 2)
    return B, Psi, _psd(Sp)


def _psd(M, floor=1e-6):
    w, V = np.linalg.eigh((M + M.T) / 2)
    return (V * np.maximum(w, floor)) @ V.T


def gauge_fix(leaves, B, Psi, Sp):
    """Step 2: A_1 from level-2 statistics (Sylvester null space + noise scale), up to sign."""
    P, _, d = leaves.shape
    n = B.shape[0]
    Pinv = np.linalg.inv(Psi)
    Jb = sum(B[i].T @ Pinv @ B[i] for i in range(n))
    Q = np.linalg.inv(np.linalg.inv(Sp) + Jb)
    M = Q @ Jb                                                             # u_hat = M u + noise
    H = Q @ Jb @ Q.T                                                       # noise covariance
    Xg = leaves.reshape(P, -1, n, d)
    U = sum(Xg[:, :, i] @ (Pinv @ B[i]) for i in range(n)) @ Q.T          # height-1 u_hat
    C2 = _cov(U.reshape(-1, n, d))                                        # height-2 grids
    Mi = np.linalg.inv(M)
    D = np.stack([np.eye(d)] + [Mi @ _ratio(C2, j, n) @ M for j in range(1, n)])
    # Sylvester: A1 B_j - D_j A1 = 0  (column-major vec)
    I = np.eye(d)
    S = np.vstack([np.kron(B[j].T, I) - np.kron(I, D[j]) for j in range(1, n)])
    K = np.linalg.svd(S)[2][-1].reshape(d, d, order='F')
    # scale from the noise covariance at level 2
    W = np.mean([np.linalg.inv(D[i]) @ Mi @ C2[i, j] @ Mi.T @ np.linalg.inv(D[j]).T
                 for i in range(n) for j in range(n) if i != j], axis=0)
    Psi_u = np.mean([Mi @ (C2[i, i] - H) @ Mi.T - D[i] @ W @ D[i].T for i in range(n)], axis=0)
    c = np.sqrt(max(np.trace(Psi_u), 1e-12) / np.trace(K @ Psi @ K.T))
    return c * K


def learn(leaves, n, L, root_prior="stationary"):
    """Full zero-label learner: returns (maps, Psi, Sigma0) up to a global sign of the latents.

    root_prior='stationary': self-similarity of the marginals (the root is just another node),
    so Sigma0 = latent covariance at height 1, estimated from all bottom grids.
    root_prior='top': estimate Sigma0 from the root-level posteriors, one sample per tree
    (the continuous analogue of the context-free root; poor for few trees)."""
    B, Psi, Sp = mom_bottom(leaves, n)
    A1 = gauge_fix(leaves, B, Psi, Sp)
    A = np.stack([B[i] @ A1 for i in range(n)])
    maps = {h: A for h in range(1, L + 1)}
    if root_prior == "stationary":
        A1i = np.linalg.inv(A1)
        return maps, Psi, _psd(A1i @ Sp @ A1i.T, 1e-3)
    _, J, mu = kalman_root(leaves, maps, Psi, np.eye(A.shape[1]) * 1e6)
    Sig0 = np.cov(mu.T) - np.linalg.inv(J) if len(mu) > 1 else np.eye(A.shape[1])
    return maps, Psi, _psd(Sig0, 1e-3)


def r2(pred, root):
    return 1 - ((pred - root) ** 2).sum() / (root ** 2).sum()


def sign_fix(pred, root, k=1):
    """Fix the global Z_2 sign with k labelled examples."""
    s = np.sign((pred[:k] * root[:k]).sum())
    return pred * (s if s != 0 else 1)
