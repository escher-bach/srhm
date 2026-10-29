"""Claim 2 (srhm.tex section 9): the label signal of additive (degree-1) leaf features.

For each of the 8 tied-RHM grammars of srhm Table 1 (v=8, m=4, s=2) this computes, exactly:
  q      = s * lambda^2, lambda = max maximal correlation of a parent-child edge (old constant)
  ||A_j||  where A_j = sum over root-to-depth-j paths of K_pi K_pi^*  (centred channel operators)
  q_1    = growth rate of ||A_j||  (the "linear Kesten-Stigum number")
  alpha  = additive conditional-variance constant of the kernel (min over node marginals, depth <= 8)
Theorem 9.1 bound:  sup corr(g(root), additive F(leaves))^2  <=  ||A_L|| / alpha.

Then, for grammar 0 (the grammar of srhm Table 3), a Monte Carlo check of that inequality: the
held-out top canonical correlation between the root and one-hot leaves (least squares on half the
sample), for L = 2..5; and P*_backprop * ||A_L|| using the backprop label counts of Table 3.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srhm import Grammar

V, M, S = 8, 4, 2


def channels(G):
    Rd = G.digits[G.R]
    Q = np.zeros((S, V, V))                                   # Q[i, x, a] = P(child_i = x | a)
    for i in range(S):
        for a in range(V):
            np.add.at(Q[i, :, a], Rd[a, :, i], 1.0 / M)
    return Q, Rd


def whitened(Qi, p):
    """Centred channel operator L^2_0(child marginal) -> L^2_0(p) in orthonormal coordinates."""
    c = Qi @ p
    B = np.sqrt(p)[:, None] * Qi.T / np.sqrt(np.maximum(c, 1e-300))[None, :]
    B[:, c < 1e-15] = 0
    return B - np.outer(np.sqrt(p), np.sqrt(c)), c


def A_op(Q, p, j):
    if j == 0:
        return np.eye(V) - np.outer(np.sqrt(p), np.sqrt(p))
    out = np.zeros((V, V))
    for i in range(S):
        W, c = whitened(Q[i], p)
        out += W @ A_op(Q, c, j - 1) @ W.T
    return out


def alpha_at(Rd, p):
    """min over centred additive b=(b_1..b_s) of E Var(sum_i b_i(z_i) | z) / sum_i Var(b_i)."""
    n = S * V
    Nq = np.zeros((n, n)); Dq = np.zeros((n, n))
    for a in range(V):
        if p[a] < 1e-15:
            continue
        E1 = np.zeros(n); E2 = np.zeros((n, n))
        for r in range(M):
            e = np.zeros(n)
            for i in range(S):
                e[i * V + Rd[a, r, i]] = 1.0
            E1 += e / M; E2 += np.outer(e, e) / M
        Nq += p[a] * (E2 - np.outer(E1, E1))
    for i in range(S):
        c = np.zeros(V)
        for a in range(V):
            for r in range(M):
                c[Rd[a, r, i]] += p[a] / M
        blk = np.diag(c) - np.outer(c, c)
        Dq[i * V:(i + 1) * V, i * V:(i + 1) * V] = blk
    w, U = np.linalg.eigh(Dq)
    keep = w > 1e-10
    T = U[:, keep] / np.sqrt(w[keep])                         # whiten D on its range
    return np.linalg.eigvalsh(T.T @ Nq @ T).min()


def alpha_min(Q, Rd, depth=8):
    margs, best = [np.ones(V) / V], np.inf
    for _ in range(depth):
        nxt = []
        for p in margs:
            best = min(best, alpha_at(Rd, p))
            nxt += [Q[i] @ p for i in range(S)]
        margs = nxt
    return best


def heldout_cancorr2(G, L, P, rng):
    X, y = G.sample(P, L, rng)
    F = np.zeros((P, X.shape[1] * V), np.float32)
    F[np.arange(P)[:, None], np.arange(X.shape[1]) * V + X] = 1.0
    Y = np.eye(V, dtype=np.float32)[y]
    h = P // 2
    Fm, Ym = F[:h].mean(0), Y[:h].mean(0)
    B, *_ = np.linalg.lstsq(F[:h] - Fm, Y[:h] - Ym, rcond=None)
    Yt, Yh = Y[h:] - Y[h:].mean(0), (F[h:] - Fm) @ B
    Yh -= Yh.mean(0)

    def isqrt(C):
        w, U = np.linalg.eigh(C); keep = w > 1e-8 * w.max()
        return U[:, keep] / np.sqrt(w[keep])
    C = isqrt(Yt.T @ Yt).T @ (Yt.T @ Yh) @ isqrt(Yh.T @ Yh)
    return float(np.linalg.svd(C, compute_uv=False)[0] ** 2)


def main():
    print("grammar |  q (old) | q_1 (growth of ||A_j||, j=7..12) | alpha")
    for seed in range(8):
        G = Grammar(V, M, S, np.random.default_rng(1000 + seed))
        Q, Rd = channels(G)
        lam, _ = G.channel_constants(8)
        n = [np.linalg.norm(A_op(Q, np.ones(V) / V, j), 2) for j in range(1, 13)]
        q1 = np.exp(np.mean(np.diff(np.log(n[5:]))))
        print(f"   {seed}    |   {S*lam**2:.2f}   |   {q1:.3f}                            | {alpha_min(Q, Rd):.3f}")

    G = Grammar(V, M, S, np.random.default_rng(1000))
    Q, Rd = channels(G)
    al = alpha_min(Q, Rd)
    Pstar = {2: 512, 3: 2048, 4: 4096, 5: 8192, 6: 16384}           # srhm Table 3, untied CNN
    rng = np.random.default_rng(0)
    print("\ngrammar 0: bound vs Monte Carlo (held-out top canonical corr^2, 400k trees)")
    print(" L | ||A_L||  | bound ||A_L||/alpha | MC best additive corr^2 | P*_bp | P*_bp*||A_L||")
    for L in range(2, 7):
        a = np.linalg.norm(A_op(Q, np.ones(V) / V, L), 2)
        mc = heldout_cancorr2(G, L, 400_000, rng) if L <= 5 else float("nan")
        print(f" {L} | {a:.4f}  |   {a/al:.4f}            |   {mc:.4f}                | {Pstar[L]:5d} | {Pstar[L]*a:6.0f}")


if __name__ == "__main__":
    main()
