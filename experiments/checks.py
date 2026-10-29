"""Numerical sanity checks for the theorems in srg.tex.

Each check is small (seconds on a laptop CPU). Run:  python verify/checks.py
"""
import itertools, math
import numpy as np

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# helpers: complete n-ary tree of depth L, leaves indexed 0..n^L-1 in base-n
# ---------------------------------------------------------------------------
def lca_height(i, j, n):
    h = 0
    while i != j:
        i //= n; j //= n; h += 1
    return h


def canon(leaves, n):
    """Canonical form of the leaf-labelled unordered tree = output of an
    SRG whose cell map f is injective on multisets (Prop. 2.2)."""
    level = [(x,) for x in leaves]
    while len(level) > 1:
        level = [tuple(sorted(level[k:k + n])) for k in range(0, len(level), n)]
    return level[0]


# ---------------------------------------------------------------------------
# (a) Prop 2.2: invariance group of an exchangeable SRG = Aut(T) (wreath product)
# ---------------------------------------------------------------------------
def check_wreath(n, L):
    m = n ** L
    inputs = list(itertools.product([0, 1], repeat=m))
    ids = {}
    val = np.array([ids.setdefault(canon(s, n), len(ids)) for s in inputs])
    # integer code of each 0/1 input, so a permuted input can be looked up fast
    bits = np.array(inputs)                      # (2^m, m)
    weights = 1 << np.arange(m)[::-1]
    count = 0
    for g in itertools.permutations(range(m)):
        perm_idx = bits[:, list(g)] @ weights
        if np.array_equal(val[perm_idx], val):
            count += 1
    expected = math.factorial(n) ** ((n ** L - 1) // (n - 1))
    return count, expected


# ---------------------------------------------------------------------------
# (b) Thm 2.3: Aut(T)-invariant covariance has L+1 eigenvalues with formula
# ---------------------------------------------------------------------------
def check_scale_spectrum(n, L):
    m = n ** L
    c = rng.normal(size=L + 1); c[0] = abs(c).sum() + 1.0      # any values work
    C = np.array([[c[lca_height(i, j, n)] for j in range(m)] for i in range(m)])
    ev = np.sort(np.linalg.eigvalsh(C))
    Np = [1] + [(n - 1) * n ** (h - 1) for h in range(1, L + 1)]
    pred = []
    for k in range(L):
        sk = sum(Np[h] * c[h] for h in range(k + 1)) - n ** k * c[k + 1]
        pred += [sk] * ((n - 1) * n ** (L - k - 1))
    pred += [sum(Np[h] * c[h] for h in range(L + 1))]
    return np.max(np.abs(ev - np.sort(pred)))


# ---------------------------------------------------------------------------
# (c) Thm 3.1: Myhill-Nerode classes = minimal state count; quotient computes phi
# ---------------------------------------------------------------------------
def subtree_classes(phi, n, L):
    """k_h = number of context-equivalence classes of height-h subtrees."""
    m = n ** L
    allsig = list(itertools.product([0, 1], repeat=m))
    phis = {s: phi(s) for s in allsig}
    ks = []
    for h in range(L + 1):
        size = n ** h
        subs = list(itertools.product([0, 1], repeat=size))
        # signature of a subtree = phi over all (position, context) pairs
        sig = {}
        for s in subs:
            key = []
            for pos in range(0, m, size):
                for ctx in allsig[:: max(1, len(allsig) // 64)] if m > 9 else allsig:
                    t = list(ctx); t[pos:pos + size] = s
                    key.append(phis[tuple(t)])
            sig[s] = tuple(key)
        ks.append(len(set(sig.values())))
    return ks


def check_myhill_nerode():
    out = {}
    n, L = 2, 3
    # random tied tree automaton with 3 states over {0,1}
    X = 3
    tab = rng.integers(0, X, size=(X, X))
    enc = rng.integers(0, X, size=2)
    out_map = rng.integers(0, 2, size=X)

    def phi_auto(s):
        lev = [enc[v] for v in s]
        while len(lev) > 1:
            lev = [tab[lev[k], lev[k + 1]] for k in range(0, len(lev), 2)]
        return out_map[lev[0]]
    out["random 3-state tied automaton, n=2,L=3 (k_h <= 3)"] = subtree_classes(phi_auto, n, L)
    out["parity, n=2,L=3 (expect all 2)"] = subtree_classes(lambda s: sum(s) % 2, 2, 3)

    def recmaj(s):
        lev = list(s)
        while len(lev) > 1:
            lev = [int(sum(lev[k:k + 3]) >= 2) for k in range(0, len(lev), 3)]
        return lev[0]
    out["recursive majority, n=3,L=2 (expect 2,2,2)"] = subtree_classes(recmaj, 3, 2)
    out["global majority, n=3,L=2 (expect 2,4,2: k_1 = n+1)"] = subtree_classes(
        lambda s: int(sum(s) >= 5), 3, 2)
    return out


# ---------------------------------------------------------------------------
# (d) Thm 5.3: variance of bottom-grid averages in a broadcast (BSC) process
# ---------------------------------------------------------------------------
def exact_var_bottom_mean(n, L, lam):
    """g(grid) = mean of its n leaf spins; average over the n^{L-1} bottom grids.
    For the binary symmetric broadcast, E[z_u z_v] = lam^{d(u,v)}."""
    V = (n + n * (n - 1) * lam ** 2) / n ** 2          # Var g for one grid
    N1 = n ** (L - 1)
    tot = N1 * V
    for j in range(1, L):                               # LCA at height j+1
        cov = lam ** (2 * j + 2)                        # leaf-leaf distance 2j+2
        tot += N1 * (n - 1) * n ** (j - 1) * cov
    return tot / N1 ** 2, V


def bound_var(n, L, lam, V):
    q = n * lam ** 2
    return V / n ** (L - 1) * (1 + (1 - 1 / n) * sum(q ** j for j in range(1, L)))


def mc_var(n, L, lam, reps=20000):
    p = (1 + lam) / 2
    z = rng.choice([-1, 1], size=(reps, 1))
    for _ in range(L):
        z = np.repeat(z, n, axis=1)
        flip = rng.random(z.shape) > p
        z = np.where(flip, -z, z)
    return z.mean(axis=1).var()


# ---------------------------------------------------------------------------
# (e) Prop 6.3: node-parallel ES == level-local (stop-gradient) gradient
# ---------------------------------------------------------------------------
def check_es(n=2, L=6, d=3, s=1e-3, draws=4000):
    P = d * n * d + d
    theta = rng.normal(scale=0.6, size=P)
    leaves = rng.normal(size=(n ** L, d))
    nodes_per_h = [n ** (L - h) for h in range(1, L + 1)]
    N = sum(nodes_per_h)
    targets = [rng.normal(size=(k, d)) for k in nodes_per_h]

    def f(C, th):                       # C: (k, n*d), th: (k, P) or (P,)
        th = np.broadcast_to(th, (C.shape[0], P))
        W = th[:, :d * n * d].reshape(-1, d, n * d); b = th[:, d * n * d:]
        return np.tanh(np.einsum('kij,kj->ki', W, C) + b)

    def run(thetas):                    # thetas: list per height of (k,P)
        x = leaves; losses = []
        for h in range(L):
            C = x.reshape(-1, n * d)
            x = f(C, thetas[h])
            losses.append(((x - targets[h]) ** 2).sum(axis=1))
        return losses

    tied = [np.broadcast_to(theta, (k, P)) for k in nodes_per_h]
    total = lambda th: sum(l.sum() for l in run([np.broadcast_to(th, (k, P)) for k in nodes_per_h]))
    eps = 1e-5
    g_full = np.array([(total(theta + eps * e) - total(theta - eps * e)) / (2 * eps)
                       for e in np.eye(P)])
    # local gradient: each node's loss w.r.t. its own copy, inputs frozen
    x = leaves; g_loc = np.zeros(P)
    for h in range(L):
        C = x.reshape(-1, n * d)
        for p_ in range(P):
            e = np.zeros(P); e[p_] = eps
            lp = ((f(C, theta + e) - targets[h]) ** 2).sum()
            lm = ((f(C, theta - e) - targets[h]) ** 2).sum()
            g_loc[p_] += (lp - lm) / (2 * eps)
        x = f(C, theta)
    # node-parallel antithetic ES: one perturbation per node, 2 forward passes
    acc = np.zeros(P)
    for _ in range(draws):
        xi = [rng.normal(size=(k, P)) for k in nodes_per_h]
        lp = run([theta + s * z for z in xi]); lm = run([theta - s * z for z in xi])
        acc += sum(((a - b)[:, None] * z).sum(axis=0) for a, b, z in zip(lp, lm, xi)) / (2 * s)
    g_es = acc / draws
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
    return rel(g_es, g_loc), rel(g_es, g_full), rel(g_loc, g_full), N


if __name__ == "__main__":
    print("(a) Prop 2.2  invariance group of exchangeable SRG")
    for n, L in [(2, 2), (2, 3), (3, 2)]:
        c, e = check_wreath(n, L)
        print(f"    n={n} L={L}: |stabiliser| = {c},  (n!)^((n^L-1)/(n-1)) = {e}")

    print("(b) Thm 2.3  scale spectrum of invariant covariance, max |error|:")
    for n, L in [(2, 4), (3, 3), (4, 2)]:
        print(f"    n={n} L={L}: {check_scale_spectrum(n, L):.2e}")

    print("(c) Thm 3.1  Myhill-Nerode class counts k_h, h=0..L")
    for k, v in check_myhill_nerode().items():
        print(f"    {k}: {v}")

    print("(d) Thm 5.3  Var(bottom-grid mean): exact  vs  bound  (q = n*lam^2)")
    for n, lam in [(3, 0.4), (3, 0.577), (3, 0.8)]:
        for L in [4, 8, 12]:
            ex, V = exact_var_bottom_mean(n, L, lam)
            bd = bound_var(n, L, lam, V)
            print(f"    n={n} lam={lam} q={n*lam**2:.2f} L={L:2d}: exact={ex:.3e} "
                  f"bound={bd:.3e} ratio={bd/ex:.2f}  N_eff={V/ex:.3g}  n^(L-1)={n**(L-1)}")
    ex, _ = exact_var_bottom_mean(3, 4, 0.8)
    print(f"    Monte-Carlo check n=3 L=4 lam=0.8: MC={mc_var(3, 4, 0.8):.3e} exact={ex:.3e}")

    print("(e) Prop 6.3  node-parallel ES vs local vs full backprop gradient")
    a, b, c, N = check_es()
    print(f"    nodes N={N}: |ES-local|/|local|={a:.3f}  |ES-full|/|full|={b:.3f}  "
          f"|local-full|/|full|={c:.3f}")
