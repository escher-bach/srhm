"""Partially tied Random Hierarchy Model PT-RHM(rho) and a zero-label level-wise learner.

Model: base grammar R (used at height 1).  At every height h >= 2, each of the v*m rule slots
(a, j) independently keeps its base rule with probability rho, else is replaced by a FRESH
tuple drawn (without replacement, per level) from V^s \\ D.  rho = 1 is the tied RHM,
rho = 0 is (close to) the untied RHM.

Learner (zero labels):
  1. bottom: context clustering + naming by self-similarity   (srhm.ctx_learn)
  2. heights h = 2..L-1: parse level h-1 with what is known; tuples outside the learned valid
     set are deviations; assign each deviation seen >= N0 times to the base class whose
     sibling-context profile at height h it matches best (profiles estimated from base tuples
     at the same height, with the same filtering).
  3. height L (root, no context): elimination -- a fresh top tuple is assigned to the class
     with the most base rules never observed at the top.
"""
import itertools
import numpy as np
from srhm import Grammar, ctx_learn, cluster_tuples

_PERMS = {}


def structural_naming(G, X, obs1, lab1, restarts=15):
    """Name bottom clusters by matching PARTITIONS across scales (robust to partial tying).

    The level-2 id-tuples are clustered by their own sibling context, giving the level-2
    synonym partition B.  For each candidate naming pi (cluster id -> symbol), the base
    grammar induces a partition A_pi of the same id-tuples (u ~ u' iff pi(u), pi(u') are
    base synonyms).  Self-similarity says A_pi = B on the tied part for the true pi, while a
    wrong pi breaks almost every synonym pair it touches.  Score = #pairs co-grouped in both.
    Returns (pi, score margin between best and second-best naming)."""
    v, m, s = G.v, G.m, G.s
    P = X.shape[0]
    lut = -np.ones(v ** s, dtype=np.int64); lut[obs1] = lab1
    ids = lut[G.encode(X.reshape(P, -1, s))]                             # height-1 cluster ids
    idt = ids.reshape(P, -1, s)
    code2 = np.where((idt >= 0).all(-1), G.encode(np.maximum(idt, 0)), -1)   # height-2 id-tuples
    grids2 = code2.reshape(P, -1, s).reshape(-1, s)                      # height-3 grids
    grids2 = grids2[(grids2 >= 0).all(1)]
    obs2, lab2, _ = cluster_tuples(grids2, v, m, s, restarts=restarts)
    if v not in _PERMS:
        _PERMS[v] = np.array(list(itertools.permutations(range(v))))
    perms = _PERMS[v]
    A = lut[G.encode(perms[:, G.decode(obs2)])]                          # (v!, K2) base class
    idx = np.where(A >= 0, A * v + lab2[None, :], v * v)
    nb = v * v + 1
    C = np.bincount((idx + nb * np.arange(len(perms))[:, None]).ravel(),
                    minlength=len(perms) * nb).reshape(len(perms), nb)[:, :v * v]
    score = (C * (C - 1)).sum(1)
    order = np.argsort(-score)
    return perms[order[0]], int(score[order[0]] - score[order[1]])


class PartialGrammar:
    def __init__(self, v, m, s, L, rho, rng):
        self.base = G = Grammar(v, m, s, rng)
        self.v, self.m, self.s, self.L, self.rho = v, m, s, L, rho
        outside = np.setdiff1d(np.arange(v ** s), G.D)
        self.R = {1: G.R.copy()}
        self.tabs = {1: G.tab.copy()}
        self.fresh = {1: np.zeros((v, m), bool)}
        for h in range(2, L + 1):
            keep = rng.random((v, m)) < rho
            R = G.R.copy()
            new = rng.choice(outside, size=int((~keep).sum()), replace=False)
            R[~keep] = new
            tab = -np.ones(v ** s, dtype=np.int64)
            for a in range(v):
                tab[R[a]] = a
            self.R[h], self.tabs[h], self.fresh[h] = R, tab, ~keep

    def sample(self, P, rng):
        G, z = self.base, rng.integers(0, self.v, size=(P, 1))
        y = z[:, 0].copy()
        for h in range(self.L, 0, -1):                       # root uses R[L], bottom uses R[1]
            j = rng.integers(0, self.m, size=z.shape)
            z = G.digits[self.R[h][z, j]].reshape(P, -1)
        return z, y

    def parse(self, leaves, tabs, upto=None):
        G, z = self.base, leaves
        for h in range(1, (upto or self.L) + 1):
            grp = z.reshape(z.shape[0], -1, self.s)
            bad = (grp < 0).any(-1)
            z = np.where(bad, -1, tabs[h][np.where(bad, 0, G.encode(grp))])
        return z


def pt_learn(PG, X, N0=8, smooth=0.5, naming="structural"):
    G, v, m, s, L = PG.base, PG.v, PG.m, PG.s, PG.L
    if naming == "structural":
        P = X.shape[0]
        grids = G.encode(X.reshape(P, -1, s)).reshape(-1, s)
        obs1, lab1, _ = cluster_tuples(grids, v, m, s, restarts=15)
        pi, _ = structural_naming(G, X, obs1, lab1)
        tab1 = -np.ones(v ** s, dtype=np.int64); tab1[obs1] = pi[lab1]
    else:                                                     # validity naming (srhm.pdf)
        tab1, _ = ctx_learn(G, X, restarts=15)
    known = tab1 >= 0                                          # learned valid set D_hat
    tabs = {1: tab1}
    z = PG.parse(X, tabs, upto=1)                              # height-1 symbols
    for h in range(2, L + 1):
        grp = z.reshape(z.shape[0], -1, s)
        bad = (grp < 0).any(-1)
        codes = np.where(bad, -1, G.encode(np.where(grp < 0, 0, grp)))   # height-h tuples
        tab = tab1.copy()
        dev = np.setdiff1d(np.unique(codes[codes >= 0]), np.flatnonzero(known))
        if h < L and len(dev):
            sib = codes.reshape(codes.shape[0], -1, s)         # height-(h+1) grids of tuples
            par = np.where(sib >= 0, tab1[np.maximum(sib, 0)], -1)
            par = np.where(np.isin(sib, dev), -1, par)         # deviations: parent unknown yet
            prof = np.full((v, s, s, v), smooth)
            ctx = {e: np.zeros((s, s, v)) for e in dev}
            for i in range(s):
                for j in range(s):
                    if i == j:
                        continue
                    ok = par[..., j] >= 0
                    a_i, t_i, x_j = par[..., i][ok], sib[..., i][ok], par[..., j][ok]
                    base_ok = a_i >= 0
                    np.add.at(prof, (a_i[base_ok], i, j, x_j[base_ok]), 1.0)
                    for e in dev:
                        sel = t_i == e
                        if sel.any():
                            np.add.at(ctx[e], (i, j, x_j[sel]), 1.0)
            # capacity: every class has exactly m rules per level, so the number of fresh slots of
            # class a is m minus the base rules of a observed at this height (label-free)
            seen_base = np.unique(codes[(codes >= 0) & known[np.maximum(codes, 0)]])
            cap = m - np.bincount(tab1[seen_base], minlength=v)
            ready = [e for e in dev if ctx[e].sum() >= N0]
            if ready:
                Ce = np.stack([ctx[e] for e in ready])                  # (n_dev, s, s, v)
                lab = None
                for _ in range(10):      # EM: profiles include currently assigned deviations
                    pr = prof.copy()
                    if lab is not None:
                        np.add.at(pr, lab, Ce)
                    logp = np.log(pr / pr.sum(-1, keepdims=True)).reshape(v, -1)
                    score = Ce.reshape(len(ready), -1) @ logp.T          # (n_dev, v)
                    new = capacity_assign(score, cap)
                    if lab is not None and np.array_equal(new, lab):
                        break
                    lab = new
                for e, a in zip(ready, lab):
                    tab[e] = int(a)
        elif h == L and len(dev):
            seen = np.isin(G.R, np.unique(codes[codes >= 0]))  # base rules seen at the top
            missing = (~seen & known[G.R]).sum(1)
            tab[dev] = int(np.argmax(missing))
        tabs[h] = tab
        z = np.where(codes >= 0, tab[np.maximum(codes, 0)], -1)
    return tabs


def capacity_assign(score, cap):
    """Assign rows to classes maximising score, class a receiving at most cap[a] rows
    (falls back to unconstrained argmax if the capacities cannot hold all rows)."""
    from scipy.optimize import linear_sum_assignment
    cap = np.maximum(cap, 0)
    if cap.sum() < score.shape[0]:
        return score.argmax(1)
    cols = np.repeat(np.arange(len(cap)), cap)
    r, c = linear_sum_assignment(-score[:, cols])
    lab = np.empty(score.shape[0], dtype=np.int64); lab[r] = cols[c]
    return lab


def evaluate(PG, tabs, n, rng):
    X, y = PG.sample(n, rng)
    root = PG.parse(X, tabs)[:, 0]
    return float((root != y).mean())
