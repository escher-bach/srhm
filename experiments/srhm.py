"""Tied (self-similar) Random Hierarchy Model and the bottom-level context learner.

Model SRHM(v, m, s, L): one grammar G = (R_a)_{a in V}, R_a subset of V^s, |R_a| = m,
pairwise disjoint, used at EVERY level (tied).  Root uniform; each node picks one of its m
rules uniformly.  Label = root symbol.  Given G, the root is recovered exactly by parsing
(a discrete scale-tied SRG with state space V + {invalid}).

Learner CTX (zero labels):
  1. cluster bottom tuples by the law of their sibling tuples   (statistical Myhill-Nerode)
  2. name the clusters by self-similarity: the id-tuples one level up must be valid tuples
  3. parse every level with the single learned table.
"""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------- grammar
class Grammar:
    def __init__(self, v, m, s, rng):
        assert v * m <= v ** s
        self.v, self.m, self.s = v, m, s
        tuples = rng.choice(v ** s, size=v * m, replace=False)          # uniform valid set D
        self.R = tuples.reshape(v, m)                                    # R[a] = codes of rules
        self.tab = -np.ones(v ** s, dtype=np.int64)                      # parent of each tuple
        for a in range(v):
            self.tab[self.R[a]] = a
        self.D = np.sort(tuples)
        self.digits = self.decode(np.arange(v ** s))                    # (v^s, s)

    def decode(self, codes):
        out = np.zeros(codes.shape + (self.s,), dtype=np.int64)
        c = codes.copy()
        for i in range(self.s - 1, -1, -1):
            out[..., i] = c % self.v; c //= self.v
        return out

    def encode(self, sym):                                               # (..., s) -> code
        c = np.zeros(sym.shape[:-1], dtype=np.int64)
        for i in range(self.s):
            c = c * self.v + sym[..., i]
        return c

    # ------------------------------------------------------------------ sampling
    def sample(self, P, L, rng, root=None):
        z = rng.integers(0, self.v, size=(P, 1)) if root is None else root.reshape(-1, 1)
        y = z[:, 0].copy()
        for _ in range(L):
            j = rng.integers(0, self.m, size=z.shape)
            z = self.digits[self.R[z, j]].reshape(P, -1)                 # expand every node
        return z, y

    def parse(self, leaves, tab=None):
        tab = self.tab if tab is None else tab
        z = leaves
        while z.shape[1] > 1:
            codes = self.encode(z.reshape(z.shape[0], -1, self.s))
            bad = (z.reshape(z.shape[0], -1, self.s) < 0).any(-1)
            z = np.where(bad, -1, tab[np.where(bad, 0, codes)])
        return z[:, 0]

    # ------------------------------------------------------------------ symmetry
    def automorphisms_of_D(self, max_v=9):
        """|{tau in S_v : tau^{(x)s}(D) = D}| by brute force (small v only)."""
        if self.v > max_v:
            return None
        valid = np.zeros(self.v ** self.s, bool); valid[self.D] = True
        Dd = self.digits[self.D]
        cnt = 0
        for p in itertools.permutations(range(self.v)):
            if valid[self.encode(np.array(p)[Dd])].all():
                cnt += 1
        return cnt

    # ------------------------------------------------------------------ spectral constants
    def channel_constants(self, L):
        """lambda = max maximal correlation over parent->child edges,
        lambda_sib = max maximal correlation over sibling pairs, for a depth-L tree
        with uniform root.  Maximal correlation = 2nd singular value of the normalised joint."""
        v, s = self.v, self.s
        Rd = self.digits[self.R]                                          # (v, m, s)
        Q = np.zeros((s, v, v))                                           # Q[i, x, a]
        for i in range(s):
            for a in range(v):
                np.add.at(Q[i, :, a], Rd[a, :, i], 1.0 / self.m)
        S = np.zeros((s, s, v, v, v))                                     # S[i,j,x,y,a]
        for i in range(s):
            for j in range(s):
                if i != j:
                    for a in range(v):
                        np.add.at(S[i, j, :, :, a], (Rd[a, :, i], Rd[a, :, j]), 1.0 / self.m)

        def mc(J):
            px, py = J.sum(1), J.sum(0)
            kx, ky = px > 1e-15, py > 1e-15
            B = J[np.ix_(kx, ky)] / np.sqrt(np.outer(px[kx], py[ky]))
            sv = np.linalg.svd(B, compute_uv=False)
            return sv[1] if len(sv) > 1 else 0.0

        lam, lam_sib = 0.0, 0.0
        marg = [np.ones(v) / v]
        for depth in range(L):
            nxt = []
            for p in marg:
                for i in range(s):
                    lam = max(lam, mc((Q[i] * p[None, :]).T))
                    nxt.append(Q[i] @ p)
                if depth <= L - 3:          # sibling links used by height-2 statistics
                    for i in range(s):
                        for j in range(s):
                            if i != j:
                                lam_sib = max(lam_sib, mc(S[i, j] @ p))
            marg = nxt
        return lam, lam_sib


def n_eff(L, s, lam, lam_sib):
    """Effective number of height-2 grids per example (generalised Thm 5.5)."""
    q = s * lam ** 2
    B = 1 + (s - 1) * lam_sib * sum(q ** j for j in range(L - 2))
    return s ** (L - 2) / B, q


# ---------------------------------------------------------------------------- learner
def ctx_learn(G, leaves, iters=30, smooth=0.5, restarts=30):
    """Zero-label learner.  Returns a learned parent table over tuple codes."""
    v, s = G.v, G.s
    P = leaves.shape[0]
    bt = G.encode(leaves.reshape(P, -1, s))                               # bottom tuples (P, n1)
    grids = bt.reshape(P, -1, s).reshape(-1, s)                           # height-2 grids
    valid = {}

    def naming_key(obs, lab, inv):   # label-free restart selection: self-similarity violations
        if 'mask' not in valid:
            valid['mask'] = np.zeros(v ** s, bool); valid['mask'][obs] = True
        u2, w2 = np.unique(G.encode(lab[inv]), return_counts=True)
        pi, n_consistent, viol = name_clusters(G, G.decode(u2), w2, valid['mask'])
        return viol, (pi, n_consistent)

    obs, lab, (pi, n_consistent) = cluster_tuples(grids, v, G.m, s, iters, smooth, restarts,
                                                  naming_key)
    tab = -np.ones(v ** s, dtype=np.int64)
    tab[obs] = pi[lab]
    return tab, n_consistent


def cluster_tuples(grids, v, m, s, iters=30, smooth=0.5, restarts=15, key_fn=None):
    """Cluster the tuple codes appearing in `grids` (N, s) into v synonym classes of size m by
    their sibling context (statistical Myhill-Nerode).  Restarts are ranked by key_fn (lower
    is better, label-free) and then by likelihood.  Returns (obs codes, cluster ids, extra)."""
    obs, inv = np.unique(grids, return_inverse=True)
    inv = inv.reshape(grids.shape)
    K = len(obs)
    # --- context counts: C[t, i, j, t'] as a flattened sparse-ish dense matrix
    feats = np.zeros((K, s, s, K))
    for i in range(s):
        for j in range(s):
            if i != j:
                np.add.at(feats, (inv[:, i], i, j, inv[:, j]), 1.0)
    F = feats.reshape(K, -1)
    # co-occurrence embedding (stochastic-block-model view: E[F] has rank <= v per block)
    U, S_, _ = np.linalg.svd(np.sqrt(F), full_matrices=False)
    emb = U[:, :v] * S_[:v]
    rng = np.random.default_rng(0)
    best = None
    for r in range(restarts):
        if r == 0:          # spectral start: balanced k-means in the embedding
            cent = emb[rng.choice(K, size=v, replace=K < v)] + 1e-6 * rng.normal(size=(v, emb.shape[1]))
            for _ in range(20):
                lab = balanced_assign(-((emb[:, None, :] - cent[None]) ** 2).sum(-1), v, m)
                cent = np.array([emb[lab == k].mean(0) if (lab == k).any() else cent[k] for k in range(v)])
        else:               # random balanced start
            lab = balanced_assign(rng.random((K, v)), v, m)
        lab, ll = refine(lab, inv, K, v, s, m, iters, smooth)
        k0, extra = key_fn(obs, lab, inv) if key_fn else (0, None)
        key = (k0, -ll)
        if best is None or key < best[0]:
            best = (key, lab, extra)
    _, lab, extra = best
    return obs, lab, extra


def refine(lab, inv, K, v, s, m, iters, smooth):
    """Hard EM for the block model: contexts expressed in sibling cluster ids."""
    for _ in range(iters):
        Fc = np.zeros((K, s, s, v))
        for i in range(s):
            for j in range(s):
                if i != j:
                    np.add.at(Fc, (inv[:, i], i, j, lab[inv[:, j]]), 1.0)
        Fc = Fc.reshape(K, -1)
        prof = (np.stack([Fc[lab == k].sum(0) for k in range(v)]) + smooth).reshape(v, s * s, v)
        prof = (prof / prof.sum(-1, keepdims=True)).reshape(v, -1)
        ll = Fc @ np.log(prof).T                                          # (K, v)
        new = balanced_assign(ll, v, m)
        if np.array_equal(new, lab):
            break
        lab = new
    return lab, ll[np.arange(K), lab].sum()


def balanced_assign(score, v, m):
    """Assign rows to v clusters with capacity m each, maximising total score."""
    K = score.shape[0]
    cap = max(m, -(-K // v))
    cost = -np.repeat(score, cap, axis=1)
    r, c = linear_sum_assignment(cost)
    lab = np.empty(K, dtype=np.int64); lab[r] = c // cap
    return lab


_PERMS = {}


def name_clusters(G, id_tuples, weights, valid_obs):
    """Find bijection pi: cluster id -> symbol s.t. pi(id tuple) is a valid tuple.
    Brute force over S_v (v <= 9); returns the best pi and the number of perfectly
    consistent namings (1 = identified with zero labels)."""
    v = G.v
    if v not in _PERMS:
        _PERMS[v] = np.array(list(itertools.permutations(range(v))))
    perms = _PERMS[v]
    viol = np.zeros(len(perms))
    for t, w in zip(id_tuples, weights):
        codes = G.encode(perms[:, t])                                     # (v!,)
        viol += w * (~valid_obs[codes])
    best = perms[np.argmin(viol)]
    return best, int((viol == 0).sum()), viol.min()
