"""learn.tex: the tree from histories.

  delta_hat(a, b) = B_hat(a, b) - (W_hat(a) + W_hat(b)) / 2      (unbiased for |mu_a - mu_b|^2 / 2)
  agglomerate: repeatedly merge the two clusters with the smallest average delta_hat (average linkage)

  python treelearn.py syn     synthetic check of learn.tex sec. syn (K1, K2)
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "treelearn_results.jsonl")
T0 = time.time()


def log(**k):
    k["t"] = round(time.time() - T0, 1); print(json.dumps(k), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(k) + "\n")


def delta_hat(K):
    """K (A, C) count matrix of A nodes (each with >= 2 units). Returns the (A, A) matrix of unbiased distances."""
    n = K.sum(1)
    P = K / n[:, None]
    W = 1 - (K * (K - 1)).sum(1) / (n * (n - 1))
    D = 1 - P @ P.T - 0.5 * (W[:, None] + W[None, :])
    np.fill_diagonal(D, 0.0)
    return D


def average_linkage(D):
    """merge order from a distance matrix. Returns the list of merges (i, j, height, members) and, for any k,
    the flat clustering with k clusters via cut()."""
    A = len(D); D = D.astype(float).copy(); np.fill_diagonal(D, np.inf)
    size = np.ones(A); alive = np.ones(A, bool); members = {i: [i] for i in range(A)}
    merges = []
    for _ in range(A - 1):
        idx = np.nonzero(alive)[0]
        sub = D[np.ix_(idx, idx)]
        f = int(np.argmin(sub)); i, j = idx[f // len(idx)], idx[f % len(idx)]
        if i > j: i, j = j, i
        h = D[i, j]
        new = (size[i] * D[i] + size[j] * D[j]) / (size[i] + size[j])       # average linkage update
        D[i, :] = new; D[:, i] = new; D[i, i] = np.inf
        alive[j] = False; D[j, :] = np.inf; D[:, j] = np.inf
        members[i] = members[i] + members.pop(j); size[i] += size[j]
        merges.append((int(i), int(j), float(h), list(members[i])))
    return merges


def cut(merges, A, k):
    """labels (A,) of the clustering with k clusters"""
    lab = np.arange(A)
    for i, j, h, mem in merges[: A - k]:
        lab[np.array(mem)] = i
    return lab


def same_partition(l1, l2):
    return len(set(zip(l1.tolist(), l2.tolist()))) == len(set(l1.tolist())) == len(set(l2.tolist()))


# ------------------------------------------------------------------------------------------------ synthetic
def dirichlet_tree(rng, C, alpha, depth=3, branch=3):
    R = np.full(C, 1.0 / C)
    mu = [rng.dirichlet(alpha * R)[None, :]]                                  # root, centred on R
    for d in range(depth):
        par = np.repeat(mu[-1], branch, axis=0)
        mu.append(np.stack([rng.dirichlet(alpha * p) for p in par]))
    return mu                                                                  # mu[d]: nodes at depth d


def run_syn():
    rng = np.random.default_rng(0)
    C, alpha, depth, branch = 50, 8.0, 3, 3
    V = 1 - 1.0 / C; r1, r2 = alpha / (alpha + 1), 1 / (alpha + 1)
    ell = {d: 0.5 * V * r1 ** d * r2 for d in (1, 2, 3)}
    want = {2: 2 * ell[3], 1: 2 * (ell[3] + ell[2]), 0: 2 * (ell[3] + ell[2] + ell[1])}
    A = branch ** depth
    idx = np.arange(A); lca = np.zeros((A, A), int)
    for a in range(A):
        for b in range(A):
            lca[a, b] = 3 if a == b else (2 if a // 3 == b // 3 else (1 if a // 9 == b // 9 else 0))
    # (K1) the metric
    acc = {0: [], 1: [], 2: []}; bias = []
    for _ in range(400):
        mu = dirichlet_tree(rng, C, alpha)[-1]
        real = 0.5 * ((mu[:, None, :] - mu[None, :, :]) ** 2).sum(2)
        for d in acc: acc[d].append(real[lca == d].mean())
        K = np.stack([rng.multinomial(10, p) for p in mu]).astype(float)
        bias.append((delta_hat(K) - real)[lca < 3].mean())
    log(data="K1", predicted={d: round(want[d], 5) for d in want}, measured={d: round(float(np.mean(acc[d])), 5) for d in acc},
        ratio={d: round(float(np.mean(acc[d]) / want[d]), 4) for d in acc},
        dhat_bias=round(float(np.mean(bias)), 6), dhat_bias_se=round(float(np.std(bias) / np.sqrt(len(bias))), 6))
    # (K2) resolution
    truth9, truth3 = idx // 3, idx // 9
    res = {}
    for n in (2, 3, 5, 10, 20, 80):
        ok = 0; trials = 300
        for _ in range(trials):
            mu = dirichlet_tree(rng, C, alpha)[-1]
            K = np.stack([rng.multinomial(n, p) for p in mu]).astype(float)
            m = average_linkage(delta_hat(K))
            ok += same_partition(cut(m, A, 9), truth9) and same_partition(cut(m, A, 3), truth3)
        res[n] = ok / trials
    # the same with the realised distances (no sampling noise): the ceiling set by one realisation
    ok = 0
    for _ in range(300):
        mu = dirichlet_tree(rng, C, alpha)[-1]
        real = 0.5 * ((mu[:, None, :] - mu[None, :, :]) ** 2).sum(2)
        m = average_linkage(real)
        ok += same_partition(cut(m, A, 9), truth9) and same_partition(cut(m, A, 3), truth3)
    log(data="K2", n_star=round(3 * alpha * np.sqrt(2 / (C - 1)), 2), success_by_n=res, success_realised_distances=ok / 300)



def run_syn_predict():
    """(K3) learned tree against true and flat trees, by predictive loss with the exact recursion (CtxTree)"""
    from code_ctx import CtxTree
    rng = np.random.default_rng(1)
    C, alpha, A = 50, 8.0, 27
    leaf = np.arange(A); grid = (1, 2, 4, 8, 16, 32, 64)

    def loss(rows_fn, Ktr, Kva, Kte):
        """rows_fn(leaf ids) -> (list of row codes coarse to fine, bases); one tied coupling chosen on validation units"""
        def units(K):
            l, x = np.nonzero(K); rep = K[l, x].astype(int)
            return np.repeat(l, rep), np.repeat(x, rep)
        ltr, xtr = units(Ktr); lva, xva = units(Kva); lte, xte = units(Kte)
        rows, bases = rows_fn(ltr); tree = CtxTree(rows, bases, xtr, C)
        best = None
        for a in grid:
            tree.tables([1.0] + [a] * tree.L, True)
            v = float(-np.log(tree.predict(rows_fn(lva)[0], xva)).mean())
            if best is None or v < best[0]: best = (v, a)
        tree.tables([1.0] + [best[1]] * tree.L, True)
        return float(-np.log(tree.predict(rows_fn(lte)[0], xte)).mean())

    def tree_rows(g3, g9):                                  # 3-level tree from cluster labels of the 27 leaves
        return lambda l: ([g3[l], g3[l] * 64 + g9[l], (g3[l] * 64 + g9[l]) * 64 + l], [1, 64, 64])
    flat_rows = lambda l: ([l], [1])
    top_rows = lambda g3: (lambda l: ([g3[l], g3[l] * 64 + l], [1, 64]))

    out = {}
    for n in (2, 5, 10, 20, 40, 160):
        acc = {k: [] for k in ("flat", "true", "learned", "learned_top")}
        for _ in range(60):
            mu = dirichlet_tree(rng, C, alpha)[-1]
            Ktr = np.stack([rng.multinomial(n, p) for p in mu]).astype(float)
            Kva = np.stack([rng.multinomial(20, p) for p in mu]).astype(float)
            Kte = np.stack([rng.multinomial(200, p) for p in mu]).astype(float)
            m = average_linkage(delta_hat(Ktr))
            g9, g3 = cut(m, A, 9), cut(m, A, 3)
            # nest: a 9-cluster belongs to the 3-cluster of its first member (cuts of one dendrogram are nested)
            acc["flat"].append(loss(flat_rows, Ktr, Kva, Kte))
            acc["true"].append(loss(tree_rows(leaf // 9, leaf // 3), Ktr, Kva, Kte))
            acc["learned"].append(loss(tree_rows(g3, g9), Ktr, Kva, Kte))
            acc["learned_top"].append(loss(top_rows(g3), Ktr, Kva, Kte))
        mean = {k: float(np.mean(v)) for k, v in acc.items()}
        gain = mean["flat"] - mean["true"]
        out[n] = dict({k: round(v, 4) for k, v in mean.items()},
                      gamma=round((mean["flat"] - mean["learned"]) / gain, 3),
                      gamma_top=round((mean["flat"] - mean["learned_top"]) / gain, 3))
        log(data="K3", n=n, **out[n])



def run_syn_tied():
    """(K4) one topology shared by R realisations: learn the tree from the averaged distances"""
    from code_ctx import CtxTree
    rng = np.random.default_rng(2)
    C, alpha, A = 50, 8.0, 27
    leaf = np.arange(A); grid = (1, 2, 4, 8, 16, 32, 64)
    truth9, truth3 = leaf // 3, leaf // 9

    def units(K):
        l, x = np.nonzero(K); rep = K[l, x].astype(int)
        return np.repeat(l, rep), np.repeat(x, rep)

    def loss(rows_fn, Ktr, Kva, Kte):
        ltr, xtr = units(Ktr); lva, xva = units(Kva); lte, xte = units(Kte)
        rows, bases = rows_fn(ltr); tree = CtxTree(rows, bases, xtr, C)
        best = None
        for a in grid:
            tree.tables([1.0] + [a] * tree.L, True)
            v = float(-np.log(tree.predict(rows_fn(lva)[0], xva)).mean())
            if best is None or v < best[0]: best = (v, a)
        tree.tables([1.0] + [best[1]] * tree.L, True)
        return float(-np.log(tree.predict(rows_fn(lte)[0], xte)).mean())

    tree_rows = lambda g3, g9: (lambda l: ([g3[l], g3[l] * 64 + g9[l], (g3[l] * 64 + g9[l]) * 64 + l], [1, 64, 64]))
    flat_rows = lambda l: ([l], [1])
    for n, Rs in ((5, (1, 4, 16, 64)), (2, (1, 16, 64, 256))):
        for R in Rs:
            trials = 30; exact = 0; acc = {k: [] for k in ("flat", "true", "learned")}
            for _ in range(trials):
                Dbar = np.zeros((A, A)); reps = []
                for r in range(R):
                    mu = dirichlet_tree(rng, C, alpha)[-1]
                    Ktr = np.stack([rng.multinomial(n, p) for p in mu]).astype(float)
                    Dbar += delta_hat(Ktr) / R
                    if r < 8: reps.append((mu, Ktr))                 # predictive loss on up to 8 realisations
                m = average_linkage(Dbar)
                g9, g3 = cut(m, A, 9), cut(m, A, 3)
                exact += same_partition(g9, truth9) and same_partition(g3, truth3)
                for mu, Ktr in reps:
                    Kva = np.stack([rng.multinomial(20, p) for p in mu]).astype(float)
                    Kte = np.stack([rng.multinomial(200, p) for p in mu]).astype(float)
                    acc["flat"].append(loss(flat_rows, Ktr, Kva, Kte))
                    acc["true"].append(loss(tree_rows(truth3, truth9), Ktr, Kva, Kte))
                    acc["learned"].append(loss(tree_rows(g3, g9), Ktr, Kva, Kte))
            mean = {k: float(np.mean(v)) for k, v in acc.items()}
            log(data="K4", n=n, R=R, exact_recovery=round(exact / trials, 3), **{k: round(v, 4) for k, v in mean.items()},
                gamma=round((mean["flat"] - mean["learned"]) / (mean["flat"] - mean["true"]), 3))



# ------------------------------------------------------------------------------------------------ programs
def feature_pool(d):
    """eight functions of earlier units, each with its number of values (learn.tex sec. prog)"""
    import astdata
    n = len(d["x"]); C, T = len(d["symbols"]), len(d["contexts"])
    x = d["x"].astype(np.int64); t = d["t"].astype(np.int64); par = d["par"].astype(np.int64); node = d["node"].astype(np.int64)
    prev, tp = astdata.features(d)
    o = np.lexsort((np.arange(n), t, par)); po, to = par[o], t[o]
    newg = np.r_[True, (po[1:] != po[:-1]) | (to[1:] != to[:-1])]
    start = np.maximum.accumulate(np.where(newg, np.arange(n), 0))
    pos = np.empty(n, np.int64); pos[o] = np.minimum(np.arange(n) - start, 3)
    prev2 = np.full(n, C, np.int64); two = (np.arange(n) - start) >= 2
    prev2[o[two]] = x[o[np.nonzero(two)[0] - 2]]
    creator = np.full(len(d["node_depth"]), -1, np.int64); creator[node[node >= 0]] = np.nonzero(node >= 0)[0]
    cp = creator[par]; has = cp >= 0; cq = np.maximum(cp, 0)
    lastx = np.r_[C, x[:-1]]; lastx[np.r_[True, d["ufile"][1:] != d["ufile"][:-1]]] = C
    return {"prev": (prev, C + 1), "prev2": (prev2, C + 1), "pos": (pos, 4), "tp": (tp, T + 1),
            "gpt": (np.where(has, tp[cq], T + 1), T + 2), "uncle": (np.where(has, prev[cq], C + 1), C + 2),
            "ppos": (np.where(has, pos[cq], 4), 5), "lastx": (lastx, C + 1)}


def within(code, x, C):
    """per unit: the unbiased within-node disagreement W_hat of its node, and the node's size"""
    u, inv, k = np.unique(code * C + x, return_inverse=True, return_counts=True)
    nodes, ninv = np.unique(u // C, return_inverse=True)
    n = np.bincount(ninv, weights=k); q = np.bincount(ninv, weights=k * (k - 1.0))
    W = np.where(n > 1, 1 - q / np.maximum(n * (n - 1), 1), np.nan)
    node_of_unit = ninv[inv]
    return W[node_of_unit], n[node_of_unit]


def gain(rows, f, card, x, C):
    """drop in within-row disagreement per unit when rows are refined by f (unbiased; singleton children keep the row's value)"""
    Wr, nr = within(rows, x, C); Wc, nc = within(rows * card + f, x, C)
    Wr = np.where(nr > 1, Wr, 0.0); Wc = np.where(nc > 1, Wc, Wr)
    return float((Wr - Wc).mean())


class DenseTree:
    """CtxTree with rows re-indexed densely at every level, so any number of refinements fits in 64 bits.
    Level j's code is (index of the parent row among the training rows of level j-1) * card_j + feature_j."""
    def __new__(cls, base_rows, feats, x, C):
        from code_ctx import CtxTree, lookup
        self = object.__new__(type("DenseTreeImpl", (CtxTree,), {"encode": cls.encode}))
        rows = [base_rows]; bases = [1]
        for f, card in feats:
            kr_prev = np.unique(rows[-1])
            rows.append(np.searchsorted(kr_prev, rows[-1]) * card + f); bases.append(card)
        L = len(rows); self.L, self.C, self.bases = L, C, bases
        self.kx, self.kr, self.xrow = [None] * L, [None] * L, [None] * L
        self.up_x, self.up_r = [None] * L, [None] * L
        for j in range(L):
            if j == L - 1:
                self.kx[j], cnt = np.unique(rows[j] * C + x, return_counts=True); self.nLx = cnt.astype(float)
            else:
                self.kx[j] = np.unique(rows[j] * C + x)
            self.kr[j] = np.unique(rows[j])
            self.xrow[j] = np.searchsorted(self.kr[j], self.kx[j] // C)
        for j in range(1, L):
            parent = (self.kx[j] // C) // bases[j]                                   # dense index of the parent row
            self.up_x[j] = np.searchsorted(self.kx[j - 1], self.kr[j - 1][parent] * C + self.kx[j] % C)
            self.up_r[j] = self.kr[j] // bases[j]
        self.x_of0 = self.kx[0] % C
        self._lookup = lookup; self.train_rows = rows
        return self

    def encode(self, base_rows, feats):
        """row codes of query units, using the training rows' dense indices (-1 where the parent row was never seen)"""
        rows = [base_rows]
        for j, (f, card) in enumerate(feats, start=1):
            idx = self._lookup(self.kr[j - 1], rows[-1])
            rows.append(np.where(idx >= 0, idx * card + f, -1))
        return rows


def dense(rows):
    return np.unique(rows, return_inverse=True)[1]


def learn_order(base_rows, feats, x, C, max_features=6):
    """the refinement order of learn.tex sec. prog on the given units: repeatedly take the feature with the largest
    unbiased drop in within-row disagreement; stop when none is positive.  feats: name -> (values, cardinality)"""
    rows = dense(base_rows); order = []; remaining = dict(feats)
    while remaining and len(order) < max_features:
        g = {name: gain(rows, f, card, x, C) for name, (f, card) in remaining.items()}
        best = max(g, key=g.get)
        if g[best] <= 0: break
        f, card = remaining.pop(best)
        rows = dense(rows * card + f); order.append(best)
    return order


def run_order():
    import astdata
    from code_ctx import fit_ctx
    d = astdata.load(); C, T = len(d["symbols"]), len(d["contexts"])
    test_f, pool_f = astdata.split(d)
    val = np.random.default_rng(7).choice(pool_f, 20, replace=False); pool_f = np.setdiff1d(pool_f, val)
    tr, va, te = (np.isin(d["ufile"], f) for f in (pool_f, val, test_f))
    feats = feature_pool(d); x = d["x"].astype(np.int64)
    base_row = np.minimum(d["dp"], 3).astype(np.int64) * T + d["t"].astype(np.int64)
    rows = dense(base_row[tr]); order = []
    remaining = dict(feats)
    while remaining:
        g = {name: gain(rows, f[tr], card, x[tr], C) for name, (f, card) in remaining.items()}
        best = max(g, key=g.get)
        log(data="order_step", corpus=astdata.CORPUS, step=len(order) + 1, gains={k: round(v, 5) for k, v in sorted(g.items(), key=lambda kv: -kv[1])})
        if g[best] <= 0: break
        f, card = remaining.pop(best)
        rows = dense(rows * card + f[tr]); order.append(best)
    log(data="order", corpus=astdata.CORPUS, order=order)
    for k in range(1, len(order) + 1):
        sel = [feats[name] for name in order[:k]]
        tree = DenseTree(base_row[tr], [(f[tr], c) for f, c in sel], x[tr], C)
        rva = tree.encode(base_row[va], [(f[va], c) for f, c in sel]); rte = tree.encode(base_row[te], [(f[te], c) for f, c in sel])
        a, v = fit_ctx(tree, rva, x[va], True)
        te_loss = float(-np.log(tree.predict(rte, x[te])).mean())
        log(data="order_eval", corpus=astdata.CORPUS, features=order[:k], finest_rows=int(len(tree.kr[-1])), alpha=[float(z) for z in a],
            val=round(v, 5), test=round(te_loss, 5))



def linkage_weighted(Num, Den):
    """average linkage on pooled evidence: the distance of two clusters is sum(w * delta) / sum(w) over everything
    that compares them; clusters with no common evidence are never merged. Returns merges (i, j, height, members)."""
    A = len(Num); Num = Num.astype(float).copy(); Den = Den.astype(float).copy()
    np.fill_diagonal(Den, 0.0); alive = np.ones(A, bool); members = {i: [i] for i in range(A)}; merges = []
    while True:
        idx = np.nonzero(alive)[0]
        D = np.where(Den[np.ix_(idx, idx)] > 0, Num[np.ix_(idx, idx)] / np.maximum(Den[np.ix_(idx, idx)], 1e-300), np.inf)
        f = int(np.argmin(D))
        if not np.isfinite(D.flat[f]): break
        i, j = sorted((idx[f // len(idx)], idx[f % len(idx)]))
        merges.append((int(i), int(j), float(D.flat[f]), members[i] + members[j]))
        Num[i] += Num[j]; Num[:, i] += Num[:, j]; Den[i] += Den[j]; Den[:, i] += Den[:, j]
        Den[i, i] = 0.0; alive[j] = False; Den[j] = 0.0; Den[:, j] = 0.0
        members[i] = members[i] + members.pop(j)
    return merges


def labels_at(merges, A, ks):
    """for each k in ks: labels of the clustering with k clusters among the items that were ever merged (others: own id)"""
    out = []
    for k in ks:
        lab = np.arange(A)
        for i, j, h, mem in merges[: max(0, len(merges) - (k - 1))]:
            lab[np.array(mem)] = i
        out.append(lab)
    return out


def value_evidence(row, val, x, C, V, min_units=2):
    """per base row: counts K[value, symbol]; yields (values present with >= min_units, count matrix)"""
    o = np.argsort(row, kind="stable"); cut = np.nonzero(np.diff(row[o]))[0] + 1
    for seg in np.split(o, cut):
        vs, inv = np.unique(val[seg], return_inverse=True)
        K = np.zeros((len(vs), C)); np.add.at(K, (inv, x[seg]), 1.0)
        ok = K.sum(1) >= min_units
        if ok.sum() >= 2:
            yield int(row[seg[0]]), vs[ok], K[ok]


def run_values():
    """(P3, P4) a tree over the values of the previous-sibling symbol: shared by all rows, or one per row"""
    import astdata
    from code_ctx import fit_ctx
    d = astdata.load(); C, T = len(d["symbols"]), len(d["contexts"])
    test_f, pool_f = astdata.split(d)
    val_f = np.random.default_rng(7).choice(pool_f, 20, replace=False); pool_f = np.setdiff1d(pool_f, val_f)
    va, te = np.isin(d["ufile"], val_f), np.isin(d["ufile"], test_f)
    x = d["x"].astype(np.int64); prev, tp = astdata.features(d); V = C + 1
    base = np.minimum(d["dp"], 3).astype(np.int64) * T + d["t"].astype(np.int64)
    KS = (4, 16, 64)

    def evaluate(tr, feats):
        tree = DenseTree(base[tr], [(f[tr], c) for f, c in feats], x[tr], C)
        rva = tree.encode(base[va], [(f[va], c) for f, c in feats]); rte = tree.encode(base[te], [(f[te], c) for f, c in feats])
        a, v = fit_ctx(tree, rva, x[va], True)
        return float(-np.log(tree.predict(rte, x[te])).mean()), [float(z) for z in a]

    for n in (len(pool_f), 256, 64, 16):
        for s in range(1 if n == len(pool_f) else 2):
            files = np.random.default_rng(100 * n + s).choice(pool_f, n, replace=False)
            tr = np.isin(d["ufile"], files)
            plain, _ = evaluate(tr, [(prev, V), (tp, T + 1)])
            # shared tree: pool delta_hat over rows
            Num = np.zeros((V, V)); Den = np.zeros((V, V)); per_row = {}
            for r, vs, K in value_evidence(base[tr], prev[tr], x[tr], C, V):
                nn = K.sum(1); w = nn[:, None] * nn[None, :] / (nn[:, None] + nn[None, :])
                Dh = delta_hat(K)
                Num[np.ix_(vs, vs)] += w * Dh; Den[np.ix_(vs, vs)] += w
                per_row[r] = (vs, Dh)
            merges = linkage_weighted(Num, Den)
            shared = [(lab[prev], V) for lab in labels_at(merges, V, KS)]
            sh, a_sh = evaluate(tr, shared + [(prev, V), (tp, T + 1)])
            # untied control: a tree per row, from that row alone; ids are local to the row
            loc = [np.full(len(base), k, np.int64) for k in KS]                  # default id k: value unseen in this row
            row_tab = {}
            for r, (vs, Dh) in per_row.items():
                m = average_linkage(Dh) if len(vs) > 2 else []
                row_tab[r] = (vs, [np.unique(cut(m, len(vs), min(k, len(vs))) if m else np.arange(len(vs)), return_inverse=True)[1] for k in KS])
            for r, (vs, labs) in row_tab.items():
                sel = np.nonzero(base == r)[0]
                j = np.searchsorted(vs, prev[sel]); j = np.minimum(j, len(vs) - 1); hit = vs[j] == prev[sel]
                for li, lab in enumerate(labs):
                    loc[li][sel[hit]] = lab[j[hit]]
            un, a_un = evaluate(tr, [(l, k + 1) for l, k in zip(loc, KS)] + [(prev, V), (tp, T + 1)])
            sizes = [len(np.unique(lab)) for lab in labels_at(merges, V, KS)]
            log(data="values", files=n, seed=s, train_units=int(tr.sum()), plain=round(plain, 5), shared=round(sh, 5), untied=round(un, 5),
                merged_values=len(merges) + 1, alpha_shared=a_sh, alpha_untied=a_un)
            if n == len(pool_f):
                sym = list(d["symbols"]) + ["<START>"]
                lab16 = labels_at(merges, V, (16,))[0]
                big = [c for c, k in zip(*np.unique(lab16, return_counts=True)) if k > 1]
                log(data="clusters16", clusters=[[sym[i] for i in np.nonzero(lab16 == c)[0][:14]] for c in big])



def run_combined():
    """measurement: histories on the finest rows of the learned context tree, for 2, 4 and 6 learned features"""
    import astdata
    from code_ctx import fit_ctx, ScopeFiles, fit_scope
    d = astdata.load(); C, T = len(d["symbols"]), len(d["contexts"])
    test_f, pool_f = astdata.split(d)
    val_f = np.random.default_rng(7).choice(pool_f, 20, replace=False); pool_f = np.setdiff1d(pool_f, val_f)
    va, te = np.isin(d["ufile"], val_f), np.isin(d["ufile"], test_f)
    feats = feature_pool(d); x = d["x"].astype(np.int64)
    base = np.minimum(d["dp"], 3).astype(np.int64) * T + d["t"].astype(np.int64)
    order = ["prev", "tp", "lastx", "gpt", "prev2", "uncle"]
    ident = [dense(base)]                                            # identity of the finest row, for the scope recursion
    for name in order:
        f, card = feats[name]; ident.append(dense(ident[-1] * card + f))
    fmt = lambda a: [("inf" if np.isinf(z) else float(z)) for z in a]
    for n in (len(pool_f), 64, 16, 1):
        files = np.random.default_rng(100 * n).choice(pool_f, n, replace=False)
        tr = np.isin(d["ufile"], files)
        for k in (2, 4, 6):
            sel = [feats[name] for name in order[:k]]
            tree = DenseTree(base[tr], [(f[tr], c) for f, c in sel], x[tr], C)
            rva = tree.encode(base[va], [(f[va], c) for f, c in sel]); rte = tree.encode(base[te], [(f[te], c) for f, c in sel])
            fit_ctx(tree, rva, x[va], True)
            pv, pt = tree.predict(rva, x[va]), tree.predict(rte, x[te])
            Vf, Tf = ScopeFiles(d, val_f, ident[k][va], va), ScopeFiles(d, test_f, ident[k][te], te)
            a, _ = fit_scope(Vf, pv, start=(4, 4, 8, np.inf))
            log(data="combined", files=n, features=k, ctx=round(float(-np.log(pt).mean()), 5), both=round(Tf.nll(pt, a), 5), alpha_scope=fmt(a))


if __name__ == "__main__":
    if "syn" in sys.argv: run_syn()
    if "pred" in sys.argv: run_syn_predict()
    if "tied" in sys.argv: run_syn_tied()
    if "order" in sys.argv: run_order()
    if "values" in sys.argv: run_values()
    if "combined" in sys.argv: run_combined()
