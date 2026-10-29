"""Fine-scale context on programs (hist.tex sec. two-trees).

Features of a unit beyond its rule row (g, t):  prev = symbol of the previous sibling in the same field (START if none),
tp = context of the unit that created the parent (ROOT if the parent is the file root).
  python code_ctx.py mi      conditional mutual information of each feature with the unit's symbol (training pool)
"""
import json, os, sys, time
import numpy as np
import astdata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "code_ctx_results.jsonl" if astdata.CORPUS == "stdlib" else f"code_ctx_results_{astdata.CORPUS}.jsonl")
T0 = time.time()


def log(**k):
    k["t"] = round(time.time() - T0, 1); print(json.dumps(k), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(k) + "\n")


features = astdata.features


def cond_mi(x, f, ctx, C, rng):
    """I(x ; f | ctx) in nats per unit: plug-in minus the same with f shuffled inside each context"""
    def plug(ff):
        k_cf = np.unique(ctx * (ff.max() + 1) + ff, return_inverse=True)[1]
        def H(code):
            c = np.unique(code, return_counts=True)[1].astype(float); p = c / c.sum()
            return float(-(p * np.log(p)).sum())
        # I(x; f | c) = H(x, c) + H(f, c) - H(x, f, c) - H(c)
        return H(ctx * C + x) + H(k_cf) - H(k_cf.astype(np.int64) * C + x) - H(ctx)
    o = np.argsort(ctx, kind="stable"); fs = f.copy()
    bounds = np.r_[0, np.nonzero(np.diff(ctx[o]))[0] + 1, len(o)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        fs[o[a:b]] = f[o[a:b]][rng.permutation(b - a)]
    return plug(f) - plug(fs), plug(f)


def run_mi():
    rng = np.random.default_rng(0)
    d = astdata.load()
    test_f, pool = astdata.split(d)
    val = np.random.default_rng(7).choice(pool, 20, replace=False); pool = np.setdiff1d(pool, val)
    prev, tp = features(d)
    um = np.isin(d["ufile"], pool)
    C, T = len(d["symbols"]), len(d["contexts"])
    x = d["x"][um].astype(np.int64); row = np.minimum(d["dp"][um], 3).astype(np.int64) * T + d["t"][um].astype(np.int64)
    pv, tq = prev[um], tp[um]
    r = {}
    r["prev | row"] = cond_mi(x, pv, row, C, rng)
    r["tp | row"] = cond_mi(x, tq, row, C, rng)
    r["prev | row,tp"] = cond_mi(x, pv, row * (T + 1) + tq, C, rng)
    r["tp | row,prev"] = cond_mi(x, tq, row * (C + 1) + pv, C, rng)
    log(data="cond_mi", units=int(um.sum()), **{k: dict(corrected=round(v[0], 4), plugin=round(v[1], 4)) for k, v in r.items()})



# ------------------------------------------------------------------------------------------------ the context tree
from code_cred import rhos


def lookup(keys, q):
    """index of q in the sorted array keys, or -1"""
    i = np.searchsorted(keys, q); i = np.minimum(i, len(keys) - 1)
    return np.where(keys[i] == q, i, -1)


class CtxTree:
    """Shared rule on a tree of rows (hist.tex sec. two).  rows: list of L code arrays, coarse to fine, where
    rows[j+1] // bases[j+1] == rows[j] (a finer row determines its coarser row).  All units sit in the finest rows."""
    def __init__(self, rows, bases, x, C):
        self.L, self.C, self.bases = len(rows), C, bases
        kx, self.inv_leaf, cnt = np.unique(rows[-1] * C + x, return_inverse=True, return_counts=True)
        self.kx = [None] * self.L; self.kr = [None] * self.L; self.xrow = [None] * self.L
        self.up_x = [None] * self.L; self.up_r = [None] * self.L
        self.kx[-1] = kx; self.nLx = cnt.astype(float)
        for j in range(self.L - 1, -1, -1):
            self.kr[j], self.xrow[j] = np.unique(self.kx[j] // C, return_inverse=True)     # rows of level j; row of each x-entry
            if j > 0:
                pk = (self.kx[j] // C // bases[j]) * C + self.kx[j] % C
                self.kx[j - 1], self.up_x[j] = np.unique(pk, return_inverse=True)           # x-entry -> parent x-entry
                self.up_r[j] = None
        for j in range(self.L - 1, 0, -1):
            self.up_r[j] = np.searchsorted(self.kr[j - 1], self.kr[j] // bases[j])          # row -> parent row
        self.x_of0 = self.kx[0] % C

    def tables(self, alpha, exact=True):
        """alpha[d], d = 0..L: coupling of a depth-d node to its parent (depth 0 = the root, to the uniform law)"""
        r1, r2 = rhos(np.asarray(alpha, float), "dirichlet")
        S = [None] * self.L; N = [None] * self.L; U = [None] * self.L
        S[-1] = self.nLx; N[-1] = np.bincount(self.xrow[-1], weights=S[-1])
        for j in range(self.L - 1, -1, -1):
            U[j] = 1.0 / (r1[j + 1] / N[j] + r2[j + 1]) if exact else N[j].copy()
            if j > 0:
                w = U[j][self.xrow[j]] * S[j] / N[j][self.xrow[j]]
                S[j - 1] = np.bincount(self.up_x[j], weights=w, minlength=len(self.kx[j - 1]))
                N[j - 1] = np.bincount(self.up_r[j], weights=U[j], minlength=len(self.kr[j - 1]))
        w = U[0][self.xrow[0]] * S[0] / N[0][self.xrow[0]]
        self.S0 = np.bincount(self.x_of0, weights=w, minlength=self.C); self.N0 = U[0].sum()
        self.S, self.N, self.U, self.alpha, self.r1, self.r2, self.exact = S, N, U, np.asarray(alpha, float), r1, r2, exact

    def predict(self, rows, x):
        """P(x | finest row) for query units; rows as in the constructor"""
        C = self.C; get = lambda arr, i: np.where(i >= 0, arr[np.maximum(i, 0)], 0.0)
        Nq, Uq, Sq = [], [], []
        for j in range(self.L):
            ir = lookup(self.kr[j], rows[j]); ix = lookup(self.kx[j], rows[j] * C + x)
            Nq.append(get(self.N[j], ir)); Uq.append(get(self.U[j], ir)); Sq.append(get(self.S[j], ix))
        ybar = [np.where(Nq[j] > 0, Sq[j] / np.maximum(Nq[j], 1e-300), 0.0) for j in range(self.L)]
        a0 = self.alpha[0]; u = 1.0 / C
        if not self.exact:                                           # interpolated back-off: full counts, own data included
            p = (a0 * u + self.S0[x]) / (a0 + self.N0)
            for j in range(self.L):
                p = (self.alpha[j + 1] * p + Sq[j]) / (self.alpha[j + 1] + Nq[j])
            return p
        Np = np.maximum(self.N0 - Uq[0], 0.0); Sp = np.maximum(self.S0[x] - Uq[0] * ybar[0], 0.0)
        m = (a0 * u + Sp) / (a0 + Np); c = a0 + Np
        for j in range(self.L):
            a = self.r1[j + 1] / (self.r2[j + 1] + 1.0 / c)
            if j < self.L - 1:
                Np = np.maximum(Nq[j] - Uq[j + 1], 0.0); Sp = np.maximum(Sq[j] - Uq[j + 1] * ybar[j + 1], 0.0)
            else:
                Np, Sp = Nq[j], Sq[j]
            m = (a * m + Sp) / (a + Np); c = a + Np
        return m


def test_ctx():
    """exact recursion against the explicit best linear predictor on a random tree of rows"""
    rng = np.random.default_rng(0); C = 3
    for trial in range(6):
        L = 2 + trial % 2; n = int(rng.integers(5, 40))
        bases = [1] + [3] * (L - 1)
        f = [rng.integers(0, 3, n + 1) for _ in range(L)]
        rows = [f[0]]
        for j in range(1, L): rows.append(rows[-1] * bases[j] + f[j])
        x = rng.integers(0, C, n + 1)
        alpha = rng.choice([0.6, 2.0, 5.0], L + 1)
        tr = CtxTree([r[:n] for r in rows], bases, x[:n], C); tr.tables(alpha, exact=True)
        got = tr.predict([r[n:] for r in rows], x[n:])[0]
        r1, r2 = rhos(alpha, "dirichlet")
        sig = [r1[0]]; tau = [r2[0]]
        for d in range(1, L + 1): tau.append(r2[d] * sig[-1]); sig.append(r1[d] * sig[-1])
        def shared(a, b):                                             # deepest common level (0 = root only)
            k = 0
            for j in range(L):
                if rows[j][a] != rows[j][b]: break
                k = j + 1
            return k
        cov = lambda a, b: sum(tau[: shared(a, b) + 1])
        K = np.array([[cov(a, b) + (sig[L] if a == b else 0.0) for b in range(n)] for a in range(n)])
        k = np.array([cov(n, b) for b in range(n)])
        y = (x[:n] == x[n]).astype(float); u = 1.0 / C
        want = u + k @ np.linalg.solve(K, y - u)
        print(f"ctx trial {trial}: L={L}, n={n}, recursion {got:.12f}, explicit {want:.12f}")
        assert abs(got - want) < 1e-9
    print("ALL OK")



# ------------------------------------------------------------------------------------------------ experiments
GRID = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def row_levels(d, prev, tp, um, levels=3):
    C, T = len(d["symbols"]), len(d["contexts"])
    r1 = np.minimum(d["dp"][um], 3).astype(np.int64) * T + d["t"][um].astype(np.int64)
    rows = [r1, r1 * (C + 1) + prev[um]]
    if levels == 3: rows.append(rows[1] * (T + 1) + tp[um])
    return rows, [1, C + 1, T + 1][:levels]


def fit_ctx(tree, vrows, vx, exact, start=None, sweeps=2):
    """couplings of the context tree by coordinate descent on validation units (alpha[0] = 1 fixed)"""
    a = np.array(start if start is not None else [1.0] + [4.0] * tree.L)
    def loss(al):
        tree.tables(al, exact); return float(-np.log(tree.predict(vrows, vx)).mean())
    best = loss(a)
    for _ in range(sweeps):
        changed = False
        for j in range(1, tree.L + 1):
            for g in GRID:
                if g == a[j]: continue
                b = a.copy(); b[j] = g; v = loss(b)
                if v < best - 1e-6: a, best, changed = b, v, True
        if not changed: break
    tree.tables(a, exact)
    return a, best


def run_m1():
    d = astdata.load()
    test_f, pool = astdata.split(d)
    val = np.random.default_rng(7).choice(pool, 20, replace=False); pool = np.setdiff1d(pool, val)
    prev, tp = features(d); C = len(d["symbols"])
    tr, va, te = (np.isin(d["ufile"], f) for f in (pool, val, test_f))
    x = d["x"].astype(np.int64)
    for levels in (2, 3):
        rtr, bases = row_levels(d, prev, tp, tr, levels); rva, _ = row_levels(d, prev, tp, va, levels)
        rte, _ = row_levels(d, prev, tp, te, levels)
        tree = CtxTree(rtr, bases, x[tr], C)
        out = {}
        for exact in (True, False):
            a, v = fit_ctx(tree, rva, x[va], exact)
            out["exact" if exact else "interpolated"] = dict(alpha=[float(z) for z in a], val=round(v, 5),
                test=round(float(-np.log(tree.predict(rte, x[te])).mean()), 5))
        log(data="m1", corpus=astdata.CORPUS, levels=levels, rows_finest=int(len(tree.kr[-1])), **out)



from code_cred import cred_predict
SGRID = (1, 2, 4, 8, 16, 32, 64, 128, np.inf)


class ScopeFiles:
    """files prepared for the scope recursion on arbitrary rows: rows (n,) global codes, r (n,) shared-rule values"""
    def __init__(self, d, files, rows, um):
        ufile = d["ufile"][um]; self.items = []; self.n = int(um.sum()); self.C = len(d["symbols"])
        gidx = np.nonzero(um)[0]
        for f in files:
            a, b = np.searchsorted(ufile, [f, f + 1])
            if b == a: continue
            u0 = gidx[a]; n0, n1 = np.searchsorted(d["node_file"], [f, f + 1])
            loc_rows, rl = np.unique(rows[a:b], return_inverse=True)
            node = d["node"][u0:u0 + b - a].astype(np.int64)
            self.items.append(dict(sl=slice(a, b), x=d["x"][u0:u0 + b - a].astype(np.int64), row=rl, nrow=len(loc_rows),
                                   par=d["par"][u0:u0 + b - a].astype(np.int64) - n0, node=np.where(node >= 0, node - n0, -1),
                                   ndep=d["node_depth"][n0:n1].astype(np.int64)))

    def nll(self, r, alpha):
        tot = 0.0
        for it in self.items:
            p = cred_predict(it["x"], it["row"], it["par"], it["node"], r[it["sl"]], alpha, nrow=it["nrow"], nsym=self.C,
                             ndep=it["ndep"], by="depth")
            tot += float(-np.log(p).sum())
        return tot / self.n


def fit_scope(F, r, start=(16, 16, 32, np.inf), sweeps=2):
    a = np.array(start, float); best = F.nll(r, a)
    for _ in range(sweeps):
        changed = False
        for j in range(len(a)):
            for g in SGRID:
                if g == a[j]: continue
                b = a.copy(); b[j] = g; v = F.nll(r, b)
                if v < best - 1e-6: a, best, changed = b, v, True
        if not changed: break
    return a, best


def run_curve():
    from code_gap import h1_rule
    d = astdata.load()
    test_f, pool = astdata.split(d)
    val = np.random.default_rng(7).choice(pool, 20, replace=False); pool = np.setdiff1d(pool, val)
    prev, tp = features(d); C, T = len(d["symbols"]), len(d["contexts"])
    va, te = np.isin(d["ufile"], val), np.isin(d["ufile"], test_f)
    x = d["x"].astype(np.int64)
    rva, bases = row_levels(d, prev, tp, va); rte, _ = row_levels(d, prev, tp, te)
    Vc, Tc = ScopeFiles(d, val, rva[0], va), ScopeFiles(d, test_f, rte[0], te)        # coarse rows (histories alone)
    Vf, Tf = ScopeFiles(d, val, rva[-1], va), ScopeFiles(d, test_f, rte[-1], te)      # finest rows (combined)
    fmt = lambda a: [("inf" if np.isinf(z) else float(z)) for z in a]
    for n in (len(pool), 256, 64, 16, 4, 1):
        for s in range(1 if n == len(pool) else 2):
            files = np.random.default_rng(100 * n + s).choice(pool, n, replace=False)   # the files of the Colab runs
            tr = np.isin(d["ufile"], files)
            plain = float(-np.log(h1_rule(d, files)[rte[0], x[te]]).mean())
            # histories alone: shared H = 1 rule on coarse rows, couplings per depth
            R = h1_rule(d, files)
            rv, rt = R[rva[0], x[va]], R[rte[0], x[te]]
            ah, _ = fit_scope(Vc, rv)
            hist = Tc.nll(rt, ah)
            # context tree alone, then histories on the finest rows
            rtr, _ = row_levels(d, prev, tp, tr)
            tree = CtxTree(rtr, bases, x[tr], C)
            ac, _ = fit_ctx(tree, rva, x[va], True)
            pv, pt = tree.predict(rva, x[va]), tree.predict(rte, x[te])
            m1 = float(-np.log(pt).mean())
            ab, _ = fit_scope(Vf, pv)
            both = Tf.nll(pt, ab)
            log(data="curve", corpus=astdata.CORPUS, files=n, seed=s, train_units=int(tr.sum()), plain=round(plain, 5), hist=round(hist, 5), alpha_hist=fmt(ah),
                ctx=round(m1, 5), alpha_ctx=fmt(ac), both=round(both, 5), alpha_both=fmt(ab))


if __name__ == "__main__":
    if "mi" in sys.argv: run_mi()
    if "test" in sys.argv: test_ctx()
    if "m1" in sys.argv: run_m1()
    if "curve" in sys.argv: run_curve()
