"""Exact local inference in a mean-preserving tied tree (hist.tex, Theorem 2), applied prequentially to programs.

Every scope a has a mean parameter mu_a with E[mu_child | mu_parent] = mu_parent; a unit of a is a draw with mean mu_a.
Second moments are tied: a child's within-scope variance is rho1 times its parent's, its between-scope variance rho2 times.
  Dirichlet tree with concentration alpha: rho1 = alpha/(alpha+1), rho2 = 1/(alpha+1);  Gaussian tree: rho1 = 1, rho2 = 1/alpha.
Upward (state of a completed scope, per rule row): N_a = #direct units + sum_children U_d,  ybar_a = weighted mean,
    U_d = 1 / (rho1/N_d + rho2)        (what scope d is worth to its parent: it saturates)
Downward (prior of a scope given everything outside it):  count a_c = rho1 / (rho2 + 1/c_parent),  mean m_parent,
    m_a = (a_c m_parent + S_a)/(a_c + N_a),  c_a = a_c + N_a,   root: a_c = alpha, mean R.
The predictive of a unit is m of its parent, using only earlier units (prequential), so the product over a file is a
valid likelihood.  The interpolated rule of code_nested.py is this with U_d = N_d and with the parent's own estimate
(which already contains the child's data) used as the child's prior.
Usage: python code_cred.py [test|curve]
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "code_cred_results.jsonl")
T0 = time.time()
NROW, NSYM = 580, 613


def log(**k):
    k["t"] = round(time.time() - T0, 1); print(json.dumps(k), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(k) + "\n")


def rhos(alpha, kind):
    """alpha: a number or an array; alpha = inf means no coupling of its own (rho1 = 1, rho2 = 0)"""
    alpha = np.asarray(alpha, np.float64); inf = np.isinf(alpha); a = np.where(inf, 1.0, alpha)
    if kind == "dirichlet":
        return np.where(inf, 1.0, a / (a + 1)), np.where(inf, 0.0, 1 / (a + 1))
    return np.ones_like(a), np.where(inf, 0.0, 1 / a)


def alpha_table(alpha, nrow, by):
    """(D, nrow) table of couplings: entry [d, r] couples a scope of depth d (clipped to D-1) to its parent, row r"""
    alpha = np.asarray(alpha, np.float64)
    if alpha.ndim == 0: return np.full((1, nrow), float(alpha))
    if alpha.ndim == 2: return alpha
    return np.repeat(alpha[:, None], nrow, 1) if by == "depth" else alpha[None, :].copy()


def prefix_sum_index(code, pos2, val, M):
    o = np.lexsort((pos2, code)); comp = code[o] * M + pos2[o]
    cs = np.concatenate([[0.0], np.cumsum(val[o])])
    def query(qcode, cutoff2):                         # sum of val over records with this code and pos2 < cutoff2
        return cs[np.searchsorted(comp, qcode * M + cutoff2)] - cs[np.searchsorted(comp, qcode * M)]
    return query


def cred_predict(x, row, par, node, r, alpha, kind="dirichlet", nrow=NROW, nsym=NSYM, ndep=None, by="row"):
    """x, row (n,): symbols and rule rows in pre-order; par (n,): local id of the parent node (root = 0);
    node (n,): local id of the child node or -1; r (n,): shared rule R[row, x].  Returns p (n,)."""
    n = len(x)
    alpha = alpha_table(alpha, nrow, by); D = len(alpha); rho1, rho2 = rhos(alpha, kind)
    nn = int(max(par.max(), node.max())) + 1
    cpos = np.full(nn, -1, np.int64); cpos[node[node >= 0]] = np.nonzero(node >= 0)[0]
    npar = np.zeros(nn, np.int64); npar[node[node >= 0]] = par[node >= 0]           # parent node of each node
    if ndep is None:
        ndep = np.zeros(nn, np.int64)
        for u in np.nonzero(node >= 0)[0]:                                           # pre-order: parents first
            ndep[node[u]] = ndep[par[u]] + 1
    # ---- upward: records (scope, row, x, pos2, val); direct units at pos2 = 2u, child summaries at 2 cpos + 1
    A, Rw, X, P2, V = [par], [row], [x], [2 * np.arange(n)], [np.ones(n)]
    pend = {}                                                                        # depth -> summary records
    cur_a, cur_r, cur_x, cur_v = par, row, x, np.ones(n)
    dep_of_rec = ndep[par]
    for d in range(int(ndep.max()), 0, -1):
        m = dep_of_rec == d
        a_, r_, x_, v_ = cur_a[m], cur_r[m], cur_x[m], cur_v[m]
        if d in pend:
            pa, pr, px, pv = pend.pop(d)
            a_, r_, x_, v_ = np.r_[a_, pa], np.r_[r_, pr], np.r_[x_, px], np.r_[v_, pv]
        if len(a_) == 0: continue
        key = (a_ * nrow + r_) * nsym + x_
        uk, inv = np.unique(key, return_inverse=True)
        S = np.bincount(inv, weights=v_)
        ar = uk // nsym                                                              # (scope, row) of each cell
        ua, inva = np.unique(ar, return_inverse=True)
        N = np.bincount(inva, weights=S)
        dsc = np.minimum(ndep[ua // nrow], D - 1)
        U = 1.0 / (rho1[dsc, ua % nrow] / N + rho2[dsc, ua % nrow])
        sc = ua[inva] // nrow; rw = ua[inva] % nrow; xs = uk % nsym
        val = U[inva] * S / N[inva]
        p_ = npar[sc]
        A.append(p_); Rw.append(rw); X.append(xs); P2.append(2 * cpos[sc] + 1); V.append(val)
        dd = ndep[p_]
        for d2 in np.unique(dd):
            mm = dd == d2
            rec = (p_[mm], rw[mm], xs[mm], val[mm])
            pend[int(d2)] = rec if d2 not in pend else tuple(np.r_[u, v] for u, v in zip(pend[int(d2)], rec))
    A, Rw, X, P2, V = map(np.concatenate, (A, Rw, X, P2, V))
    M = 2 * n + 2
    qS = prefix_sum_index((A * nrow + Rw) * nsym + X, P2, V, M)
    qN = prefix_sum_index(A * nrow + Rw, P2, V, M)
    # ---- downward along each unit's ancestor path, outermost scope first
    cp = cpos[par]                                                                   # creator of the parent (-1: root)
    starts, anc = [cp], [par]
    while (starts[-1] >= 0).any():
        s = starts[-1]; ok = s >= 0
        anc.append(np.where(ok, par[np.maximum(s, 0)], -1))
        starts.append(np.where(ok, cp[np.maximum(s, 0)], -2))
    p = r.astype(np.float64).copy(); c = np.zeros(n)
    idx_all = np.arange(n)
    for k in range(len(anc) - 1, -1, -1):
        ex = anc[k] >= 0
        if not ex.any(): continue
        i = idx_all[ex]; a = anc[k][i]
        cut = 2 * i if k == 0 else 2 * starts[k - 1][i] + 1
        Nq = qN(a * nrow + row[i], cut); Sq = qS((a * nrow + row[i]) * nsym + x[i], cut)
        root = starts[k][i] == -1
        ri = row[i]; da = np.minimum(ndep[a], D - 1)
        a0 = np.where(root, np.minimum(alpha[0, ri], 1e12), rho1[da, ri] / (rho2[da, ri] + 1.0 / np.maximum(c[i], 1e-300)))
        p[i] = (a0 * p[i] + Sq) / (a0 + Nq); c[i] = a0 + Nq
    return p


# ------------------------------------------------------------------------------------------------ brute-force check
def random_tree(rng, max_depth=4):
    par, node, x, row = [], [], [], []; nid = [0]
    def grow(p, depth):
        for _ in range(rng.integers(1, 5)):
            x.append(int(rng.integers(3))); row.append(int(rng.integers(2))); par.append(p)
            if rng.random() < 0.3:
                node.append(-1); continue
            nid[0] += 1; cc = nid[0]; node.append(cc)
            if depth < max_depth and rng.random() < 0.65: grow(cc, depth + 1)
    grow(0, 0)
    return tuple(map(np.array, (x, row, par, node)))


def brute_blp(x, row, par, node, R, alpha, kind):
    """best linear predictor of 1[x_i = x] from the earlier same-row units, by explicit covariance matrices"""
    n = len(x); al = np.atleast_1d(np.asarray(alpha, np.float64)); rho1, rho2 = rhos(al, kind)
    parent_of = {int(node[u]): int(par[u]) for u in range(n) if node[u] >= 0}
    sig2, tau2, dep = {0: rho1[0]}, {0: rho2[0]}, {0: 0}
    def chain(a):
        ch = [a]
        while ch[-1] != 0: ch.append(parent_of[ch[-1]])
        return ch[::-1]                                                              # root first
    for u in range(n):                                                               # pre-order: parents first
        if node[u] >= 0:
            c_, p_ = int(node[u]), int(par[u]); dep[c_] = dep[p_] + 1; dd = min(dep[c_], len(al) - 1)
            sig2[c_] = rho1[dd] * sig2[p_]; tau2[c_] = rho2[dd] * sig2[p_]
    out = np.empty(n)
    for i in range(n):
        J = [j for j in range(i) if row[j] == row[i]]
        ci = chain(int(par[i])); r = R[row[i], x[i]]
        if not J:
            out[i] = r; continue
        def cov(a_chain, b_chain):
            s = 0.0
            for u, v in zip(a_chain, b_chain):
                if u != v: break
                s += tau2[u]
            return s
        chains = [chain(int(par[j])) for j in J]
        K = np.array([[cov(chains[a], chains[b]) + (sig2[int(par[J[a]])] if a == b else 0.0) for b in range(len(J))]
                      for a in range(len(J))])
        k = np.array([cov(ci, ch) for ch in chains])
        y = (x[J] == x[i]).astype(float)
        out[i] = r + k @ np.linalg.solve(K, y - r)
    return out


def test():
    rng = np.random.default_rng(0)
    for kind in ("dirichlet", "gauss"):
        for trial in range(6):
            x, row, par, node = random_tree(rng)
            alpha = float(rng.choice([0.5, 2.0, 7.0]))
            R = rng.dirichlet(np.ones(3), size=2)
            fast = cred_predict(x, row, par, node, R[row, x], alpha, kind, nrow=2, nsym=3)
            slow = brute_blp(x, row, par, node, R, alpha, kind)
            err = np.abs(fast - slow).max()
            print(f"{kind} trial {trial}: units {len(x)}, alpha {alpha}, max |recursion - explicit BLP| = {err:.2e}")
            assert err < 1e-9
    for trial in range(6):                                       # couplings that depend on depth, some infinite
        x, row, par, node = random_tree(rng)
        al = np.array([3.0, 0.8, 5.0, np.inf]) if trial % 2 else np.array([2.0, np.inf, 1.5])
        R = rng.dirichlet(np.ones(3), size=2)
        fast = cred_predict(x, row, par, node, R[row, x], al, "dirichlet", nrow=2, nsym=3, by="depth")
        slow = brute_blp(x, row, par, node, R, al, "dirichlet")
        err = np.abs(fast - slow).max()
        print(f"per-depth trial {trial}: units {len(x)}, alpha {al}, max err = {err:.2e}")
        assert err < 1e-9
    print("ALL OK")


# ------------------------------------------------------------------------------------------------ programs
class Files:
    """per-file local arrays, sliced once (units and nodes of a file are contiguous)"""
    def __init__(self, d, files):
        T = len(d["contexts"]); self.items = []
        for f in files:
            u0, u1 = np.searchsorted(d["ufile"], [f, f + 1]); n0, n1 = np.searchsorted(d["node_file"], [f, f + 1])
            x = d["x"][u0:u1].astype(np.int64); t = d["t"][u0:u1].astype(np.int64); dp = d["dp"][u0:u1].astype(np.int64)
            row = np.minimum(dp, 3) * T + t
            par = d["par"][u0:u1].astype(np.int64) - n0
            node = d["node"][u0:u1].astype(np.int64); node = np.where(node >= 0, node - n0, -1)
            self.items.append((x, row, par, node, d["node_depth"][n0:n1].astype(np.int64)))
        self.n = sum(len(it[0]) for it in self.items)

    def nll(self, R, alpha, kind="dirichlet", per_file=False, by="row"):
        out = [float(-np.log(cred_predict(x, row, par, node, R[row, x], alpha, kind, ndep=nd, by=by)).sum())
               for x, row, par, node, nd in self.items]
        return np.array(out) if per_file else sum(out) / self.n


def curve():
    import astdata
    from code_gap import h1_rule
    d = astdata.load()
    test_f, pool = astdata.split(d)
    val = np.random.default_rng(7).choice(pool, 20, replace=False); pool = np.setdiff1d(pool, val)
    V, Te = Files(d, val), Files(d, test_f)
    grid = (2, 4, 8, 16, 32, 64, 128, 256)
    for n in (len(pool), 1, 4, 16, 64, 256):
        for s in range(1 if n == len(pool) else 2):
            files = np.random.default_rng(100 * n + s).choice(pool, n, replace=False)   # the files of the Colab runs
            R = h1_rule(d, files)
            v = {a: V.nll(R, a) for a in grid}
            a = min(v, key=v.get)
            log(data="cred", files=n, seed=s, alpha=a, val={k: round(x, 5) for k, x in v.items()}, test=round(Te.nll(R, a), 5))
            if n == len(pool):                                                           # (E2) alpha from one file
                per = {a2: V.nll(R, a2, per_file=True) for a2 in grid}
                sizes = np.array([len(it[0]) for it in V.items])
                res = []
                for i in range(len(V.items)):
                    ai = min(grid, key=lambda a2: per[a2][i])
                    others = np.arange(len(V.items)) != i
                    loss = lambda a2: per[a2][others].sum() / sizes[others].sum()
                    aall = min(grid, key=loss)
                    res.append(dict(units=int(sizes[i]), alpha_one=ai, alpha_rest=aall, regret=round(loss(ai) - loss(aall), 5)))
                log(data="cred_E2", per_file=res, within=int(sum(r["regret"] <= 0.005 for r in res)))


if __name__ == "__main__":
    test() if "test" in sys.argv else curve()
