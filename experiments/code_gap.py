"""Where can a generic sequence learner beat the tree on programs?  Two data measurements (no learner of the theory):

 (a) sibling dependence: I(x_i ; x_{i+1} | context) for adjacent children of one parent in the same field, plug-in
     minus within-context shuffle baseline.  In the observed-rule model siblings are independent given the context;
     in the latent tree whatever they share must pass through the parent's type (boundary law, <= log H nats).
 (b) within-file reuse: gain in test nats/unit of mixing the H = 1 partial rule with a cache, the empirical law of the
     previous K units of the same file (all units, or units with the same context); lambda chosen on validation files.
     This is dependence that follows position in the file, not the tree.
"""
import json, os, time
import numpy as np
import astdata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "code_gap_results.jsonl")
T0 = time.time()


def log(**k):
    k["t"] = round(time.time() - T0, 1); print(json.dumps(k), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(k) + "\n")


def sibling_mi(d, files, rng):
    um = np.isin(d["ufile"], files)
    x, t, par = d["x"][um].astype(np.int64), d["t"][um].astype(np.int64), d["par"][um].astype(np.int64)
    C = len(d["symbols"])
    # adjacent siblings: consecutive units (in pre-order) among those with the same parent and context
    order = np.lexsort((np.arange(len(x)), t, par))
    xs, ts, ps = x[order], t[order], par[order]
    adj = (ps[1:] == ps[:-1]) & (ts[1:] == ts[:-1])
    a, b, tt = xs[:-1][adj], xs[1:][adj], ts[:-1][adj]
    tot, base, n = 0.0, 0.0, len(a)
    for c in np.unique(tt):
        m = tt == c
        if m.sum() < 200: continue
        def mi(u, v):
            j = np.bincount(u * C + v, minlength=C * C).reshape(C, C) / len(u)
            pu, pv = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
            nz = j > 0
            return float((j[nz] * np.log(j[nz] / (pu @ pv)[nz])).sum())
        tot += m.sum() * mi(a[m], b[m]); base += m.sum() * mi(a[m], rng.permutation(b[m]))
    return dict(pairs=int(n), units=int(um.sum()), mi_per_pair=(tot - base) / n, mi_per_unit=(tot - base) / um.sum(),
                plugin_per_pair=tot / n, baseline_per_pair=base / n)


def h1_rule(d, files):
    um = np.isin(d["ufile"], files)
    x, t, g = d["x"][um].astype(np.int64), d["t"][um].astype(np.int64), np.minimum(d["dp"][um], 3).astype(np.int64)
    C, T = len(d["symbols"]), len(d["contexts"])
    m = np.bincount(x, minlength=C) + 1.0; m /= m.sum()
    n = np.bincount((g * T + t) * C + x, minlength=4 * T * C).reshape(4 * T, C).astype(float)
    return (n + m) / (n.sum(1, keepdims=True) + 1.0)


def cache_counts(x, key, K):
    """for every unit i: number of previous units j in [i-K, i) with key_j == key_i and x_j == x_i, and with key_j == key_i"""
    n = len(x); idx = np.arange(n)
    def prev_within(code):
        o = np.argsort(code, kind="stable"); cs, ps = code[o], idx[o]
        start = np.r_[0, np.nonzero(np.diff(cs))[0] + 1]; gid = np.cumsum(np.r_[0, np.diff(cs) != 0])
        rank = np.arange(n) - start[gid]
        # earliest position in the same group that is >= ps - K: search inside each group via global offsets
        lo = np.searchsorted(cs.astype(np.float64) * (n + 1) + ps, cs.astype(np.float64) * (n + 1) + np.maximum(ps - K, 0))
        out = np.empty(n, np.int64); out[o] = np.arange(n) - lo
        return out
    return prev_within(key * 100000 + x), prev_within(key)


def cache_eval(d, R, files, K, mode, lams):
    C, T = len(d["symbols"]), len(d["contexts"])
    tot = {l: 0.0 for l in lams}; base = 0.0; n = 0
    for f in files:
        um = d["ufile"] == f
        x, t, g = d["x"][um].astype(np.int64), d["t"][um].astype(np.int64), np.minimum(d["dp"][um], 3).astype(np.int64)
        r = R[g * T + t, x]
        key = t if mode == "context" else np.zeros_like(t)
        hit, win = cache_counts(x, key, K)
        c = np.where(win > 0, hit / np.maximum(win, 1), 0.0)
        base += -np.log(r).sum(); n += len(x)
        for l in lams:
            ll = np.where(win > 0, l, 0.0)
            tot[l] += -np.log((1 - ll) * r + ll * c).sum()
    return base / n, {l: v / n for l, v in tot.items()}


def main():
    rng = np.random.default_rng(0)
    d = astdata.load()
    test, pool = astdata.split(d)
    val = np.random.default_rng(7).choice(pool, 20, replace=False); pool = np.setdiff1d(pool, val)
    log(data="sibling_mi", **sibling_mi(d, pool, rng))
    R = h1_rule(d, pool)
    lams = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
    for mode in ("all", "context"):
        for K in (64, 512):
            _, v = cache_eval(d, R, val, K, mode, lams)
            best = min(v, key=v.get)
            b, te = cache_eval(d, R, test, K, mode, (best,))
            log(data="cache", mode=mode, K=K, lam=best, h1_test=round(b, 5), cache_test=round(te[best], 5),
                gain=round(b - te[best], 5))


if __name__ == "__main__" and "polya" not in __import__("sys").argv:
    main()


def polya_eval(d, R, files, alphas):
    """exact marginal nats/unit of the file-adapted rule: p = (alpha R + count)/(alpha + n) over earlier same-row units"""
    C, T = len(d["symbols"]), len(d["contexts"])
    tot = {a: 0.0 for a in alphas}; n = 0
    for f in files:
        um = d["ufile"] == f
        x, t, g = d["x"][um].astype(np.int64), d["t"][um].astype(np.int64), np.minimum(d["dp"][um], 3).astype(np.int64)
        row = g * T + t
        hit, win = cache_counts(x, row, len(x) + 1)
        r = R[row, x]; n += len(x)
        for a in alphas:
            tot[a] += -np.log((a * r + hit) / (a + win)).sum()
    return {a: v / n for a, v in tot.items()}


def polya_curve():
    d = astdata.load()
    test, pool = astdata.split(d)
    val = np.random.default_rng(7).choice(pool, 20, replace=False); pool = np.setdiff1d(pool, val)
    alphas = (0.5, 1, 2, 4, 8, 16, 32)
    # slice the test and validation units once (speed)
    for n in (1, 4, 16, 64, 256, len(pool)):
        for s in range(1 if n == len(pool) else 2):
            files = np.random.default_rng(100 * n + s).choice(pool, n, replace=False)   # the files of the Colab runs
            R = h1_rule(d, files)
            v = polya_eval(d, R, val, alphas); a = min(v, key=v.get)
            te = polya_eval(d, R, test, (a,))[a]
            um = np.isin(d["ufile"], test)
            x, t, g = d["x"][um].astype(np.int64), d["t"][um].astype(np.int64), np.minimum(d["dp"][um], 3).astype(np.int64)
            h1 = float(-np.log(R[g * len(d["contexts"]) + t, x]).mean())
            log(data="polya", files=n, seed=s, alpha=a, val=round(v[a], 5), h1_test=round(h1, 5), test=round(te, 5))


if __name__ == "__main__" and "polya" in __import__("sys").argv:
    polya_curve()
