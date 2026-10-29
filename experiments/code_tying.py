"""F1 for programs (where.tex sec. programs): is the rule 'parent (type, field) -> child symbol' shared across depth?

Units: every (parent depth band k, context t = (parent type, field), child symbol c) in the ASTs of the local Python
standard library.  The transfer statistic (the stage1 sigma with log-likelihood in place of regression signal):

    sigma(k -> k') = [LL_k'(R_k) - LL_k'(B_k')] / [LL_k'(R_k') - LL_k'(B_k')]

R_k: rule fitted on band k (train files), B_k': the context-free marginal of c in band k' (train files); both
bands fitted on the same number of units; evaluated on held-out files of band k', on contexts seen >= 5 times in
both training bands.  Calibration: the same units with c resampled from rules of known tying rho.
Usage: python code_tying.py [calib|real|all]
"""
import ast, glob, json, os, sys, time
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "code_tying_results.jsonl")
BANDS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 8), (9, 99)]
ALPHA, MIN_CTX, SPLITS = 1.0, 5, 5
T0 = time.time()


def log(**k):
    k["t"] = round(time.time() - T0, 1)
    print(json.dumps(k), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(k) + "\n")


def child_sym(n):
    if isinstance(n, ast.Name): return "N:" + n.id
    if isinstance(n, ast.Constant): return "C:" + type(n.value).__name__
    return type(n).__name__


def load_units():
    lib = os.path.dirname(os.__file__)
    files = sorted(f for f in glob.glob(os.path.join(lib, "**", "*.py"), recursive=True) if "site-packages" not in f)
    sys.setrecursionlimit(20000)
    rows = []                                            # (file, depth, ctx string, child symbol)
    for fi, f in enumerate(files):
        try:
            with open(f, "rb") as fh:
                tr = ast.parse(fh.read())
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        st = [(tr, 0)]
        while st:
            nd, k = st.pop()
            for fld, val in ast.iter_fields(nd):
                for c in (val if isinstance(val, list) else [val]):
                    if isinstance(c, ast.AST) and not isinstance(c, ast.expr_context):
                        rows.append((fi, k, type(nd).__name__ + "." + fld, child_sym(c)))
                        st.append((c, k + 1))
    cnt = Counter(r[3] for r in rows)
    keep = {w for w, n in cnt.items() if n >= 200}       # child vocabulary: symbols seen >= 200 times, else OTHER
    cv = {w: i for i, w in enumerate(sorted(keep))}; C = len(cv) + 1
    tv = {w: i for i, w in enumerate(sorted({r[2] for r in rows}))}
    band = np.full(max(r[1] for r in rows) + 1, -1)
    for b, (lo, hi) in enumerate(BANDS):
        band[lo:min(hi, len(band) - 1) + 1] = b
    f = np.array([r[0] for r in rows]); k = band[np.array([r[1] for r in rows])]
    t = np.array([tv[r[2]] for r in rows]); c = np.array([cv.get(r[3], C - 1) for r in rows])
    return f, k, t, c, len(tv), C, len(files)


def rule(t, c, T, C):
    n = np.bincount(t * C + c, minlength=T * C).reshape(T, C).astype(float)
    marg = (n.sum(0) + 1) / (n.sum() + C)
    R = (n + ALPHA * marg) / (n.sum(1, keepdims=True) + ALPHA)
    return R, marg, n.sum(1)


def band_stats(f, k, t, c, T, C, rng):
    """per band b, on held-out files: tying bias beta_b = 1 - gain(pooled rule)/gain(own rule), and leave-band-out
    transfer sigma_b = gain(rule pooled over the OTHER bands)/gain(own rule); gains are held-out log-likelihood
    per unit over the band's own marginal, on contexts seen >= MIN_CTX times in both training sets."""
    nb = len(BANDS); B = np.full((SPLITS, nb), np.nan); S = np.full((SPLITS, nb), np.nan)
    G = np.full((SPLITS, nb), np.nan)
    files = np.unique(f)
    for s in range(SPLITS):
        tr = np.isin(f, rng.permutation(files)[: len(files) // 2])
        Rp, _, _ = rule(t[tr], c[tr], T, C)
        for b in range(nb):
            ib, io = tr & (k == b), tr & (k != b)
            if ib.sum() < 2000: continue
            Rb, mb, nbc = rule(t[ib], c[ib], T, C)
            Ro, _, no = rule(t[io], c[io], T, C)
            te = np.where(~tr & (k == b))[0]
            te = te[(nbc[t[te]] >= MIN_CTX) & (no[t[te]] >= MIN_CTX)]
            if len(te) < 1000: continue
            base = np.log(mb[c[te]]).mean()
            own = np.log(Rb[t[te], c[te]]).mean() - base
            B[s, b] = 1 - (np.log(Rp[t[te], c[te]]).mean() - base) / own
            S[s, b] = (np.log(Ro[t[te], c[te]]).mean() - base) / own
            G[s, b] = own
    return B, S, G


def summarize(name, B, S, G, extra=None):
    lab = [f"{lo}" if lo == hi else f"{lo}-{hi if hi < 99 else '+'}" for lo, hi in BANDS]
    r = lambda A: [None if np.isnan(x) else round(float(x), 3) for x in A]
    log(data=name, bands=lab, beta=r(np.nanmean(B, 0)), beta_sd=r(np.nanstd(B, 0)),
        sigma_lbo=r(np.nanmean(S, 0)), sigma_sd=r(np.nanstd(S, 0)), gain_own_nats=r(np.nanmean(G, 0)), **(extra or {}))


def synthetic(k, t, T, C, pooled, rho, rng):
    """child symbols resampled: each (band, context) keeps the pooled rule w.p. rho, else a fresh Dirichlet rule on
    the same support"""
    nb = len(BANDS); c = np.empty_like(t)
    for b in range(nb):
        for ctx in np.unique(t[k == b]):
            p = pooled[ctx]
            if rng.random() >= rho:
                sup = p > 0
                q = np.zeros(C); q[sup] = rng.dirichlet(np.ones(sup.sum())); p = q
            idx = np.where((k == b) & (t == ctx))[0]
            c[idx] = rng.choice(C, size=len(idx), p=p)
    return c


def main(what):
    rng = np.random.default_rng(0)
    f, k, t, c, T, C, nf = load_units()
    log(data="units", n=int(len(t)), files=nf, contexts=T, child_symbols=C,
        per_band=[int((k == b).sum()) for b in range(len(BANDS))])
    if what in ("calib", "all"):
        n = np.bincount(t * C + c, minlength=T * C).reshape(T, C).astype(float)
        pooled = n / np.maximum(n.sum(1, keepdims=True), 1)
        for rho in (1.0, 0.8, 0.5, 0.0):
            cs = synthetic(k, t, T, C, pooled, rho, rng)
            summarize(f"calib_rho{rho}", *band_stats(f, k, t, cs, T, C, rng))
    if what in ("real", "all"):
        summarize("python_stdlib", *band_stats(f, k, t, c, T, C, rng))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
