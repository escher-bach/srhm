"""Claim 1 on programs, observed-rule level (where.tex sec. prog-claim1).

The AST is encoded as a full top-down generative process: every node's children are drawn field by field from
R(c | t), t = (parent type, field); list fields end with a STOP symbol and empty optional fields emit NONE, so the
units below are a lossless encoding of the tree (up to the symbol vocabulary) and the sum of their log-probabilities
is the log-likelihood of the tree.

Models (all smoothed the same way, R = (n + a m)/(n_t + a), m = training marginal of c, a = 1):
  tied     one rule for all depths
  untied   one rule per depth band
  partial  own rules for bands 0-2 (module / class level), one tied rule for bands 3+
Usage: python code_claim1.py [count|curve]
"""
import ast, glob, json, os, sys, time
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "code_claim1_results.jsonl")
BANDS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 8), (9, 99)]
TOP = 3                     # bands < TOP are untied in the partial model
A = 1.0
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


def units_of(tree):
    """(depth of parent, context, child symbol) for every child slot, STOP after lists, NONE for empty options"""
    out = []
    st = [(tree, 0)]
    while st:
        nd, k = st.pop()
        for fld, val in ast.iter_fields(nd):
            ctx = type(nd).__name__ + "." + fld
            if isinstance(val, list):
                if val and not isinstance(val[0], ast.AST):
                    continue                              # lists of identifier strings (e.g. Global.names)
                for c in val:
                    if isinstance(c, ast.AST) and not isinstance(c, ast.expr_context):
                        out.append((k, ctx, child_sym(c))); st.append((c, k + 1))
                out.append((k, ctx, "<STOP>"))
            elif isinstance(val, ast.AST):
                if not isinstance(val, ast.expr_context):
                    out.append((k, ctx, child_sym(val))); st.append((val, k + 1))
            elif val is None and fld in OPTIONAL.get(type(nd).__name__, ()):
                out.append((k, ctx, "<NONE>"))
    return out


def optional_fields():
    opt = {}
    for name in dir(ast):
        cls = getattr(ast, name)
        if isinstance(cls, type) and issubclass(cls, ast.AST):
            fs = [f for f, ty in getattr(cls, "_field_types", {}).items() if str(ty).startswith("ast.") and "None" in str(ty)]
            if fs: opt[name] = tuple(fs)
    return opt


OPTIONAL = optional_fields()


def load():
    lib = os.path.dirname(os.__file__)
    files = sorted(f for f in glob.glob(os.path.join(lib, "**", "*.py"), recursive=True) if "site-packages" not in f)
    sys.setrecursionlimit(20000)
    rows, fid = [], []
    for fi, f in enumerate(files):
        try:
            with open(f, "rb") as fh:
                tr = ast.parse(fh.read())
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        u = units_of(tr); rows += u; fid += [fi] * len(u)
    cnt = Counter(r[2] for r in rows)
    keep = sorted(w for w, n in cnt.items() if n >= 200)
    cv = {w: i for i, w in enumerate(keep)}; C = len(cv) + 1
    tv = {w: i for i, w in enumerate(sorted({r[1] for r in rows}))}
    band = np.full(200, len(BANDS) - 1)
    for b, (lo, hi) in enumerate(BANDS):
        band[lo:min(hi, 199) + 1] = b
    f = np.array(fid); k = band[np.array([r[0] for r in rows])]
    t = np.array([tv[r[1]] for r in rows])
    c = np.array([cv.get(r[2], C - 1) for r in rows])
    return f, k, t, c, len(tv), C


def counts(t, c, T, C):
    return np.bincount(t * C + c, minlength=T * C).reshape(T, C).astype(float)


def model_groups(kind):
    nb = len(BANDS)
    if kind == "tied": return [list(range(nb))]
    if kind == "untied": return [[b] for b in range(nb)]
    if kind == "partial": return [[b] for b in range(TOP)] + [list(range(TOP, nb))]
    raise ValueError(kind)


def fit_eval(tr, te, k, t, c, T, C, kind):
    m = np.bincount(c[tr], minlength=C) + 1.0; m /= m.sum()
    ll = 0.0
    for g in model_groups(kind):
        itr, ite = tr & np.isin(k, g), te & np.isin(k, g)
        n = counts(t[itr], c[itr], T, C)
        R = (n + A * m) / (n.sum(1, keepdims=True) + A)
        ll += np.log(R[t[ite], c[ite]]).sum()
    return -ll / te.sum()


def n_params(k, t, c, T, C, kind, mask):
    """free parameters of the MLE: sum over (group, context) of (support size - 1)"""
    tot = 0
    for g in model_groups(kind):
        sel = mask & np.isin(k, g)
        n = counts(t[sel], c[sel], T, C)
        tot += int(np.maximum((n > 0).sum(1) - 1, 0).sum())
    return tot


def main(what):
    rng = np.random.default_rng(0)
    f, k, t, c, T, C = load()
    files = np.unique(f)
    perm = rng.permutation(files)
    test_files, pool = perm[: len(files) // 2], perm[len(files) // 2:]
    te = np.isin(f, test_files)
    if what == "count":
        trall = np.isin(f, pool)
        per_file = np.bincount(f)[pool]
        log(data="counts", units=int(len(t)), contexts=T, symbols=C, test_units=int(te.sum()),
            train_units_all=int(trall.sum()), units_per_file_median=float(np.median(per_file)),
            units_per_file_mean=float(per_file.mean()),
            k={kd: n_params(k, t, c, T, C, kd, trall) for kd in ("tied", "untied", "partial")},
            share_per_band=[round(float((k[trall] == b).mean()), 4) for b in range(len(BANDS))],
            asymptote={kd: round(fit_eval(trall, te, k, t, c, T, C, kd), 5) for kd in ("tied", "untied", "partial")})
    if what == "curve":
        for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, len(pool)):
            for s in range(3 if n < len(pool) else 1):
                sub = rng.choice(pool, n, replace=False)
                tr = np.isin(f, sub)
                log(data="curve", files=n, seed=s, train_units=int(tr.sum()),
                    nats={kd: round(fit_eval(tr, te, k, t, c, T, C, kd), 5) for kd in ("tied", "untied", "partial")})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "count")
