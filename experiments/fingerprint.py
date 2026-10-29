"""F2 of where.tex: does a dataset's dependence follow tree distance or position distance?

  distance test U (Def. U): mutual information I_{d,l} of leaf pairs tabulated by position distance d and
      tree distance l; U = what l adds beyond d / (that + what d adds beyond l), fitted on log I.
  boundary profile b(h) (Prop. boundary, Cor. perimeter): Gaussian-proxy MI between the w=2 strips on either
      side of the shared edge of sibling blocks, per height h, minus a shuffled-pairs baseline.

Data: calibration (tied binary-tree broadcast; stationary Markov chain), Kodak (quadtree on pixels), the
local Python standard library (abstract syntax trees), and for b(h) the controls 1/f^2 noise and a tied
quadtree cascade.  Usage: python fingerprint.py [calib|kodak|ast|boundary|all]
"""
import ast, glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "fingerprint_results.jsonl")
DS = (1, 2, 4, 8, 16)
MIN_PAIRS = 5000
T0 = time.time()


def log(**k):
    k["t"] = round(time.time() - T0, 1)
    print(json.dumps(k), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(k) + "\n")


# ---------------------------------------------------------------- mutual information
def plugin_mi(a, b, V):
    c = np.bincount(a * V + b, minlength=V * V).reshape(V, V).astype(float)
    n = c.sum(); p = c / n
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum())


def mi(a, b, V, rng):
    """plug-in MI minus the plug-in MI of the same pairs with b shuffled (bias baseline), nats"""
    return plugin_mi(a, b, V) - plugin_mi(a, rng.permutation(b), V)


def table(pairs, V, rng):
    """pairs: dict (d, l) -> (a, b) arrays.  Returns list of cells with >= MIN_PAIRS pairs."""
    cells = []
    for (d, l), (a, b) in sorted(pairs.items()):
        if len(a) >= MIN_PAIRS:
            cells.append(dict(d=d, l=l, n=len(a), I=mi(a, b, V, rng)))
    return cells


def collect(pairs, d, l_arr, a, b):
    for l in np.unique(l_arr):
        m = l_arr == l
        key = (d, int(l))
        if key in pairs:
            pairs[key] = (np.concatenate([pairs[key][0], a[m]]), np.concatenate([pairs[key][1], b[m]]))
        else:
            pairs[key] = (a[m], b[m])


I_MIN = 0.01    # nats; cells below this are estimator noise on the log scale (calibration, where.tex sec. test)


def U_stat(cells, floor=1e-4):
    cells = [c for c in cells if c["I"] >= I_MIN]
    d = np.array([c["d"] for c in cells]); l = np.array([c["l"] for c in cells])
    y = np.log(np.maximum([c["I"] for c in cells], floor))
    def within(g):                                     # pooled within-group variance of y, groups by g
        ss, df = 0.0, 0
        for v in np.unique(g):
            yy = y[g == v]
            if len(yy) >= 2:
                ss += ((yy - yy.mean()) ** 2).sum(); df += len(yy) - 1
        return ss / df if df else float("nan")
    s_l, s_d = within(d), within(l)                    # spread across l at fixed d; across d at fixed l
    return dict(U=s_l / (s_l + s_d), spread_l_given_d=s_l, spread_d_given_l=s_d, cells=len(cells))


def d1_profile(cells):
    c1 = [c for c in cells if c["d"] == 1]
    return {c["l"]: round(c["I"], 5) for c in c1}


# ---------------------------------------------------------------- calibration
def kernel(v, eps, rng):
    P = np.eye(v)[rng.permutation(v)]
    return (1 - eps) * P + eps * rng.dirichlet(np.ones(v), size=v)


def stationary(M):
    w, vec = np.linalg.eig(M.T)
    p = np.real(vec[:, np.argmin(np.abs(w - 1))]); return p / p.sum()


def sample_rows(M, parents, rng):
    cum = M.cumsum(1)[parents]
    u = rng.random(parents.shape)[..., None]
    return np.minimum((u > cum).sum(-1), M.shape[0] - 1)


def tree_leaves_1d(M, L, T, rng):
    z = sample_rows(np.tile(stationary(M), (1, 1)), np.zeros(T, int), rng)[:, None]
    for _ in range(L):
        z = sample_rows(M, np.repeat(z, 2, axis=1), rng)
    return z                                           # (T, 2^L)


def chain_1d(M, n, T, rng):
    z = np.empty((T, n), int); z[:, 0] = sample_rows(np.tile(stationary(M), (1, 1)), np.zeros(T, int), rng)
    for i in range(1, n):
        z[:, i] = sample_rows(M, z[:, i - 1], rng)
    return z


def seq_pairs(z, V):
    """z (T, n) symbols on the leaves of a binary tree over positions 0..n-1"""
    pairs = {}
    n = z.shape[1]; x = np.arange(n)
    for d in DS:
        l = 2 * np.array([int(v).bit_length() for v in (x[:-d] ^ x[d:])])
        collect(pairs, d, np.tile(l, z.shape[0]), z[:, :-d].ravel(), z[:, d:].ravel())
    return pairs


def run_calib():
    rng = np.random.default_rng(0)
    V, L, T = 8, 12, 150
    for name, eps in (("tree", 0.35), ("tree", 0.2)):
        M = kernel(V, eps, rng)
        cells = table(seq_pairs(tree_leaves_1d(M, L, T, rng), V), V, rng)
        log(data=f"calib_{name}_eps{eps}", **U_stat(cells), d1=d1_profile(cells))
    M = kernel(V, 0.35, rng)
    cells = table(seq_pairs(chain_1d(M, 2 ** L, T, rng), V), V, rng)
    log(data="calib_markov_eps0.35", **U_stat(cells), d1=d1_profile(cells))


# ---------------------------------------------------------------- Kodak
def kodak_grey(size=512):
    from PIL import Image
    ims = []
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "kodak", "*.png"))):
        g = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64).mean(2) / 255.0
        ims.append(g[:size, :size])
    return np.stack(ims)


def run_kodak():
    rng = np.random.default_rng(1)
    g = kodak_grey()
    V = 8
    edges = np.quantile(g, np.linspace(0, 1, V + 1)[1:-1])
    q = np.digitize(g, edges)
    pairs = {}
    x = np.arange(g.shape[1])
    for d in DS:
        l = 2 * np.array([int(v).bit_length() for v in (x[:-d] ^ x[d:])])
        for arr in (q, q.transpose(0, 2, 1)):          # horizontal and vertical pairs
            a, b = arr[:, :, :-d], arr[:, :, d:]
            L_ = np.broadcast_to(l, a.shape)
            collect(pairs, d, L_.ravel(), a.ravel(), b.ravel())
    cells = table(pairs, V, rng)
    log(data="kodak_pixels_q8", **U_stat(cells), d1=d1_profile(cells))


# ---------------------------------------------------------------- Python ASTs
def ast_leaves(tree):
    """DFS over AST (expr_context nodes dropped). Returns symbols, depths, and m[j] = depth of the LCA of
    leaves j-1 and j (m[0] = -1)."""
    syms, deps, ms = [], [], []
    pending = [10 ** 9]
    def sym(n):
        t = type(n).__name__
        if isinstance(n, ast.Name): return "N:" + n.id
        if isinstance(n, ast.Constant): return "C:" + type(n.value).__name__
        return t
    def walk(n, k):
        ch = [c for c in ast.iter_child_nodes(n) if not isinstance(c, ast.expr_context)]
        if not ch:
            syms.append(sym(n)); deps.append(k)
            ms.append(pending[0] if len(syms) > 1 else -1); pending[0] = 10 ** 9
            return
        for j, c in enumerate(ch):
            if j > 0: pending[0] = min(pending[0], k)
            walk(c, k + 1)
    walk(tree, 0)
    return syms, deps, ms


def run_ast(max_files=None):
    rng = np.random.default_rng(2)
    lib = os.path.dirname(os.__file__)
    files = sorted(glob.glob(os.path.join(lib, "**", "*.py"), recursive=True))
    files = [f for f in files if "site-packages" not in f]
    if max_files: files = files[:max_files]
    sys.setrecursionlimit(20000)
    S, D, Mm = [], [], []
    nf = 0
    for f in files:
        try:
            with open(f, "rb") as fh:
                tr = ast.parse(fh.read())
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        try:
            s, d, m = ast_leaves(tr)
        except RecursionError:
            continue
        S += s; D += d; Mm += m; nf += 1
    from collections import Counter
    cnt = Counter(S)
    top = [w for w, _ in cnt.most_common(47)]
    idx = {w: i for i, w in enumerate(top)}
    V = len(top) + 1
    sym = np.array([idx.get(w, V - 1) for w in S]); dep = np.array(D); m = np.array(Mm)
    n = len(sym)
    pairs = {}
    lca = m[1:].copy()                                   # d = 1: LCA of (i, i+1) is m[i+1]
    for d in range(1, max(DS) + 1):
        if d > 1:
            lca = np.minimum(lca[:-1], m[d:])
        if d in DS:
            ok = lca >= 0                                  # same module
            a, b = sym[:n - d][ok], sym[d:][ok]
            l = (dep[:n - d] + dep[d:] - 2 * lca)[ok]
            collect(pairs, d, l, a, b)
    cells = table(pairs, V, rng)
    log(data="python_stdlib_ast", files=nf, leaves=int(n), V=V, **U_stat(cells), d1=d1_profile(cells),
        top_symbols=top[:15])


# ---------------------------------------------------------------- boundary profile b(h)
def gauss_mi(X, Y, ridge=1e-6):
    X = X - X.mean(0); Y = Y - Y.mean(0); n = len(X)
    def isqrt(C):
        C = C + ridge * np.trace(C) / len(C) * np.eye(len(C))
        w, U = np.linalg.eigh(C); return U @ np.diag(w ** -0.5) @ U.T
    Cx, Cy, Cxy = X.T @ X / n, Y.T @ Y / n, X.T @ Y / n
    r = np.linalg.svd(isqrt(Cx) @ Cxy @ isqrt(Cy), compute_uv=False)
    r = np.clip(r, 0, 1 - 1e-9)
    return float(-0.5 * np.log(1 - r ** 2).sum())


def strips(imgs, h, w=2):
    """sibling blocks of side m = 2^(h-1) inside height-h nodes; w columns (rows) either side of the shared edge"""
    m = 2 ** (h - 1); S = 2 * m; w = min(w, m)
    Xs, Ys = [], []
    for arr in (imgs, imgs.transpose(0, 2, 1)):
        N, H, W = arr.shape
        for y0 in range(0, H - S + 1, S):
            for half in (0, m):                            # top and bottom sibling pair of the node
                rows = arr[:, y0 + half:y0 + half + m, :]
                for x0 in range(0, W - S + 1, S):
                    Xs.append(rows[:, :, x0 + m - w:x0 + m].reshape(N, -1))
                    Ys.append(rows[:, :, x0 + m:x0 + m + w].reshape(N, -1))
    return np.concatenate(Xs), np.concatenate(Ys)


def b_profile(imgs, name, hs, rng):
    out = {}
    for h in hs:
        X, Y = strips(imgs, h)
        if len(X) < 20 * X.shape[1]:
            break
        out[h] = round(gauss_mi(X, Y) - gauss_mi(X, Y[rng.permutation(len(Y))]), 4)
    hh = np.array([h for h in out if h >= 2]); bb = np.array([out[h] for h in hh])
    slope = float(np.polyfit(hh, np.log2(np.maximum(bb, 1e-6)), 1)[0]) if len(hh) >= 2 else float("nan")
    log(data=f"boundary_{name}", b=out, slope_log2b_per_level=round(slope, 3))


def pink_noise(n, size, rng):
    f = np.fft.fftfreq(size); r = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2); r[0, 0] = 1
    amp = 1 / r; amp[0, 0] = 0
    out = []
    for _ in range(n):
        z = np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((size, size)))).real
        out.append((z - z.mean()) / z.std())
    return np.stack(out)


def cascade_2d(n, L, rng, v=8, eps=0.35):
    M = kernel(v, eps, rng); c = np.sort(rng.normal(size=v))
    out = []
    for _ in range(n):
        z = sample_rows(np.tile(stationary(M), (1, 1)), np.zeros((1, 1), int), rng)
        for _ in range(L):
            z = sample_rows(M, np.repeat(np.repeat(z, 2, 0), 2, 1), rng)
        out.append(c[z] + 0.1 * rng.normal(size=z.shape))
    return np.stack(out)


def run_boundary():
    rng = np.random.default_rng(3)
    hs = range(1, 8)
    b_profile(kodak_grey(), "kodak", hs, rng)
    b_profile(pink_noise(48, 512, rng), "pink_1overf2", hs, rng)
    b_profile(cascade_2d(48, 9, rng), "tied_quadtree_cascade", hs, rng)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("calib", "all"): run_calib()
    if what in ("kodak", "all"): run_kodak()
    if what in ("ast", "all"): run_ast()
    if what in ("boundary", "all"): run_boundary()
