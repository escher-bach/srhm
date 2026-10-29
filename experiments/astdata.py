"""Lossless top-down encoding of Python ASTs, shared by the latent tree model and the transformer baseline.

Every child slot is one UNIT, in pre-order (a unit is emitted, then the subtree of its child, then the next slot;
a list field ends with a <STOP> unit; an empty optional AST field is a <NONE> unit).  Per unit:
  ufile file id                     x     child symbol id (Name ids and Constant kinds folded in; rare -> <OTHER>)
  t     context id (parent type.field) dp    depth of the parent (root Module = 0)
  par   node index of the parent       node  node index of the child (-1 for <STOP>/<NONE>)
Per node: depth, file, internal (has units below it).  Roots (one Module per file) are nodes without a unit.
The same units as verify/code_claim1.py, so an H = 1 model reproduces its numbers exactly.
"""
import ast, glob, os, sys
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
CORPUS = os.environ.get("CORPUS", "stdlib")        # "stdlib" (the Python standard library) or a package in site-packages
CACHE = os.path.join(HERE, "astcache.npz" if CORPUS == "stdlib" else f"astcache_{CORPUS}.npz")
MIN_COUNT = 200


def child_sym(n):
    if isinstance(n, ast.Name): return "N:" + n.id
    if isinstance(n, ast.Constant): return "C:" + type(n.value).__name__
    return type(n).__name__


def optional_fields():
    opt = {}
    for name in dir(ast):
        cls = getattr(ast, name)
        if isinstance(cls, type) and issubclass(cls, ast.AST):
            fs = [f for f, ty in getattr(cls, "_field_types", {}).items() if str(ty).startswith("ast.") and "None" in str(ty)]
            if fs: opt[name] = tuple(fs)
    return opt


def encode_file(tree, fi, node0, OPT):
    """returns unit rows (file, ctx, sym, dp, par, node) and node rows (depth, internal), node ids from node0"""
    units, nodes = [], [[0, 0]]                      # root
    def walk(nd, nid, k):
        for fld, val in ast.iter_fields(nd):
            ctx = type(nd).__name__ + "." + fld
            if isinstance(val, list):
                if val and not isinstance(val[0], ast.AST):
                    continue
                for c in val:
                    if isinstance(c, ast.AST) and not isinstance(c, ast.expr_context):
                        emit(c, ctx, nid, k)
                units.append((fi, ctx, "<STOP>", k, nid, -1))
            elif isinstance(val, ast.AST):
                if not isinstance(val, ast.expr_context):
                    emit(val, ctx, nid, k)
            elif val is None and fld in OPT.get(type(nd).__name__, ()):
                units.append((fi, ctx, "<NONE>", k, nid, -1))
    def emit(c, ctx, pid, k):
        cid = node0 + len(nodes); nodes.append([k + 1, 0])
        units.append((fi, ctx, child_sym(c), k, pid, cid))
        n_before = len(units)
        walk(c, cid, k + 1)
        if len(units) > n_before:
            nodes[cid - node0][1] = 1
    walk(tree, node0, 0)
    if units: nodes[0][1] = 1
    return units, nodes


def build():
    lib = os.path.dirname(os.__file__)
    if CORPUS == "stdlib":
        files = sorted(f for f in glob.glob(os.path.join(lib, "**", "*.py"), recursive=True) if "site-packages" not in f)
    else:
        files = sorted(glob.glob(os.path.join(lib, "site-packages", CORPUS, "**", "*.py"), recursive=True))
    sys.setrecursionlimit(20000)
    OPT = optional_fields()
    U, N, NF = [], [], []
    for fi, f in enumerate(files):
        try:
            with open(f, "rb") as fh:
                tr = ast.parse(fh.read())
            u, n = encode_file(tr, fi, len(N), OPT)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        U += u; N += n; NF += [fi] * len(n)
    cnt = Counter(r[2] for r in U)
    keep = sorted(w for w, n in cnt.items() if n >= MIN_COUNT)
    sym = keep + ["<OTHER>"]; sv = {w: i for i, w in enumerate(sym)}
    ctxs = sorted({r[1] for r in U}); tv = {w: i for i, w in enumerate(ctxs)}
    d = dict(
        ufile=np.array([r[0] for r in U], np.int32), t=np.array([tv[r[1]] for r in U], np.int16),
        x=np.array([sv.get(r[2], len(sym) - 1) for r in U], np.int16), dp=np.array([r[3] for r in U], np.int16),
        par=np.array([r[4] for r in U], np.int32), node=np.array([r[5] for r in U], np.int32),
        node_depth=np.array([n[0] for n in N], np.int16), node_internal=np.array([n[1] for n in N], np.int8),
        node_file=np.array(NF, np.int32), symbols=np.array(sym), contexts=np.array(ctxs))
    return d


def load(path=CACHE):
    if not os.path.exists(path):
        np.savez_compressed(path, **build())
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def features(d):
    """per unit: prev = symbol of the previous sibling in the same field (C if none); tp = context of the unit that
    created the parent (T if the parent is the file root).  Both are functions of earlier units only."""
    n = len(d["x"]); C, T = len(d["symbols"]), len(d["contexts"])
    x = d["x"].astype(np.int64); t = d["t"].astype(np.int64); par = d["par"].astype(np.int64); node = d["node"].astype(np.int64)
    o = np.lexsort((np.arange(n), t, par))                       # siblings of one field, in order
    same = (par[o][1:] == par[o][:-1]) & (t[o][1:] == t[o][:-1])
    prev = np.full(n, C, np.int64); prev[o[1:][same]] = x[o[:-1][same]]
    creator = np.full(len(d["node_depth"]), -1, np.int64); creator[node[node >= 0]] = np.nonzero(node >= 0)[0]
    cp = creator[par]
    tp = np.where(cp >= 0, t[np.maximum(cp, 0)], T)
    return prev, tp


def split(d, seed=0):
    """the test/pool split of code_claim1.py: first half of a seed-0 permutation of the parsed file ids is test"""
    files = np.unique(d["ufile"])
    perm = np.random.default_rng(seed).permutation(files)
    return perm[: len(files) // 2], perm[len(files) // 2:]


if __name__ == "__main__":
    d = load()
    print({k: (v.shape, v.dtype) for k, v in d.items()}, os.path.getsize(CACHE) / 1e6, "MB")
