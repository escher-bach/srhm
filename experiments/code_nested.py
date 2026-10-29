"""Nested histories on programs (where.tex sec. prog-nested): every node adapts around its parent's prediction,
    p_a(x | r) = (alpha_a p_parent(a)(x | r) + c_a(x | r)) / (alpha_a + n_a(r)),   p_(above root) = R (shared H = 1 rule),
with c_a, n_a the counts of EARLIER units inside a's subtree (a contiguous block in pre-order) in rule row r.
alpha_a = alpha_root for the file root and alpha_in for every inner node (tied: alpha_root = alpha_in).
Usage: python code_nested.py [test|curve]
"""
import json, os, sys, time
import numpy as np
import astdata
from code_gap import h1_rule

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "code_nested_results.jsonl")
T0 = time.time()


def log(**k):
    k["t"] = round(time.time() - T0, 1); print(json.dumps(k), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(k) + "\n")


def file_arrays(d, f, T):
    um = d["ufile"] == f
    x, t = d["x"][um].astype(np.int64), d["t"][um].astype(np.int64)
    dp = d["dp"][um].astype(np.int64); par, node = d["par"][um].astype(np.int64), d["node"][um].astype(np.int64)
    row = np.minimum(dp, 3) * T + t
    n = len(x); base = par.min()                                  # the root is the smallest node id of the file
    creator = np.full(max(par.max(), node.max()) - base + 1, -1, np.int64)
    creator[node[node >= 0] - base] = np.nonzero(node >= 0)[0]    # position of the unit that created each node
    cp = creator[par - base]                                      # creator of the parent (-1: parent is the root)
    return x, row, dp, cp


def range_counter(code, n):
    o = np.argsort(code, kind="stable"); comp = code[o].astype(np.float64) * (n + 1) + o
    rank = np.empty(n, np.int64); rank[o] = np.arange(n)
    def count(idx, start):                                        # units j with start < j < idx[i], code_j == code_idx[i]
        return rank[idx] - np.searchsorted(comp, code[idx].astype(np.float64) * (n + 1) + start + 1)
    return count


def nested_nll(x, row, dp, cp, r, pairs):
    """sum of -log p over the file's units, for each (alpha_root, alpha_in) in pairs"""
    n = len(x)
    cx, cr = range_counter(row * 100000 + x, n), range_counter(row, n)
    # chain of window starts: level 0 = parent's subtree, level k = k-th ancestor above; the root's start is -1
    starts = [cp]
    while (starts[-1] >= 0).any():
        s = starts[-1]; starts.append(np.where(s >= 0, cp[np.maximum(s, 0)], -2))   # -2: above the root (no such level)
    out = {}
    for ar, ai in pairs:
        p = r.copy()
        for k in range(len(starts) - 1, -1, -1):                  # outermost scope first
            s = starts[k]; idx = np.nonzero(s >= -1)[0]
            a = np.where(s[idx] == -1, ar, ai)
            p[idx] = (a * p[idx] + cx(idx, s[idx])) / (a + cr(idx, s[idx]))
        out[(ar, ai)] = float(-np.log(p).sum())
    return out


def evaluate(d, R, files, pairs):
    T = len(d["contexts"]); tot = {p: 0.0 for p in pairs}; n = 0
    for f in files:
        x, row, dp, cp = file_arrays(d, f, T)
        res = nested_nll(x, row, dp, cp, R[row, x], pairs); n += len(x)
        for p in pairs: tot[p] += res[p]
    return {p: v / n for p, v in tot.items()}


def test():
    """brute force: explicit ancestor sets on random pre-order trees"""
    rng = np.random.default_rng(0)
    for trial in range(5):
        par, node, x, row = [], [], [], []
        nid = [0]
        def grow(p, depth):
            for _ in range(rng.integers(1, 4)):
                x.append(int(rng.integers(3))); row.append(int(rng.integers(2))); par.append(p)
                if rng.random() < 0.25:
                    node.append(-1); continue
                nid[0] += 1; c = nid[0]; node.append(c)
                if depth < 4 and rng.random() < 0.6: grow(c, depth + 1)
        grow(0, 0)
        n = len(x); x, row, par, node = map(np.array, (x, row, par, node))
        parent_of = {int(node[i]): int(par[i]) for i in range(n) if node[i] >= 0}
        def chain(p):                                              # p and its ancestors, innermost first
            c = [p]
            while c[-1] != 0: c.append(parent_of[c[-1]])
            return c
        R = rng.dirichlet(np.ones(3), size=2); r = R[row, x]
        ar, ai = 3.0, 0.7; nll = 0.0
        for i in range(n):
            ch = chain(int(par[i])); p = r[i]
            for a in reversed(ch):
                inside = [j for j in range(i) if a in chain(int(par[j])) and row[j] == row[i]]
                c = sum(1 for j in inside if x[j] == x[i])
                al = ar if a == 0 else ai
                p = (al * p + c) / (al + len(inside))
            nll -= np.log(p)
        creator = np.full(nid[0] + 1, -1, np.int64); creator[node[node >= 0]] = np.nonzero(node >= 0)[0]
        cp = creator[par]
        got = nested_nll(x, row, None, cp, r.copy(), [(ar, ai)])[(ar, ai)]
        print(f"trial {trial}: units {n}, brute {nll:.10f}, fast {got:.10f}")
        assert abs(nll - got) < 1e-9
    print("ALL OK")


def curve():
    d = astdata.load()
    test_f, pool = astdata.split(d)
    val = np.random.default_rng(7).choice(pool, 20, replace=False); pool = np.setdiff1d(pool, val)
    grid = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    tied = [(a, a) for a in grid]; untied = [(a, b) for a in grid for b in grid]
    for n in (1, 4, 16, 64, 256, len(pool)):
        for s in range(1 if n == len(pool) else 2):
            files = np.random.default_rng(100 * n + s).choice(pool, n, replace=False)   # the files of the Colab runs
            R = h1_rule(d, files)
            v = evaluate(d, R, val, untied)
            bt = min(tied, key=v.get); bu = min(untied, key=v.get)
            te = evaluate(d, R, test_f, sorted({bt, bu}))
            log(data="nested", files=n, seed=s, alpha_tied=bt[0], test_tied=round(te[bt], 5),
                alpha_untied=list(bu), test_untied=round(te[bu], 5))


if __name__ == "__main__":
    test() if "test" in sys.argv else curve()
