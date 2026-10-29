"""The coupling alpha from pair statistics (hist.tex Proposition alpha, sec. mom): no validation data.

For a scope a and a rule row r: W = P(two units of the SAME child scope of a disagree), B = P(two units of DIFFERENT
child scopes of a disagree); model: W / B = alpha / (alpha + 1) for every a at every depth, so
    alpha_hat = sum w W / (sum w B - sum w W),   w shared by numerator and denominator.
File roots are the children of one virtual scope above them (depth -1).
Usage: python code_mom.py
"""
import json, os, time
import numpy as np
import astdata
from code_gap import h1_rule
from code_cred import Files, NROW, NSYM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "code_mom_results.jsonl")
T0 = time.time()


def log(**k):
    k["t"] = round(time.time() - T0, 1); print(json.dumps(k), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(k) + "\n")


def groups(d, files):
    """per (parent scope a, row r): W_hat, B_hat, weight, depth of a, row"""
    T = len(d["contexts"])
    um = np.isin(d["ufile"], files)
    x = d["x"][um].astype(np.int64); par = d["par"][um].astype(np.int64); node = d["node"][um].astype(np.int64)
    row = np.minimum(d["dp"][um], 3).astype(np.int64) * T + d["t"][um].astype(np.int64)
    nn = len(d["node_depth"])
    pn = np.full(nn + 1, nn, np.int64)                    # parent scope of each scope; roots -> virtual scope nn
    pn[node[node >= 0]] = par[node >= 0]
    depth = np.r_[d["node_depth"].astype(np.int64), -1]
    c = par; a = pn[c]
    # k[c, r, x]
    kc, k = np.unique((c * NROW + row) * NSYM + x, return_counts=True)
    cr = kc // NSYM
    ucr, inv = np.unique(cr, return_inverse=True)
    n_c = np.bincount(inv, weights=k); q_c = np.bincount(inv, weights=k.astype(float) ** 2)
    a_of = pn[ucr // NROW]; r_of = ucr % NROW
    ar = a_of * NROW + r_of
    uar, inv2 = np.unique(ar, return_inverse=True)
    PW = np.bincount(inv2, weights=n_c * (n_c - 1) / 2); DW = np.bincount(inv2, weights=(n_c ** 2 - q_c) / 2)
    ntot = np.bincount(inv2, weights=n_c)
    # K[a, r, x]
    ka, K = np.unique((a * NROW + row) * NSYM + x, return_counts=True)
    ia = np.searchsorted(uar, ka // NSYM)
    K2 = np.bincount(ia, weights=K.astype(float) ** 2, minlength=len(uar))
    PA = ntot * (ntot - 1) / 2; DA = (ntot ** 2 - K2) / 2
    PB, DB = PA - PW, DA - DW
    ok = (PW > 0) & (PB > 0)
    W, B = DW[ok] / PW[ok], DB[ok] / PB[ok]
    w = PW[ok] * PB[ok] / (PW[ok] + PB[ok])
    return dict(W=W, B=B, w=w, depth=depth[uar[ok] // NROW], row=uar[ok] % NROW)


def alpha_hat(g, m=None):
    m = np.ones(len(g["w"]), bool) if m is None else m
    sw, sb = (g["w"][m] * g["W"][m]).sum(), (g["w"][m] * g["B"][m]).sum()
    return float(sw / (sb - sw)) if sb > sw else float("inf")


def main():
    d = astdata.load()
    test_f, pool = astdata.split(d)
    val = np.random.default_rng(7).choice(pool, 20, replace=False); pool = np.setdiff1d(pool, val)
    g = groups(d, pool)
    a_all = alpha_hat(g)
    by_depth = {}
    for lo, hi, name in ((-1, -1, "above root"), (0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 99, "5+")):
        m = (g["depth"] >= lo) & (g["depth"] <= hi)
        by_depth[name] = dict(alpha=round(alpha_hat(g, m), 2), groups=int(m.sum()),
                              W=round(float((g["w"][m] * g["W"][m]).sum() / g["w"][m].sum()), 4),
                              B=round(float((g["w"][m] * g["B"][m]).sum() / g["w"][m].sum()), 4))
    log(data="moment", alpha_all=round(a_all, 2), groups=len(g["w"]), by_depth=by_depth)
    # per row
    a_row = np.full(NROW, a_all); n_est = 0
    for r in np.unique(g["row"]):
        m = g["row"] == r
        if m.sum() >= 100:
            ar = alpha_hat(g, m)
            if np.isfinite(ar):
                a_row[r] = float(np.clip(ar, 1, 512)); n_est += 1
    est = a_row[a_row != a_all]
    log(data="moment_rows", rows_estimated=n_est, quantiles={q: round(float(np.quantile(est, q)), 2) for q in (0.1, 0.25, 0.5, 0.75, 0.9)})
    # single files (descriptive): the estimate from one training file, for files with >= 2500 units
    sizes = np.bincount(d["ufile"])[pool]
    one = []
    for f in pool[sizes >= 2500][:60]:
        one.append(alpha_hat(groups(d, np.array([f]))))
    one = np.array(one)
    log(data="moment_one_file", files=len(one), quantiles={q: round(float(np.quantile(one, q)), 2) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        within_factor_2_of_pool=float(np.mean((one >= a_all / 2) & (one <= a_all * 2))))
    # test loss with the moment estimates (shared rule from all pool files)
    R = h1_rule(d, pool)
    Te = Files(d, test_f)
    log(data="moment_test", alpha_32=round(Te.nll(R, 32.0), 5), alpha_moment=round(Te.nll(R, a_all), 5),
        alpha_per_row=round(Te.nll(R, a_row), 5))


if __name__ == "__main__":
    main()
