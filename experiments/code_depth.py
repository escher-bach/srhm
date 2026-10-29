"""hist.tex sec. depth: couplings per depth. (E6) the four moment couplings, no validation. (E7) couplings chosen per
depth on the validation files by coordinate descent, for comparison with the moments."""
import json, os, time
import numpy as np
import astdata
from code_gap import h1_rule
from code_cred import Files

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "code_depth_results.jsonl")
T0 = time.time()
MOMENT = np.array([4.78, 1.94, 4.66, np.inf])
GRID = (1, 2, 4, 8, 16, 32, 64, 128, np.inf)


def log(**k):
    k["t"] = round(time.time() - T0, 1); print(json.dumps(k), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(k) + "\n")


def coord_descent(V, R, start, sweeps=3):
    a = np.array(start, float); best = V.nll(R, a, by="depth")
    for _ in range(sweeps):
        changed = False
        for j in range(len(a)):
            for g in GRID:
                if g == a[j]: continue
                b = a.copy(); b[j] = g; v = V.nll(R, b, by="depth")
                if v < best - 1e-6: a, best, changed = b, v, True
        if not changed: break
    return a, best


def main():
    d = astdata.load()
    test_f, pool = astdata.split(d)
    val = np.random.default_rng(7).choice(pool, 20, replace=False); pool = np.setdiff1d(pool, val)
    V, Te = Files(d, val), Files(d, test_f)
    R = h1_rule(d, pool)
    log(data="E6", alpha=[4.78, 1.94, 4.66, "inf"], val=round(V.nll(R, MOMENT, by="depth"), 5),
        test=round(Te.nll(R, MOMENT, by="depth"), 5), test_single_32=round(Te.nll(R, 32.0), 5))
    a, v = coord_descent(V, R, [32, 32, 32, 32])
    log(data="E7", alpha=[("inf" if np.isinf(z) else z) for z in a], val=round(v, 5), test=round(Te.nll(R, a, by="depth"), 5))
    a5, v5 = coord_descent(V, R, list(a) + [a[-1]])                     # one more level, as a check on the cut at depth 3
    log(data="E7_5levels", alpha=[("inf" if np.isinf(z) else z) for z in a5], val=round(v5, 5), test=round(Te.nll(R, a5, by="depth"), 5))


if __name__ == "__main__":
    main()
