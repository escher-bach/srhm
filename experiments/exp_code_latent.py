"""Learning curves of the latent tree model on Python ASTs (where.tex sec. prog-latent), H = 1..16 by splitting.
Train on N random files of the pool (validation files excluded), report validation and test nats per unit per H.
Run locally:  python exp_code_latent.py [smoke]      Colab: bundle with make_bundle.py (see run_code_latent.sh)
"""
import json, math, os, sys, time
import numpy as np
import torch
import astdata
from code_latent import Corpus, LatentTree

BUDGET = float(os.environ.get("BUDGET_SECONDS", "1e9")); T0 = time.time()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_VAL = 20
H_MAX = int(os.environ.get("H_MAX", "16"))
EM_ITERS, PROBE_ITERS, RESTARTS, TOL = 40, 6, 2, 1e-5


def log(**k):
    k["t"] = round(time.time() - T0, 1); print(json.dumps(k), flush=True)
    if os.environ.get("PROGRESS_FILE"):
        with open(os.environ["PROGRESS_FILE"], "a") as f:
            f.write(json.dumps(k) + "\n")


def nats(m, c):
    return -m.loglik(c) / c.n_units


def curve_point(d, files, val, te, seed):
    rng = np.random.default_rng(1000 + seed)
    tr = Corpus(d, files, DEV)
    m = LatentTree(4, len(d["contexts"]), len(d["symbols"]), 1, DEV)
    m.set_marginal(tr)
    m.em(tr, 3)
    out = {1: (nats(m, val), nats(m, te), -m.loglik(tr) / tr.n_units)}
    while m.H < H_MAX and time.time() - T0 < BUDGET - 300:
        base = m.state(); best = None
        for r in range(RESTARTS):
            m.load_state(base); m.split(rng)
            h = m.em(tr, PROBE_ITERS, tol=TOL)
            if best is None or h[-1] > best[0]:
                best = (h[-1], m.state())
        m.load_state(best[1])
        hist = m.em(tr, EM_ITERS, tol=TOL)
        out[m.H] = (nats(m, val), nats(m, te), -m.loglik(tr) / tr.n_units)
        log(stage=True, files=len(files), seed=seed, H=m.H, em_iters=len(hist), val=round(out[m.H][0], 5), test=round(out[m.H][1], 5))
    return tr.n_units, out


def main():
    smoke = "smoke" in sys.argv or os.environ.get("SMOKE") == "1"
    d = astdata.load()
    test, pool = astdata.split(d)
    rs = np.random.default_rng(7)
    val = rs.choice(pool, N_VAL, replace=False); pool = np.setdiff1d(pool, val)
    if smoke:
        test = test[:40]
    val_c, te_c = Corpus(d, val, DEV), Corpus(d, test, DEV)
    log(dev=DEV, pool=len(pool), test_units=te_c.n_units, val_units=val_c.n_units)
    sizes = [2, 4] if smoke else [1, 4, 16, 64, 256, len(pool)]
    if os.environ.get("FILES"):
        sizes = [min(int(v), len(pool)) for v in os.environ["FILES"].split(",")]
    global H_MAX
    if smoke: H_MAX = 4
    for n in sizes:
        for s in range(1 if n == len(pool) else 2):
            if time.time() - T0 > BUDGET - 300:
                log(stopped="budget"); return
            files = np.random.default_rng(100 * n + s).choice(pool, n, replace=False)
            units, out = curve_point(d, files, val_c, te_c, s)
            best = min(out, key=lambda h: out[h][0])
            log(files=n, seed=s, train_units=units,
                by_H={h: {"val": round(v[0], 5), "test": round(v[1], 5), "train": round(v[2], 5)} for h, v in out.items()},
                best_H=best, best_test=round(out[best][1], 5))


if __name__ == "__main__":
    main()
