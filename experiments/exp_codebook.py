"""V1' (srhm.tex section 8.7): tied codebook tree on the V1 data (bounded nonlinear Gaussian trees).

Codebook sizes v in {4, 8, 16, 32}; training data (U trees of depth L):
  (1,7) N1=729, (1,8) N1=2187, (3,8) N1=6561, (9,8) N1=19683,    and at N1=6561 the shallow splits
  (81,5), (243,4) to test packaging and depth extrapolation (Q2, Q3).
Always tested on 400 depth-8 trees: root R^2 of E[c_root | leaves] against the Bayes oracle.  Zero labels.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ngt, codebook as cbm

OUT = os.path.join(os.path.dirname(os.path.abspath(globals().get("__file__", "./x"))), "exp_codebook_results.jsonl")
DATA = [(1, 7), (1, 8), (3, 8), (9, 8), (81, 5), (243, 4)]


def job(args):
    kseed, v, U, L, seed = args
    true = ngt.random_true_kernel(np.random.default_rng(100 + kseed))
    X, _ = ngt.sample_trees(true, U, L, np.random.default_rng(1000 * kseed + seed))
    t = time.time()
    cb, ll = cbm.fit(X, v, np.random.default_rng(seed))
    Xt, rt = ngt.sample_trees(true, 400, 8, np.random.default_rng(7))
    r2 = lambda p: float(1 - ((p - rt) ** 2).mean() / rt.var())
    return dict(kernel=kseed, v=v, U=U, L=L, N1=U * 3 ** (L - 1), seed=seed, labels=0,
                n_params=3 * v * (v - 1) + v + 1, r2=round(r2(cbm.root_value(cb, Xt)), 4),
                r2_oracle=round(r2(ngt.root_posterior_mean(true, Xt)), 4), tau=round(cb.tau, 4),
                ll_per_leaf=round(ll / X.size, 4), secs=round(time.time() - t, 1))


if __name__ == "__main__":
    jobs = [(k, v, U, L, s) for (U, L) in DATA for v in (32, 16, 8, 4) for k in (0, 1) for s in range(2)]
    jobs.sort(key=lambda j: -j[2] * 3 ** (j[3] - 1) * j[1] ** 2)          # largest first
    on_colab = "BUDGET_SECONDS" in os.environ
    workers = int(os.environ.get("WORKERS", max(1, os.cpu_count() - 1) if on_colab else 3))
    print(json.dumps(dict(event="start", workers=workers, jobs=len(jobs))), flush=True)
    open(OUT, "w").close()
    with Pool(workers) as pool:
        for rec in pool.imap_unordered(job, jobs):
            print(json.dumps(rec), flush=True)
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
