"""V1 (srhm.tex section 8.6): the pure zero-label learner on bounded nonlinear Gaussian tied trees.

  P1  one unlabelled tree, L = 3..8: kernel error, psi error, root R^2 against the Bayes oracle.
  P2  fixed total bottom grids N_1 in {81, 243, 729}, split into U trees of depth L: error vs split.
2 random true kernels x 4 data seeds.  Output: exp_ngt_results.jsonl (one line per fit).
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ngt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_ngt_results.jsonl")


def job(args):
    kseed, U, L, seed = args
    true = ngt.random_true_kernel(np.random.default_rng(100 + kseed))
    X, _ = ngt.sample_trees(true, U, L, np.random.default_rng(1000 * kseed + seed))
    t = time.time()
    ker, ll = ngt.fit(X, np.random.default_rng(seed))
    err, perr = ngt.kernel_error(ker, true)
    Xt, rt = ngt.sample_trees(true, 400, L, np.random.default_rng(7))
    r2 = lambda p: float(1 - ((p - rt) ** 2).mean() / rt.var())
    return dict(kernel=kseed, U=U, L=L, seed=seed, N1=U * 3 ** (L - 1), labels=0,
                kernel_err=round(err, 4), psi_rel_err=round(perr, 4),
                r2_learned=round(r2(ngt.root_posterior_mean(ker, Xt)), 4),
                r2_oracle=round(r2(ngt.root_posterior_mean(true, Xt)), 4),
                secs=round(time.time() - t, 1))


def main():
    configs = set()
    for L in range(3, 9):                                   # P1: one tree
        configs.add((1, L))
    for N1, splits in {81: [(1, 5), (3, 4), (9, 3)],        # P2: same N_1, different packaging
                       243: [(1, 6), (3, 5), (9, 4), (27, 3)],
                       729: [(1, 7), (3, 6), (9, 5), (27, 4), (81, 3)]}.items():
        configs.update(splits)
    jobs = [(k, U, L, s) for k in (0, 1) for (U, L) in sorted(configs, key=lambda c: -c[1])
            for s in range(4)]
    open(OUT, "w").close()
    with Pool(3) as pool:
        for rec in pool.imap_unordered(job, jobs):
            print(json.dumps(rec), flush=True)
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
