"""V1' follow-up: is the v=32 shortfall of Table v1p lack of data or EM local optima?
Coarse-to-fine EM (v = 4 -> 8 -> 16 -> 32 [-> 64]) on GPU, R^2 on depth-8 test trees at every size.
Same kernels/seeds/test sets as exp_codebook.py; adds N1 = 59049.  Zero labels.
Prediction: if the shortfall was optimization, coarse-to-fine lifts v=32 at N1=19683 above the random-
init value (0.61, kernel 0); if it was data, v=32 improves only with N1.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(globals().get("__file__", "./x"))))
import ngt, codebook_torch as cbt

DATA = [(3, 8), (9, 8), (27, 8)]


def main():
    print(json.dumps(dict(event="start", device=cbt.DEV)), flush=True)
    tests = {}
    for kseed in (0, 1):
        true = ngt.random_true_kernel(np.random.default_rng(100 + kseed))
        Xt, rt = ngt.sample_trees(true, 400, 8, np.random.default_rng(7))
        tests[kseed] = (true, Xt, rt, float(1 - ((ngt.root_posterior_mean(true, Xt) - rt) ** 2).mean() / rt.var()))
    for (U, L) in DATA:
        for kseed in (0, 1):
            true, Xt, rt, r2o = tests[kseed]
            for seed in range(2):
                X, _ = ngt.sample_trees(true, U, L, np.random.default_rng(1000 * kseed + seed))
                t = time.time()
                vmax = 64 if U == 27 else 32
                fits = cbt.fit_coarse_to_fine(X, vmax, np.random.default_rng(seed))
                for v, (cb, ll) in sorted(fits.items()):
                    r2 = float(1 - ((cbt.root_value(cb, Xt) - rt) ** 2).mean() / rt.var())
                    print(json.dumps(dict(kernel=kseed, U=U, L=L, N1=U * 3 ** (L - 1), seed=seed, v=v,
                                          init="coarse_to_fine", r2=round(r2, 4), r2_oracle=round(r2o, 4),
                                          ll_per_leaf=round(ll / X.size, 4), tau=round(cb.tau, 4),
                                          secs_total=round(time.time() - t, 1))), flush=True)
    print(json.dumps(dict(event="done")), flush=True)


if __name__ == "__main__":
    main()
