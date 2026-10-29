"""Diagnostic for exp_structured.py failures: local optima or non-identifiability?
For failing configurations, 8 independent generalized-EM runs (different initial theta), each carried
through the whole ladder v = 8 -> 256.  If the true maps are identified by the likelihood (Theorem ngt),
higher final log-likelihood must go with low map error / high R^2, and selection by likelihood over
enough restarts fixes the failures.
"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(globals().get("__file__", "./x"))))
import ngt_nd as nd, codebook_torch as cbt, codebook_smooth as sm

CONFIGS = [(201, 3, 8, 1), (201, 9, 8, 1), (200, 1, 8, 1), (201, 1, 7, 1)]      # (kernel, U, L, seed)
VS = [8, 16, 32, 64, 128, 256]


def r2v(p, R):
    return float(1 - ((p - R) ** 2).sum() / ((R - R.mean(0)) ** 2).sum())


def main():
    print(json.dumps(dict(event="start", device=cbt.DEV)), flush=True)
    for kseed, U, L, seed in CONFIGS:
        ker = nd.random_kernel(2, np.random.default_rng(kseed))
        T, R = nd.sample_trees(ker, 300, 6, np.random.default_rng(7))
        Z = nd.sample_trees(ker, 3000, 0, np.random.default_rng(3))[0][:, 0]
        X, _ = nd.sample_trees(ker, U, L, np.random.default_rng(1000 * kseed + seed))
        for rs in range(8):
            fits = sm.fit(X, VS, seed=1000 + rs, em_iters=40, restarts=1)
            k, cb, ll = fits[256]
            k8, _, ll8 = fits[8]
            print(json.dumps(dict(kernel=kseed, U=U, L=L, N1=U * 3 ** (L - 1), data_seed=seed, restart=rs,
                                  ll8_per_leaf=round(ll8 / X.shape[0] / X.shape[1], 5),
                                  ll256_per_leaf=round(ll / X.shape[0] / X.shape[1], 5),
                                  r2_256=round(r2v(cbt.q_root_value(cb, T), R), 4),
                                  phi_err_256=round(sm.phi_error(k, ker.phi, Z, ker.psi), 3))), flush=True)
    print(json.dumps(dict(event="done")), flush=True)


if __name__ == "__main__":
    main()
