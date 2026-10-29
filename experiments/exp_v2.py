"""V2 (srhm.tex section 8.8): hard-quantized codebook tree in d = 2, zero labels, GPU.

Two random true kernels (ngt_nd.random_kernel, singular values in [1.3, 2.0]), 2 data seeds.
Training data (U trees, depth L): (3,8) N1=6561, (9,8) N1=19683, (27,8) N1=59049, and the packaging
point (81,5) N1=6561.  Coarse-to-fine EM, v = 4 -> ... -> 256 (TSVQ depth 8).
Test: 300 trees of depth 6; root R^2 (vector) against the grid oracle (101^2 grid, box [-5,5]^2).
Predictions: R1 no floor; R2 larger v needed than in d = 1; R3 packaging-invariant at fixed N1.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(globals().get("__file__", "./x"))))
import ngt_nd as nd, codebook_torch as cbt, codebook_vq as vq

DATA = [(3, 8), (9, 8), (27, 8), (81, 5)]


def r2v(p, R):
    return float(1 - ((p - R) ** 2).sum() / ((R - R.mean(0)) ** 2).sum())


def main():
    print(json.dumps(dict(event="start", device=cbt.DEV)), flush=True)
    for kseed in (200, 201):
        ker = nd.random_kernel(2, np.random.default_rng(kseed))
        T, R = nd.sample_trees(ker, 300, 6, np.random.default_rng(7))
        t = time.time()
        r2o = r2v(nd.GridOracle(ker, n=101, lim=5.0).root_mean(T), R)
        print(json.dumps(dict(event="oracle", kernel=kseed, r2_oracle=round(r2o, 4),
                              secs=round(time.time() - t, 1))), flush=True)
        for (U, L) in DATA:
            for seed in range(2):
                X, _ = nd.sample_trees(ker, U, L, np.random.default_rng(1000 * kseed + seed))
                t = time.time()
                fits = vq.vq_fit_coarse_to_fine(X, 256, np.random.default_rng(seed))
                for v, (cb, ll) in sorted(fits.items()):
                    print(json.dumps(dict(kernel=kseed, U=U, L=L, N1=U * 3 ** (L - 1), seed=seed, v=v,
                                          r2=round(r2v(cbt.q_root_value(cb, T), R), 4),
                                          r2_oracle=round(r2o, 4), ll_per_leaf=round(ll / X.shape[0] / X.shape[1], 4),
                                          secs_total=round(time.time() - t, 1))), flush=True)
    print(json.dumps(dict(event="done")), flush=True)


if __name__ == "__main__":
    main()
