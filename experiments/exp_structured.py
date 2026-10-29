"""Structured (smooth) kernel on the adaptive codebook, d = 2 (srhm.tex section 8.10).
Same processes, oracles and test trees as exp_v2.py.  Data: (1,7) N1=729, (1,8) 2187, (3,8) 6561,
(9,8) 19683; coarse-to-fine v = 8 -> 256 with theta reused; 2 kernels x 2 seeds; zero labels.
Reports root R^2 (vs oracle) and the error of the learned maps phi_i under the stationary law (S3).
"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(globals().get("__file__", "./x"))))
import ngt_nd as nd, codebook_torch as cbt, codebook_vq as vq, codebook_smooth as sm

DATA = [(1, 7), (1, 8), (3, 8), (9, 8)]
VS = [8, 16, 32, 64, 128, 256]


def r2v(p, R):
    return float(1 - ((p - R) ** 2).sum() / ((R - R.mean(0)) ** 2).sum())


def main():
    print(json.dumps(dict(event="start", device=cbt.DEV)), flush=True)
    for kseed in (200, 201):
        ker = nd.random_kernel(2, np.random.default_rng(kseed))
        T, R = nd.sample_trees(ker, 300, 6, np.random.default_rng(7))
        Z = nd.sample_trees(ker, 3000, 0, np.random.default_rng(3))[0][:, 0]          # stationary latents
        r2o = r2v(nd.GridOracle(ker, n=101, lim=5.0).root_mean(T), R)
        print(json.dumps(dict(event="oracle", kernel=kseed, r2_oracle=round(r2o, 4))), flush=True)
        for (U, L) in DATA:
            for seed in range(2):
                X, _ = nd.sample_trees(ker, U, L, np.random.default_rng(1000 * kseed + seed))
                t = time.time()
                fits = sm.fit(X, VS, seed=seed, em_iters=40, restarts=2)
                for v, (k, cb, ll) in sorted(fits.items()):
                    print(json.dumps(dict(kernel=kseed, U=U, L=L, N1=U * 3 ** (L - 1), seed=seed, v=v,
                                          r2=round(r2v(cbt.q_root_value(cb, T), R), 4), r2_oracle=round(r2o, 4),
                                          phi_err=round(sm.phi_error(k, ker.phi, Z, ker.psi), 4),
                                          psi_hat=round(float(torch.exp(k.log_psi.detach())), 4),
                                          secs_total=round(time.time() - t, 1))), flush=True)
    print(json.dumps(dict(event="done")), flush=True)


if __name__ == "__main__":
    main()
