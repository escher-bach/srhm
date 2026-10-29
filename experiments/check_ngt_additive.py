"""Claim-2 prediction for the nonlinear Gaussian tree: how much of the Bayes-optimal root prediction is
available to ADDITIVE leaf features (sum over leaves of a_l x_l + b_l x_l^2), by depth?  Held-out R^2 of a
least-squares additive predictor (20k training trees) against the Bayes oracle (grid BP), kernels 0, 1."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ngt

for k in (0, 1):
    true = ngt.random_true_kernel(np.random.default_rng(100 + k))
    for L in range(3, 8):
        X, r = ngt.sample_trees(true, 20000, L, np.random.default_rng(11))
        Xt, rt = ngt.sample_trees(true, 2000, L, np.random.default_rng(12))
        F = lambda Z: np.concatenate([Z, Z ** 2, np.ones((len(Z), 1))], 1)
        w, *_ = np.linalg.lstsq(F(X), r, rcond=None)
        r2 = lambda p: float(1 - ((p - rt) ** 2).mean() / rt.var())
        rec = dict(kernel=k, L=L, r2_additive=round(r2(F(Xt) @ w), 4),
                   r2_oracle=round(r2(ngt.root_posterior_mean(true, Xt)), 4))
        rec["ratio"] = round(rec["r2_additive"] / rec["r2_oracle"], 3)
        print(json.dumps(rec), flush=True)
