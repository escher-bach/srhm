"""Compare validity naming vs structural naming under partial tying; report error vs (1-rho)."""
import sys
import numpy as np
from ptrhm import PartialGrammar, pt_learn, evaluate

L, P, seeds = 8, 256, 6
rhos = [float(r) for r in sys.argv[1:]] or [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
for rho in rhos:
    for naming in ["validity", "structural"]:
        ok, errs = [], []
        for seed in range(seeds):
            PG = PartialGrammar(8, 4, 2, L, rho, np.random.default_rng(2000 + seed))
            X, _ = PG.sample(P, np.random.default_rng(seed))
            tabs = pt_learn(PG, X, naming=naming)
            obs = np.flatnonzero(tabs[1] >= 0)
            ok.append(np.array_equal(tabs[1][obs], PG.base.tab[obs]))
            errs.append(evaluate(PG, tabs, 2000, np.random.default_rng(99)))
        print(f"rho={rho:.2f} {naming:10s}: base grammar exact {sum(ok)}/{seeds}   "
              f"root error mean={np.mean(errs):.3f} (per grammar {np.round(errs, 2)})   "
              f"1-rho={1 - rho:.2f}", flush=True)
