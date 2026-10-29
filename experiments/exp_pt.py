"""Partial tying sweep: zero-label root error vs (rho, L, P), against the exact zero-label floor.

floor = (K - max_a k_a) / (v m): K fresh slots at the root level, k_a of them belonging to
class a.  This is the Bayes error on the root's own rule for a learner that knows everything
else (Thm. PT-lower): a fresh root tuple has no context, so the best zero-label guess is the
class with the most fresh slots.
"""
import json, os, sys, time
import numpy as np
from ptrhm import PartialGrammar, pt_learn, evaluate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_pt_results.jsonl")
V, M, S = 8, 4, 2

for rho in [0.9, 0.8, 0.7, 0.6, 0.5]:
    for L in [6, 8, 10]:
        for P in [64, 256, 1024, 4096]:
            if P * S ** (L - 1) > 2 ** 19:
                continue
            errs, floors, t0 = [], [], time.time()
            for seed in range(4):
                PG = PartialGrammar(V, M, S, L, rho, np.random.default_rng(2000 + seed))
                k = PG.fresh[L].sum(1)
                floors.append(float((k.sum() - k.max()) / (V * M)))
                X, _ = PG.sample(P, np.random.default_rng(seed))
                errs.append(evaluate(PG, pt_learn(PG, X), 2000, np.random.default_rng(99)))
            rec = dict(rho=rho, L=L, P=P, err=float(np.mean(errs)), floor=float(np.mean(floors)),
                       errs=[round(e, 3) for e in errs], secs=round(time.time() - t0))
            print(json.dumps(rec), flush=True)
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
