"""Head to head on the tied RHM: zero-label context learner (srhm.tex sections 3-4) against the
backprop runs of exp_claim2_bp.py.  Same grammars (seeds 1000, 1002), same test sets
(G.sample(4000, L, rng(99))).  The learner sees P unlabelled trees and uses no labels; it knows only
(v, m, s).  For each (grammar, L) it doubles P until all 3 data seeds give test accuracy 1.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srhm import Grammar, ctx_learn

V, M, S = 8, 4, 2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_head2head_results.jsonl")


def main():
    open(OUT, "w").close()
    for g in (0, 2):
        G = Grammar(V, M, S, np.random.default_rng(1000 + g))
        for L in (2, 3, 4, 5, 6, 7, 9, 11):
            Xt, yt = G.sample(4000, L, np.random.default_rng(99))
            for P in [2 ** k for k in range(0, 12)]:
                if P * S ** (L - 1) > 16384:            # far beyond the known threshold
                    break
                t = time.time(); accs = []
                for seed in range(3):
                    X, _ = G.sample(P, L, np.random.default_rng(seed))      # labels discarded
                    tab, _ = ctx_learn(G, X, restarts=15)
                    accs.append(float((G.parse(Xt, tab) == yt).mean()))
                rec = dict(grammar=g, L=L, P_unlabelled=P, labels=0, bottom_grids=P * S ** (L - 1),
                           acc=round(float(np.mean(accs)), 4), accs=[round(a, 4) for a in accs],
                           secs=round(time.time() - t, 1))
                print(json.dumps(rec), flush=True)
                with open(OUT, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                if min(accs) == 1.0:
                    break


if __name__ == "__main__":
    main()
