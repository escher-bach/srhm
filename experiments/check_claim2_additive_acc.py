"""Best accuracy reachable with additive (one-hot, degree-1) features of the leaves: a least-squares
linear classifier fitted on a large sample, held-out accuracy.  Compared with the accuracy the CNN
reaches before its jump to full accuracy (exp_claim2_bp_results.jsonl)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srhm import Grammar

V = 8
for g in (0, 2):
    G = Grammar(V, 4, 2, np.random.default_rng(1000 + g))
    for L in range(2, 8):
        P = 120_000 if L <= 6 else 60_000
        X, y = G.sample(P, L, np.random.default_rng(7))
        F = np.zeros((P, X.shape[1] * V + 1), np.float32); F[:, -1] = 1
        F[np.arange(P)[:, None], np.arange(X.shape[1]) * V + X] = 1.0
        h = P * 3 // 4
        B, *_ = np.linalg.lstsq(F[:h], np.eye(V, dtype=np.float32)[y[:h]], rcond=None)
        acc = float(((F[h:] @ B).argmax(1) == y[h:]).mean())
        print(json.dumps(dict(grammar=g, L=L, additive_acc=round(acc, 4))), flush=True)
