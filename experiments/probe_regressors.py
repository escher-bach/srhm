"""Which regressor captures the most of the detail rule?  Own-scale R^2 on held-out Kodak
images, for parent-only and parent+context inputs."""
import os, sys, time
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selfsim as ss

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
pyrs = ss.kodak(5)
rng = np.random.default_rng(0)
A, B = rng.permutation(24)[:12], rng.permutation(24)[12:]
for k in [0, 2]:
    cat = lambda idx: [np.concatenate(z) for z in zip(*[ss.rule_samples_ctx(pyrs[i][k], pyrs[i][k + 1]) for i in idx])]
    Ca, Pa, Ya = cat(A); Cb, Pb, Yb = cat(B)
    n = min(len(Ca), 6000); i = rng.choice(len(Ca), n, replace=False)
    Ca, Pa, Ya = Ca[i], Pa[i], Ya[i]
    mu = Ya.mean(0); null = ((Yb - mu) ** 2).sum()
    r2 = lambda pred: 1 - ((Yb - pred) ** 2).sum() / null
    models = {"kNN parent": (KNeighborsRegressor(32), Pa, Pb),
              "ridge parent": (Ridge(1.0), Pa, Pb),
              "ridge context": (Ridge(1.0), Ca, Cb),
              "MLP parent": (MLPRegressor((256,), early_stopping=True, max_iter=300, random_state=0), Pa, Pb),
              "MLP context": (MLPRegressor((256,), early_stopping=True, max_iter=300, random_state=0), Ca, Cb)}
    for name, (m, Xtr, Xte) in models.items():
        t = time.time(); m.fit(Xtr, Ya)
        print(f"scale {k}->{k+1} n={n}: {name:14s} own R2 = {r2(m.predict(Xte)):.3f}  ({time.time()-t:.1f}s)", flush=True)
