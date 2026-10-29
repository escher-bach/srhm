"""Control for the Kodak measurement: is high cross-scale transfer specific to the SAME image
source, or does any smooth-image rule transfer?  sigma_cross = signal captured on Kodak scale k'
by a rule learned on ANOTHER source (1/f^2 noise, tied cascade) at scale k, relative to Kodak's
own scale-k' rule (same training size, held-out Kodak images).
"""
import json, os, sys
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selfsim as ss

K = 5
rng = np.random.default_rng(0)
kod = ss.kodak(K)
sources = {"kodak (other images)": kod,
           "pink noise 1/f^2": [ss.pyramid(im, K) for im in ss.pink_noise_images(24, 512, np.random.default_rng(5))],
           "tied cascade": [ss.pyramid(im, K) for im in ss.cascade_images(24, 7, 1.0, np.random.default_rng(10))]}


def samples(pyrs, idx, k):
    return [np.concatenate(z) for z in zip(*[ss.rule_samples(pyrs[i][k], pyrs[i][k + 1]) for i in idx])]


out = {}
for k, k2 in [(0, 1), (1, 2), (0, 2), (1, 3)]:
    for name, src in sources.items():
        vals = []
        for _ in range(10):
            perm = rng.permutation(24); A, B = perm[:12], perm[12:]
            Xs, Ys = samples(src, A if name.startswith("kodak") else rng.permutation(24)[:12], k)
            Xo, Yo = samples(kod, A, k2)                     # Kodak's own rule at k2
            Xt, Yt = samples(kod, B, k2)                     # held-out Kodak targets
            n = min(len(Xs), len(Xo), 4000)
            i1, i2 = rng.choice(len(Xs), n, replace=False), rng.choice(len(Xo), n, replace=False)
            mu = Yo[i2].mean(0)
            sig = []
            for X, Y in ((Xs[i1], Ys[i1]), (Xo[i2], Yo[i2])):
                pred = KNeighborsRegressor(n_neighbors=32).fit(X, Y).predict(Xt)
                sig.append(((pred - mu) * (Yt - mu)).sum())
            vals.append(sig[0] / sig[1])
        out[f"{name} | {k}->{k2}"] = (round(float(np.mean(vals)), 3), round(float(np.std(vals)), 3))
        print(f"rule from {name:22s} scale {k} -> Kodak scale {k2}: sigma = {np.mean(vals):.3f} +/- {np.std(vals):.3f}", flush=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_selfsim_control.json"), "w") as fh:
    json.dump(out, fh, indent=1)
