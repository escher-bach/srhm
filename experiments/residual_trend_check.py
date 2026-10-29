"""Is the growth of residual detail energy with scale specific to natural images?  Same
statistic as exp_selfsim_div2k_ridge.py on 1/f^2 noise (exactly scale-invariant in law, but
finite-size) and on Kodak."""
import json, os, sys
import numpy as np
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selfsim as ss

K = 6
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
rng = np.random.default_rng(0)


def stats(samples_per_image, kmax):
    out = {}
    for k in range(kmax):
        m = len(samples_per_image); perm = rng.permutation(m); A, B = perm[: m // 2], perm[m // 2:]
        Xa = np.concatenate([samples_per_image[i][k][0] for i in A]); Ya = np.concatenate([samples_per_image[i][k][2] for i in A])
        Xb = np.concatenate([samples_per_image[i][k][0] for i in B]); Yb = np.concatenate([samples_per_image[i][k][2] for i in B])
        R = Yb - Ridge(1.0).fit(Xa, Ya).predict(Xb)
        z = R.ravel()
        out[k] = dict(kurtosis=round(float(((z - z.mean()) ** 4).mean() / z.var() ** 2), 2),
                      energy=round(float((R ** 2).sum(1).mean()), 3),
                      detail_energy=round(float((Yb ** 2).sum(1).mean()), 3), n=len(Xa))
    return out


def per_image(gray, K_):
    S = ss.pyramid(gray.astype(np.float32), K_)
    res = []
    for k in range(K_):
        C, P, Y = ss.rule_samples_ctx(S[k], S[k + 1])
        if len(C) > 600:
            i = rng.choice(len(C), 600, replace=False); C, P, Y = C[i], P[i], Y[i]
        res.append((C, P, Y))
    return res


pink = [per_image(im[:1280, :1792], K) for im in ss.pink_noise_images(24, 2048, np.random.default_rng(5))]
kod = [per_image(ss.load_gray(f)[:512, :512] if ss.load_gray(f).shape[0] >= 512 else ss.load_gray(f), 5)
       for f in sorted(__import__("glob").glob("data/kodak/*.png"))]
res = {"pink_1/f2": stats(pink, 5), "kodak": stats(kod, 4)}
for name, r in res.items():
    print(name, {k: (v["kurtosis"], v["energy"], v["detail_energy"]) for k, v in r.items()})
json.dump(res, open("verify/residual_trend_check.json", "w"), indent=1)
