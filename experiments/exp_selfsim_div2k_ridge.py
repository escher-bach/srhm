"""DIV2K, best available rule: ridge regression on parent + 3x3 context.
(1) sigma for the same scale pairs as the context run (predictable part of the rule);
(2) distributional check of the UNPREDICTABLE part: kurtosis and energy of the residual detail at
    each scale (a tied rule has the same residual law at every scale; Gaussian kurtosis = 3).
Outputs verify/exp_selfsim_div2k_ridge.json.
"""
import io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selfsim as ss
from remotezip import open_remote_zip
from exp_selfsim_div2k import URL, K
from exp_selfsim_div2k_ctx import fetch, per_image, PAIRS

HERE = os.path.dirname(os.path.abspath(__file__))


def cat(L, idx, k):
    return (np.concatenate([L[i][k][0] for i in idx]), np.concatenate([L[i][k][2] for i in idx]))


def sigma_ridge(src, tgt, k, k2, rng, splits, same):
    vals, r2s = [], []
    for _ in range(splits):
        perm = rng.permutation(len(tgt)); A, B = perm[: len(tgt) // 2], perm[len(tgt) // 2:]
        sidx = A if same else rng.permutation(len(src))[: len(src) // 2]
        Xs, Ys = cat(src, sidx, k); Xo, Yo = cat(tgt, A, k2); Xt, Yt = cat(tgt, B, k2)
        n = min(len(Xs), len(Xo), 6000)
        i1, i2 = rng.choice(len(Xs), n, replace=False), rng.choice(len(Xo), n, replace=False)
        mu = Yo[i2].mean(0)
        sig = []
        for X, Y in ((Xs[i1], Ys[i1]), (Xo[i2], Yo[i2])):
            pred = Ridge(1.0).fit(X, Y).predict(Xt)
            sig.append(((pred - mu) * (Yt - mu)).sum())
        vals.append(sig[0] / sig[1])
        r2s.append(1 - ((Yt - pred) ** 2).sum() / ((Yt - mu) ** 2).sum())
    return float(np.mean(vals)), float(np.std(vals)), float(np.mean(r2s))


if __name__ == "__main__":
    t0 = time.time()
    names = sorted(n for n in open_remote_zip(URL).namelist() if n.endswith(".png"))
    with ThreadPoolExecutor(4) as ex:
        div = [s for part in ex.map(fetch, [names[i::4] for i in range(4)], range(4)) for s in part]
    prng = np.random.default_rng(77)
    pink = [per_image(im, prng) for im in ss.pink_noise_images(24, 2048, np.random.default_rng(5))]
    print(f"streamed in {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    res = {"sigma": {}, "residual": {}}
    for k, k2 in PAIRS:
        s, sd, r2 = sigma_ridge(div, div, k, k2, rng, 10, True)
        sp, _, _ = sigma_ridge(pink, div, k, k2, rng, 6, False)
        res["sigma"][f"{k}->{k2}"] = dict(sigma=[round(s, 3), round(sd, 3)], pink=round(sp, 3), r2_own=round(r2, 3))
        print(f"ridge-context {k}->{k2}: sigma={s:.3f}+/-{sd:.3f} pink={sp:.3f} own R2={r2:.3f}", flush=True)
    for k in range(K):                      # residual law per scale (fit on half, residuals on the rest)
        perm = rng.permutation(len(div)); A, B = perm[:50], perm[50:]
        Xa, Ya = cat(div, A, k); Xb, Yb = cat(div, B, k)
        R = Yb - Ridge(1.0).fit(Xa, Ya).predict(Xb)
        z = R.ravel()
        kurt = float(((z - z.mean()) ** 4).mean() / z.var() ** 2)
        res["residual"][str(k)] = dict(kurtosis=round(kurt, 2), energy=round(float((R ** 2).sum(1).mean()), 4),
                                       frac_unexplained=round(float((R ** 2).sum() / ((Yb - Ya.mean(0)) ** 2).sum()), 3))
        print(f"scale {k}: residual kurtosis={kurt:.2f} energy={res['residual'][str(k)]['energy']}", flush=True)
    res["secs"] = round(time.time() - t0)
    with open(os.path.join(HERE, "exp_selfsim_div2k_ridge.json"), "w") as fh:
        json.dump(res, fh, indent=1)
