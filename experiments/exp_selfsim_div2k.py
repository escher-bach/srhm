"""Stage 1 on DIV2K (100 lossless 2K images), streamed from the remote zip (no local copy).

For every ordered pair of scales (k -> k2), k, k2 in 0..5:
  sigma      = signal on held-out DIV2K targets at k2 captured by the rule learned at k,
               relative to the rule learned at k2 (same training size; image-split cross-fitting)
  sigma_pink = same, with the rule learned on 1/f^2 Gaussian noise at scale k (generic baseline)
Outputs verify/exp_selfsim_div2k.json.
"""
import io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selfsim as ss
from remotezip import open_remote_zip

URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
K, CAP, W, H = 6, 600, 1792, 1280           # 6 rule scales; crop to a multiple of 4 * 2^6
HERE = os.path.dirname(os.path.abspath(__file__))


def per_image_samples(gray, rng):
    if gray.shape[0] > gray.shape[1]:
        gray = gray.T
    S = ss.pyramid(gray[:H, :W].astype(np.float32), K)
    out = []
    for k in range(K):
        X, Y = ss.rule_samples(S[k], S[k + 1])
        if len(X) > CAP:
            i = rng.choice(len(X), CAP, replace=False); X, Y = X[i], Y[i]
        out.append((X.astype(np.float32), Y.astype(np.float32)))
    return out


def fetch(names_chunk, seed):
    z = open_remote_zip(URL)
    rng = np.random.default_rng(seed)
    res = []
    for nm in names_chunk:
        im = np.asarray(Image.open(io.BytesIO(z.read(nm))).convert("RGB"), dtype=np.float32) / 255
        res.append(per_image_samples(im @ np.array([0.299, 0.587, 0.114], np.float32), rng))
        print(f"  {nm} done", flush=True)
    return res


def sigma(src, tgt, k, k2, rng, splits, same_source):
    """src/tgt: lists (per image) of per-scale (X, Y)."""
    vals, r2own = [], []
    for _ in range(splits):
        perm = rng.permutation(len(tgt)); A, B = perm[: len(tgt) // 2], perm[len(tgt) // 2:]
        sidx = A if same_source else rng.permutation(len(src))[: len(src) // 2]
        cat = lambda L, idx, kk: (np.concatenate([L[i][kk][0] for i in idx]),
                                  np.concatenate([L[i][kk][1] for i in idx]))
        Xs, Ys = cat(src, sidx, k); Xo, Yo = cat(tgt, A, k2); Xt, Yt = cat(tgt, B, k2)
        n = min(len(Xs), len(Xo), 6000)
        i1, i2 = rng.choice(len(Xs), n, replace=False), rng.choice(len(Xo), n, replace=False)
        mu = Yo[i2].mean(0)
        sig = []
        for X, Y in ((Xs[i1], Ys[i1]), (Xo[i2], Yo[i2])):
            pred = KNeighborsRegressor(n_neighbors=32).fit(X, Y).predict(Xt)
            sig.append(((pred - mu) * (Yt - mu)).sum())
        vals.append(sig[0] / sig[1])
        r2own.append(1 - ((Yt - pred) ** 2).sum() / ((Yt - mu) ** 2).sum())
    return float(np.mean(vals)), float(np.std(vals)), float(np.mean(r2own)), int(n)


if __name__ == "__main__":
    t0 = time.time()
    names = sorted(n for n in open_remote_zip(URL).namelist() if n.endswith(".png"))
    chunks = [names[i::4] for i in range(4)]
    with ThreadPoolExecutor(4) as ex:
        div = [s for part in ex.map(fetch, chunks, range(4)) for s in part]
    print(f"streamed {len(div)} images in {time.time() - t0:.0f}s", flush=True)
    prng = np.random.default_rng(77)
    pink = [per_image_samples(im, prng) for im in ss.pink_noise_images(24, 2048, np.random.default_rng(5))]
    counts = [int(sum(len(d[k][0]) for d in div)) for k in range(K)]
    rng = np.random.default_rng(0)
    res = dict(n_images=len(div), samples_per_scale=counts, sigma={}, sigma_pink={}, r2_own={})
    for k in range(K):
        for k2 in range(K):
            if k == k2:
                continue
            s, sd, r2, n = sigma(div, div, k, k2, rng, 20, True)
            sp, spd, _, _ = sigma(pink, div, k, k2, rng, 10, False)
            res["sigma"][f"{k}->{k2}"] = [round(s, 3), round(sd, 3), n]
            res["sigma_pink"][f"{k}->{k2}"] = [round(sp, 3), round(spd, 3)]
            res["r2_own"][f"{k2}"] = round(r2, 3)
            print(f"{k}->{k2}: sigma={s:.3f}+/-{sd:.3f}  pink={sp:.3f}+/-{spd:.3f}  (n={n})", flush=True)
    res["secs"] = round(time.time() - t0)
    with open(os.path.join(HERE, "exp_selfsim_div2k.json"), "w") as fh:
        json.dump(res, fh, indent=1)
