"""Robustness of the DIV2K result: does the rule stay shared across scales when it is made
richer (parent + 3x3 neighbourhood of context) and explains more of the detail?
Same streaming, splits and sigma statistic as exp_selfsim_div2k.py; features reduced by one PCA
(32 dims) fitted on data pooled equally over scales.  Also reruns the parent-only rule on the
same samples for a like-for-like comparison.  Outputs verify/exp_selfsim_div2k_ctx.json.
"""
import io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selfsim as ss
from remotezip import open_remote_zip
from exp_selfsim_div2k import URL, K, CAP, W, H, sigma

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = [(0, 1), (1, 0), (0, 3), (3, 0), (1, 4), (4, 1), (0, 5), (5, 0), (2, 4), (4, 2)]


def per_image(gray, rng):
    if gray.shape[0] > gray.shape[1]:
        gray = gray.T
    S = ss.pyramid(gray[:H, :W].astype(np.float32), K)
    out = []
    for k in range(K):
        C, Pp, Y = ss.rule_samples_ctx(S[k], S[k + 1])
        if len(C) > CAP:
            i = rng.choice(len(C), CAP, replace=False); C, Pp, Y = C[i], Pp[i], Y[i]
        out.append((C.astype(np.float32), Pp.astype(np.float32), Y.astype(np.float32)))
    return out


def fetch(names_chunk, seed):
    z = open_remote_zip(URL)
    rng = np.random.default_rng(seed)
    res = []
    for nm in names_chunk:
        im = np.asarray(Image.open(io.BytesIO(z.read(nm))).convert("RGB"), dtype=np.float32) / 255
        res.append(per_image(im @ np.array([0.299, 0.587, 0.114], np.float32), rng))
    return res


if __name__ == "__main__":
    t0 = time.time()
    names = sorted(n for n in open_remote_zip(URL).namelist() if n.endswith(".png"))
    with ThreadPoolExecutor(4) as ex:
        div = [s for part in ex.map(fetch, [names[i::4] for i in range(4)], range(4)) for s in part]
    prng = np.random.default_rng(77)
    pink = [per_image(im, prng) for im in ss.pink_noise_images(24, 2048, np.random.default_rng(5))]
    print(f"streamed in {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    pool = np.concatenate([np.concatenate([d[k][0] for d in div])[:8000] for k in range(K)])
    pca = PCA(32, random_state=0).fit(pool)
    as_ctx = lambda L: [[(pca.transform(c).astype(np.float32), y) for c, _, y in img] for img in L]
    as_par = lambda L: [[(p_, y) for _, p_, y in img] for img in L]
    variants = {"context": (as_ctx(div), as_ctx(pink)), "parent_only": (as_par(div), as_par(pink))}
    res = {}
    for name, (D, Pk) in variants.items():
        res[name] = {}
        for k, k2 in PAIRS:
            s, sd, r2, n = sigma(D, D, k, k2, rng, 10, True)
            sp, spd, _, _ = sigma(Pk, D, k, k2, rng, 6, False)
            res[name][f"{k}->{k2}"] = dict(sigma=[round(s, 3), round(sd, 3)],
                                           pink=[round(sp, 3), round(spd, 3)], r2_own=round(r2, 3), n=n)
            print(f"{name:12s} {k}->{k2}: sigma={s:.3f}+/-{sd:.3f} pink={sp:.3f} own R2={r2:.3f}", flush=True)
    res["secs"] = round(time.time() - t0)
    with open(os.path.join(HERE, "exp_selfsim_div2k_ctx.json"), "w") as fh:
        json.dump(res, fh, indent=1)
