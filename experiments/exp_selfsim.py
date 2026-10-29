"""Stage 1: measure scale self-similarity (rho) of images.
  python verify/exp_selfsim.py calib    # synthetic images with known rho (+ 1/f^2 noise)
  python verify/exp_selfsim.py kodak    # Kodak 24 natural images
Scale pairs are named by the finer scale k of the rule (k -> k+1); k = 0 is full resolution.
"""
import json, os, sys, time
import numpy as np
import selfsim as ss

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_selfsim_results.jsonl")
K, V = 5, 16
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (0, 3), (1, 3)]


def probe(name, imgs_or_pyrs, is_pyr=False, seed=0):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    pyrs = imgs_or_pyrs if is_pyr else [ss.pyramid(im, K) for im in imgs_or_pyrs]
    km = ss.fit_codebook(pyrs, V, rng)
    tabs = ss.image_tables(pyrs, km)
    flat = [float(np.mean([t[k][:, 1:] == 0 for t in tabs])) for k in range(K)]
    rec = dict(data=name, v=V, n_images=len(pyrs), flat_frac=[round(f, 3) for f in flat],
               MI_bits=[round(ss.mutual_info(tabs, k, V), 3) for k in range(K)])
    for k, k2 in PAIRS:
        (r, acc, wit), (rs, _, _) = ss.rho_hat(tabs, k, k2, V, rng)
        rec[f"rhoTV_{k}_{k2}"] = [round(r, 3), round(rs, 3)]
        if k2 <= 3:
            (sig, tau, r2x, r2own, n), (sigs, _, _, _, _) = ss.transfer(pyrs, k, k2, rng)
            rec[f"sigma_{k}_{k2}"] = [round(sig, 3), round(sigs, 3)]
            rec[f"tau_{k}_{k2}"] = [round(tau, 3), round(r2x, 3), round(r2own, 3), int(n)]
    rec["secs"] = round(time.time() - t0)
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "calib":
        for rho in [1.0, 0.8, 0.5, 0.0]:
            imgs = ss.cascade_images(24, 7, rho, np.random.default_rng(int(rho * 10)))
            probe(f"cascade_rho{rho}", imgs)
        probe("pink_noise_1/f2", ss.pink_noise_images(24, 512, np.random.default_rng(5)))
    elif sys.argv[1] == "kodak":
        probe("kodak24", ss.kodak(K), is_pyr=True)
