"""CIFAR-10 joint experiment for srhm.tex section 10 (predictions I1-I6 are fixed there before this runs).

One zero-label model (imtree.ImageTree: tied structured kernel on the thumbnail pyramid, H hidden
context types, exact sum-product, generalized EM = Fisher-identity gradient of grid-local statistics)
is evaluated three ways:
  density     test bits/dim, total and per height (truncated pyramids);
  readout     linear heads on the ROOT state only: (i) the Bayes head P(y|h) learned by EM as a label leaf,
              (ii) multinomial logistic regression on (log root belief, root thumbnail); P labels;
  generation  ancestral samples: per-height Haar energy and kurtosis vs the test set; a PNG grid.

  usage:  exp_cifar.py main     -- N = 50000, H in (1, 16, 128) tied, H = 16 untied
          exp_cifar.py sweep    -- claim 1: H = 16, tied vs untied, N = 64 ... 50000 unlabelled images
"""
import os, sys, io, json, time, math, base64
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(globals().get("__file__", "./x"))))
import imtree as it

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BUDGET = float(os.environ.get("BUDGET_SECONDS", "1e9")); T0 = time.time()
LN2 = math.log(2.0)


def log(**k):
    k["t"] = round(time.time() - T0, 1); print(json.dumps(k), flush=True)


def left():
    return BUDGET - (time.time() - T0)


# ------------------------------------------------------------------ data
def load_cifar():
    try:
        import pandas as pd
        from PIL import Image
        base = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
        out = []
        for split in ("train", "test"):
            df = pd.read_parquet(base + f"{split}-00000-of-00001.parquet")
            X = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in df["img"]])
            out.append((X, df["label"].to_numpy()))
        src = "hf"
    except Exception as e:                                           # fall back to the Toronto mirror
        log(event="hf_failed", err=str(e)[:200])
        import torchvision
        out = []
        for tr in (True, False):
            d = torchvision.datasets.CIFAR10("/content/cifar", train=tr, download=True)
            out.append((d.data, np.array(d.targets)))
        src = "torchvision"
    (Xtr, ytr), (Xte, yte) = out
    log(event="data", src=src, n_train=len(ytr), n_test=len(yte))
    t = lambda a: torch.tensor(a, device=DEV)
    return t(Xtr), t(ytr), t(Xte), t(yte)


def deq(Xu8, gen=None):
    return (Xu8.float() + torch.rand(Xu8.shape, device=Xu8.device, generator=gen)) / 256.0


# ------------------------------------------------------------------ learning (zero labels)
def fit(model, X, steps, batch, lr=3e-3, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))
    N = X.shape[0]
    run, t = None, time.time()
    for s in range(steps):
        idx = torch.randint(0, N, (batch,), device=DEV) if N > batch else torch.arange(N, device=DEV)
        negQ, lp = model.q_objective(it.pyramid(deq(X[idx])))
        opt.zero_grad(); negQ.backward(); opt.step(); sched.step()
        bpd = float(-lp.mean() / (3072 * LN2) + 8)
        run = bpd if run is None else 0.98 * run + 0.02 * bpd
        if s % 500 == 0 or s == steps - 1:
            log(event="train", tag=tag, step=s, train_bpd=round(run, 4), secs=round(time.time() - t, 1))
    return model


def timed_steps(model, X, batch, alloc, cap):
    """Steps that fit in `alloc` seconds (measured on 10 steps), at most cap."""
    m2 = it.ImageTree(model.H, untied=model.untied, hid=model.hid, seed=99).to(DEV)
    opt = torch.optim.Adam(m2.parameters(), lr=1e-4)
    for k in range(12):
        if k == 2:
            torch.cuda.synchronize() if DEV == "cuda" else None; t = time.time()
        idx = torch.randint(0, X.shape[0], (min(batch, X.shape[0]),), device=DEV)
        negQ, _ = m2.q_objective(it.pyramid(deq(X[idx])))
        opt.zero_grad(); negQ.backward(); opt.step()
    torch.cuda.synchronize() if DEV == "cuda" else None
    tps = (time.time() - t) / 10
    return int(min(cap, max(200, alloc / tps))), tps


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def eval_density(model, X, bs=500):
    g = torch.Generator(device=DEV).manual_seed(123)
    L = model.L
    tot = torch.zeros(L + 1, dtype=torch.float64, device=DEV)
    for s in range(0, X.shape[0], bs):
        P = it.pyramid(deq(X[s:s + bs], g))
        lls = {c: model.emis(c, P["tp"][c], P["det"][c]) for c in range(L)}
        B = P["tR"].shape[0]
        for c in range(L):                                          # Jacobian constant is PER IMAGE
            tot[c] += model.upward(P, lls, cmin=c)[3].double().sum() - B * sum(
                P["det"][k].shape[1] for k in range(c, L)) * 3 * LN2
        tot[L] += torch.logsumexp(model.logpi() + model.root_emis(P["tR"]), -1).double().sum()
    lp = (tot / X.shape[0]).cpu().numpy()                          # mean log p(levels >= c), c = 0..L
    n = [4 ** (L - c) for c in range(L)]
    per = {f"c{c}": round(float(-(lp[c] - lp[c + 1]) / (9 * n[c] * LN2)), 4) for c in range(L)}
    per["root"] = round(float(-lp[L] / (12 * LN2)), 4)
    return round(float(-lp[0] / (3072 * LN2) + 8), 4), per, float(lp[0])


@torch.no_grad()
def root_features(model, X, bs=1000):
    g = torch.Generator(device=DEV).manual_seed(7)
    lq, tr = [], []
    for s in range(0, X.shape[0], bs):
        a, b = model.root_state(it.pyramid(deq(X[s:s + bs], g)))
        lq.append(a.clamp(min=-50)); tr.append(b)
    return torch.cat(lq), torch.cat(tr)


def bayes_head(lq_tr, ytr, lq_te, yte, P, iters=200):
    """Label leaf of the root: P(y|h) by EM on P labelled images (root model fixed).  Prediction
    P(y|X) = sum_h beta(h) P(y|h): linear in the root belief (Prop. readout)."""
    q = torch.exp(lq_tr[:P]); y = ytr[:P]
    H = q.shape[1]
    Py = torch.full((H, 10), 0.1, device=DEV)
    Y = torch.nn.functional.one_hot(y, 10).float()
    for _ in range(iters):
        w = q * Py[:, y].T
        w = w / w.sum(1, keepdim=True).clamp(min=1e-30)
        Py = (w.T @ Y + 1e-3) / (w.sum(0)[:, None] + 1e-2)
    pred = (torch.exp(lq_te) @ Py).argmax(1)
    return float((pred == yte).float().mean())


def logistic_head(F_tr, ytr, F_te, yte, P, steps=600, wd=1e-4):
    mu, sd = F_tr[:P].mean(0), F_tr[:P].std(0) + 1e-6
    A, B = (F_tr[:P] - mu) / sd, (F_te - mu) / sd
    lin = torch.nn.Linear(A.shape[1], 10).to(DEV)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(lin(A), ytr[:P]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float((lin(B).argmax(1) == yte).float().mean())


def detail_stats(dets, L):
    out = {}
    for c in range(L):
        d = dets[c].reshape(-1, 9).double()
        k = ((d - d.mean(0)) ** 4).mean(0) / d.var(0) ** 2
        out[f"c{c}"] = dict(energy=round(float((d ** 2).sum(1).mean()), 6), kurt=round(float(k.mean()), 2))
    return out


def png_grid(x, nrow=10):
    from PIL import Image
    x = (x.clamp(0, 1) * 255).round().byte().cpu().numpy()
    B = x.shape[0]; ncol = int(math.ceil(B / nrow))
    G = np.full((ncol * 34, nrow * 34, 3), 255, np.uint8)
    for k in range(B):
        r, c = divmod(k, nrow)
        G[r * 34 + 1: r * 34 + 33, c * 34 + 1: c * 34 + 33] = x[k]
    buf = io.BytesIO(); Image.fromarray(G).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def evaluate(model, Xte, yte, Xtr, ytr, tag, full=True):
    bpd, per, lp = eval_density(model, Xte)
    res = dict(event="eval", tag=tag, H=model.H, untied=model.untied, nparams=it.nparams(model),
               test_bpd=bpd, per_height_bits=per, test_logp=round(lp, 2))
    lq_tr, t_tr = root_features(model, Xtr); lq_te, t_te = root_features(model, Xte)
    F_tr, F_te = torch.cat([lq_tr, t_tr], 1), torch.cat([lq_te, t_te], 1)
    Ps = (100, 1000, 10000, 50000) if full else (50000,)
    res["bayes_head"] = {P: round(bayes_head(lq_tr, ytr, lq_te, yte, P), 4) for P in Ps}
    res["logistic_head"] = {P: round(logistic_head(F_tr, ytr, F_te, yte, P), 4) for P in Ps}
    res["root_belief_entropy"] = round(float(-(torch.exp(lq_te) * lq_te).sum(1).mean()), 3)
    if full:
        g = torch.Generator(device=DEV).manual_seed(5)
        dets_s = {c: [] for c in range(model.L)}
        for _ in range(10):
            x, d, _ = model.sample(1000, gen=g)
            for c in range(model.L):
                dets_s[c].append(d[c])
        res["sample_stats"] = detail_stats({c: torch.cat(v) for c, v in dets_s.items()}, model.L)
        res["data_stats"] = detail_stats(it.pyramid(deq(Xte))["det"], model.L)
        res["png"] = png_grid(model.sample(100, gen=torch.Generator(device=DEV).manual_seed(11))[0])
    log(**res)
    return res


# ------------------------------------------------------------------ drivers
def main(mode):
    log(event="start", mode=mode, dev=DEV, budget=BUDGET)
    torch.manual_seed(0)
    Xtr, ytr, Xte, yte = load_cifar()
    perm = torch.randperm(Xtr.shape[0], generator=torch.Generator().manual_seed(0)).to(DEV)
    Xtr, ytr = Xtr[perm], ytr[perm]                                   # fixed random order: label subsets = prefixes
    if mode == "main":
        log(event="real_data_png", png=png_grid(deq(Xte[:100]).float()))
        configs = [(1, False), (16, False), (128, False), (16, True)]
        for k, (H, untied) in enumerate(configs):
            model = it.ImageTree(H, untied=untied, seed=0).to(DEV)
            alloc = (left() - 600) / (len(configs) - k) * 0.75
            steps, tps = timed_steps(model, Xtr, 128, alloc, 6000)
            log(event="plan", H=H, untied=untied, steps=steps, secs_per_step=round(tps, 4), nparams=it.nparams(model))
            fit(model, Xtr, steps, 128, tag=f"H{H}{'u' if untied else ''}")
            evaluate(model, Xte, yte, Xtr, ytr, tag=f"H{H}{'u' if untied else ''}_N50000")
    elif mode == "sweep":
        H = 16
        runs = [(N, s, u) for N in (64, 256, 1024, 4096, 16384, 50000) for s in ((0, 1) if N <= 1024 else (0,))
                for u in (False, True)]
        for k, (N, seed, untied) in enumerate(runs):
            if left() < 300:
                log(event="budget_stop", remaining=len(runs) - k); break
            sub = Xtr[seed * 1000: seed * 1000 + N] if N <= 1024 else Xtr[:N]
            model = it.ImageTree(H, untied=untied, seed=seed).to(DEV)
            steps = 3000
            fit(model, sub, steps, min(N, 128), tag=f"N{N}s{seed}{'u' if untied else ''}")
            r = evaluate(model, Xte, yte, Xtr, ytr, tag=f"sweep_N{N}_s{seed}{'u' if untied else ''}", full=False)
    log(event="done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
