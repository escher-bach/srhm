"""A context for the root: views of one scene (srhm.tex Sec. views; predictions W1-W6 are fixed there before this runs).

Models, all trained with ZERO labels on the 50,000 CIFAR-10 training images (scenes):
  R   IT(16) tied, one image per scene (the old root; its state is the root belief)
  A   SG(16, 16), K = 1, no transformation            (removes the 16-way merge; still no context)
  B   SG(16, 16), K = 2 views: random crop (4 px reflect padding) + horizontal flip
  C   SG(16, 16), K = 2 views: crop + flip + colour jitter (brightness, contrast, saturation, channel gains, grey)
  D   as C with K = 4 (if the budget allows)
State of an image: the single-view posterior of the scene type, w(z) (R: the root belief).
Measured: test bits/dim of single clean images; heads on the state (post-hoc Bayes P(y|z) with P labels, logistic);
invariance TV(w(gX), w(X)) under geometric and under colour transformations, against TV between two different images;
R^2 of the image's mean colour from the state.
"""
import os, sys, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(globals().get("__file__", "./x"))))
import imtree as it
import scenetree as stree
import exp_cifar as ex

DEV, LN2, log, left = ex.DEV, ex.LN2, ex.log, ex.left
H, M, S_BATCH = 16, 16, 128
STEPS = int(os.environ.get("VIEWS_STEPS", "3000"))


# ------------------------------------------------------------------ view families (uint8 in, uint8 out)
def geo(X, gen):
    B = X.shape[0]
    Xp = torch.nn.functional.pad(X.permute(0, 3, 1, 2).float(), (4, 4, 4, 4), mode="reflect").permute(0, 2, 3, 1)
    oy = torch.randint(0, 9, (B,), device=X.device, generator=gen); ox = torch.randint(0, 9, (B,), device=X.device, generator=gen)
    ar = torch.arange(32, device=X.device)
    out = Xp[torch.arange(B, device=X.device)[:, None, None], (oy[:, None] + ar)[:, :, None], (ox[:, None] + ar)[:, None, :]]
    flip = torch.rand(B, device=X.device, generator=gen) < 0.5
    out = torch.where(flip[:, None, None, None], out.flip(2), out)
    return out.round().clamp(0, 255).to(torch.uint8)


def colour(X, gen):
    B = X.shape[0]
    u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(B, *s, device=X.device, generator=gen)
    x = X.float()
    w = torch.tensor([0.299, 0.587, 0.114], device=X.device)
    x = x * u(0.6, 1.4, 1, 1, 1)                                                    # brightness
    m = (x @ w).mean((1, 2))[:, None, None, None]
    x = (x - m) * u(0.6, 1.4, 1, 1, 1) + m                                          # contrast
    g = (x @ w)[..., None]
    x = g + (x - g) * u(0.4, 1.6, 1, 1, 1)                                          # saturation
    x = x * u(0.8, 1.2, 1, 1, 3)                                                    # colour balance
    grey = torch.rand(B, device=X.device, generator=gen) < 0.2
    x = torch.where(grey[:, None, None, None], (x @ w)[..., None].expand_as(x), x)
    return x.round().clamp(0, 255).to(torch.uint8)


FAMILIES = {"none": lambda X, g: X, "geo": geo, "geo+col": lambda X, g: colour(geo(X, g), g)}


# ------------------------------------------------------------------ learning (zero labels)
def fit_views(model, X, K, fam, steps=STEPS, S=S_BATCH, lr=3e-3, tag=""):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))
    gen = torch.Generator(device=DEV).manual_seed(1)
    run, t = None, time.time()
    for s in range(steps):
        idx = torch.randint(0, X.shape[0], (S,), device=DEV)
        V = FAMILIES[fam](X[idx].repeat_interleave(K, 0), gen)
        negQ, lp = model.q_objective(it.pyramid(ex.deq(V)), K)
        opt.zero_grad(); negQ.backward(); opt.step(); sched.step()
        v = float(-lp.mean() / (K * 3072 * LN2) + 8)
        run = v if run is None else 0.98 * run + 0.02 * v
        if s % 500 == 0 or s == steps - 1:
            log(event="train", tag=tag, step=s, train_bpd_per_view=round(run, 4), secs=round(time.time() - t, 1))
    return model


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def state(model, X, bs=500, seed=7):
    g = torch.Generator(device=DEV).manual_seed(seed)
    out = []
    for s in range(0, X.shape[0], bs):
        P = it.pyramid(ex.deq(X[s:s + bs], g))
        out.append(model.root_state(P)[0] if isinstance(model, it.ImageTree) else model.z_posterior(P))
    return torch.cat(out).clamp(min=-50)


@torch.no_grad()
def density(model, X, bs=500):
    g = torch.Generator(device=DEV).manual_seed(123)
    tot = 0.0
    for s in range(0, X.shape[0], bs):
        P = it.pyramid(ex.deq(X[s:s + bs], g))
        lp = model.logp_image(P) if isinstance(model, it.ImageTree) else model.logp_scenes(P, 1)
        tot += float(lp.double().sum())
    return round(-tot / X.shape[0] / (3072 * LN2) + 8, 4)


def tv(a, b):
    return 0.5 * (torch.exp(a) - torch.exp(b)).abs().sum(1)


@torch.no_grad()
def invariance(model, X, n=2000):
    gen = torch.Generator(device=DEV).manual_seed(9)
    s0 = state(model, X[:n])
    out = {}
    for name, fam in (("geo", geo), ("col", colour)):
        out[name] = round(float(tv(s0, state(model, fam(X[:n], gen))).mean()), 4)
    out["other_image"] = round(float(tv(s0, s0.roll(1, 0)).mean()), 4)
    return out


def colour_r2(F_tr, X_tr, F_te, X_te):
    Y_tr = X_tr.float().mean((1, 2)) / 255; Y_te = X_te.float().mean((1, 2)) / 255
    A = torch.cat([F_tr, torch.ones(F_tr.shape[0], 1, device=DEV)], 1).double()
    W = torch.linalg.lstsq(A, Y_tr.double()).solution
    P = torch.cat([F_te, torch.ones(F_te.shape[0], 1, device=DEV)], 1).double() @ W
    return round(float(1 - ((P - Y_te) ** 2).sum() / ((Y_te - Y_te.mean(0)) ** 2).sum()), 4)


def evaluate(model, tag, Xtr, ytr, Xte, yte, K, fam):
    res = dict(event="eval", tag=tag, K=K, family=fam, test_bpd=density(model, Xte))
    s_tr, s_te = state(model, Xtr), state(model, Xte)
    res["bayes_head"] = {P: round(ex.bayes_head(s_tr, ytr, s_te, yte, P), 4) for P in (100, 1000, 50000)}
    res["logistic_state"] = {P: round(ex.logistic_head(s_tr, ytr, s_te, yte, P), 4) for P in (1000, 50000)}
    g = torch.Generator(device=DEV).manual_seed(7)
    tR_tr = torch.cat([it.pyramid(ex.deq(Xtr[s:s + 1000], g))["tR"] for s in range(0, Xtr.shape[0], 1000)])
    tR_te = torch.cat([it.pyramid(ex.deq(Xte[s:s + 1000], g))["tR"] for s in range(0, Xte.shape[0], 1000)])
    res["logistic_state_thumb"] = round(ex.logistic_head(torch.cat([s_tr, tR_tr], 1), ytr,
                                                         torch.cat([s_te, tR_te], 1), yte, 50000), 4)
    res["state_entropy"] = round(float(-(torch.exp(s_te) * s_te).sum(1).mean()), 3)
    res["state_usage"] = [round(float(v), 3) for v in torch.exp(s_te).mean(0)]
    res["invariance_tv"] = invariance(model, Xte)
    res["colour_r2"] = colour_r2(s_tr[:20000], Xtr[:20000], s_te, Xte)
    log(**res)
    return res


def main():
    log(event="start", dev=DEV, budget=ex.BUDGET, steps=STEPS)
    torch.manual_seed(0)
    Xtr, ytr, Xte, yte = ex.load_cifar()
    perm = torch.randperm(Xtr.shape[0], generator=torch.Generator().manual_seed(0)).to(DEV)
    Xtr, ytr = Xtr[perm], ytr[perm]
    # R: the old root
    model = it.ImageTree(H, seed=0).to(DEV)
    ex.fit(model, Xtr, STEPS, S_BATCH, tag="R")
    evaluate(model, "R_IT16", Xtr, ytr, Xte, yte, 1, "none")
    for tag, K, fam in (("A", 1, "none"), ("B", 2, "geo"), ("C", 2, "geo+col"), ("D", 4, "geo+col")):
        if left() < (900 if K < 4 else 2400):
            log(event="budget_stop", skipped=tag); continue
        model = stree.SceneGrammar(H, M, seed=0).to(DEV)
        log(event="plan", tag=tag, K=K, family=fam, nparams=stree.nparams(model))
        fit_views(model, Xtr, K, fam, tag=tag)
        evaluate(model, f"{tag}_SG16x16_K{K}_{fam}", Xtr, ytr, Xte, yte, K, fam)
    log(event="done")


if __name__ == "__main__":
    main()
