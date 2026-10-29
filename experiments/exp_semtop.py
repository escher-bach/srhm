"""The semantic top on CIFAR-10 (srhm.tex Sec. semtop; predictions S1-S5 are fixed there before `run`).

  usage:  exp_semtop.py cal   -- calibration, run BEFORE the predictions: a zero-label tied IT(16); Birkhoff
                                 contractions kappa of its kernel, label-kernel spreads lambda_y, posterior
                                 certainty eps per height; and the class information in the state grids G_c
                                 (thumbnails + messages from below, Cor. reach item 2), c = 1, 2, 3, and root.
          exp_semtop.py run   -- ST(16, c*) for c* in (5 = tied, 4, 3) and P in (0, 1000, 50000) labelled images
                                 among N = 50000, trained by generalized EM on the joint likelihood.  Density,
                                 Bayes head, logistic head, measured label reach per height against Thm. reach.
"""
import os, sys, math, time, copy
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(globals().get("__file__", "./x"))))
import imtree as it
import exp_cifar as ex

DEV, LN2, log, left = ex.DEV, ex.LN2, ex.log, ex.left
H, STEPS, BATCH = 16, int(os.environ.get("SEMTOP_STEPS", "3000")), 128


def fit_joint(model, X, y, P, steps=STEPS, batch=BATCH, lr=3e-3, tag=""):
    """Generalized EM on sum_n log p(X_n) + sum_{n<P} log p(y_n | X_n) (the joint likelihood with the label leaf
    observed on the first P images, summed out on the others).  Uniform minibatches over all N images."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))
    N = X.shape[0]
    run, t = None, time.time()
    for s in range(steps):
        idx = torch.randint(0, N, (batch,), device=DEV)
        yb = torch.where(idx < P, y[idx], torch.full_like(idx, -1)) if model.C > 0 else None
        negQ, lp = model.q_objective(it.pyramid(ex.deq(X[idx])), y=yb)
        opt.zero_grad(); negQ.backward(); opt.step(); sched.step()
        v = float(-lp.mean() / (3072 * LN2) + 8)
        run = v if run is None else 0.98 * run + 0.02 * v
        if s % 500 == 0 or s == steps - 1:
            log(event="train", tag=tag, step=s, train_joint_bpd=round(run, 4), secs=round(time.time() - t, 1))
    return model


# ------------------------------------------------------------------ calibration helpers
def mlp_head(F_tr, y_tr, F_te, y_te, hidden=512, epochs=int(os.environ.get("SEMTOP_EPOCHS", "40")), wd=5e-4,
             lin=False, nval=5000):
    """Classifier on fixed features; model selection by the best epoch on a held-out part of the training set."""
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-6
    A, B = (F_tr - mu) / sd, (F_te - mu) / sd
    nval = min(nval, A.shape[0] // 10)
    Av, yv, At, yt = A[:nval], y_tr[:nval], A[nval:], y_tr[nval:]
    torch.manual_seed(0)
    net = (torch.nn.Linear(A.shape[1], 10) if lin else torch.nn.Sequential(
        torch.nn.Linear(A.shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(0.3),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(hidden, 10))).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    best, best_te = -1, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(At.shape[0], device=DEV)
        for k in range(0, At.shape[0], 256):
            b = perm[k:k + 256]
            loss = torch.nn.functional.cross_entropy(net(At[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va = float((net(Av).argmax(1) == yv).float().mean())
            if va > best:
                best, best_te = va, float((net(B).argmax(1) == y_te).float().mean())
    return round(best_te, 4)


@torch.no_grad()
def grids(model, X, c, bs=1000):
    g = torch.Generator(device=DEV).manual_seed(7)
    th, ms = [], []
    for s in range(0, X.shape[0], bs):
        a, b = model.state_grid(it.pyramid(ex.deq(X[s:s + bs], g)), c)
        th.append(a.reshape(a.shape[0], -1)); ms.append(b.clamp(min=-30).reshape(b.shape[0], -1))
    return torch.cat(th), torch.cat(ms)


@torch.no_grad()
def certainty(model, X, bs=500):
    """Mean eps_v = 1 - max_h q_v(h) per height (posterior without label), and root."""
    P = it.pyramid(ex.deq(X[:bs], torch.Generator(device=DEV).manual_seed(3)))
    lls = {c: model.emis(c, P["tp"][c], P["det"][c]) for c in range(model.L)}
    q, qR, _, _ = model.posteriors(P, lls)
    out = {f"c{c}": round(float((1 - q[c].max(-1).values).mean()), 4) for c in range(model.L)}
    out["root"] = round(float((1 - qR.max(-1).values).mean()), 4)
    return out


# ------------------------------------------------------------------ label reach (Thm. reach)
@torch.no_grad()
def reach(model, X, y, bs=500):
    """Measured TV(q^y_v, q_v) and d_H per height (true labels), and the bound of Thm. reach node by node.
    Float64 copy of the model, so that small influences at deep nodes are not rounding noise."""
    model = copy.deepcopy(model).double()
    P = it.pyramid(ex.deq(X[:bs], torch.Generator(device=DEV).manual_seed(4)).double())
    yb = y[:bs]
    lls = {c: model.emis(c, P["tp"][c], P["det"][c]) for c in range(model.L)}
    q0, qR0, _, _ = model.posteriors(P, lls)
    q1, qR1, _, _ = model.posteriors(P, lls, y=yb)
    ll = model.logl()                                                   # (H, C)
    lam = (ll.max(0).values - ll.min(0).values)[yb]                     # (B,)
    kap = model.kappa()                                                 # (nl, 4)
    def dH(p, r):
        d = torch.log(p.clamp(min=1e-300)) - torch.log(r.clamp(min=1e-300))
        return d.max(-1).values - d.min(-1).values
    out = {}
    tvR = 0.5 * (qR1 - qR0).abs().sum(-1)
    out["root"] = dict(tv=round(float(tvR.mean()), 4), dH=round(float(dH(qR1, qR0).mean()), 3),
                       lam=round(float(lam.mean()), 3))
    for c in range(model.L - 1, -1, -1):
        n = q0[c].shape[1]
        k = torch.arange(n, device=DEV)
        prod = torch.ones(n, device=DEV, dtype=torch.float64); idx = k.clone()
        for cc in range(c, model.L):
            prod = prod * kap[model.lvl(cc)][idx % 4]; idx = idx // 4
        D = lam[:, None] * prod[None, :]                                # (B, n)
        eps = 1 - q0[c].max(-1).values
        bound = torch.minimum(torch.tanh(D / 4), eps * torch.exp(D.clamp(max=50)) / (1 - eps).clamp(min=1e-12))
        tv = 0.5 * (q1[c] - q0[c]).abs().sum(-1)
        out[f"c{c}"] = dict(tv=float(f"{float(tv.mean()):.3g}"), tv_max=float(f"{float(tv.max()):.3g}"),
                            dH=float(f"{float(dH(q1[c], q0[c]).mean()):.3g}"),
                            bound=float(f"{float(bound.mean()):.3g}"), D=float(f"{float(D.mean()):.3g}"),
                            viol=int((tv > bound + 1e-6).sum()))
    return out


# ------------------------------------------------------------------ evaluation of a (joint) model
@torch.no_grad()
def joint_bayes_head(model, lq_te, yte):
    return round(float(((torch.exp(lq_te) @ torch.exp(model.logl())).argmax(1) == yte).float().mean()), 4)


def evaluate(model, Xte, yte, Xtr, ytr, tag, P):
    bpd, per, lp = ex.eval_density(model, Xte)
    res = dict(event="eval", tag=tag, cstar=model.cstar, P=P, nparams=it.nparams(model), test_bpd=bpd,
               per_height_bits=per)
    lq_tr, t_tr = ex.root_features(model, Xtr); lq_te, t_te = ex.root_features(model, Xte)
    F_tr, F_te = torch.cat([lq_tr, t_tr], 1), torch.cat([lq_te, t_te], 1)
    if P > 0:
        res["bayes_head_joint"] = joint_bayes_head(model, lq_te, yte)
    res["bayes_head_posthoc"] = {Q: round(ex.bayes_head(lq_tr, ytr, lq_te, yte, Q), 4) for Q in (1000, 50000)}
    res["logistic_head"] = {Q: round(ex.logistic_head(F_tr, ytr, F_te, yte, Q), 4) for Q in (1000, 50000)}
    res["root_belief_entropy"] = round(float(-(torch.exp(lq_te) * lq_te).sum(1).mean()), 3)
    res["kappa"] = [[round(float(v), 4) for v in row] for row in model.kappa()]
    res["certainty_eps"] = certainty(model, Xte)
    if P > 0:
        res["reach"] = reach(model, Xte, yte)
    log(**res)
    return res


def main(mode):
    log(event="start", mode=mode, dev=DEV, budget=ex.BUDGET)
    torch.manual_seed(0)
    Xtr, ytr, Xte, yte = ex.load_cifar()
    perm = torch.randperm(Xtr.shape[0], generator=torch.Generator().manual_seed(0)).to(DEV)
    Xtr, ytr = Xtr[perm], ytr[perm]                                   # fixed random order: labelled = prefix
    if mode == "cal":
        model = it.ImageTree(H, seed=0).to(DEV)
        fit_joint(model, Xtr, ytr, 0, tag="cal_H16_tied")
        bpd, per, _ = ex.eval_density(model, Xte)
        lq_tr, t_tr = ex.root_features(model, Xtr); lq_te, t_te = ex.root_features(model, Xte)
        # label kernel fitted on the frozen root (as in Sec. cifar-res), for lambda_y
        q = torch.exp(lq_tr); Y = torch.nn.functional.one_hot(ytr, 10).float()
        Py = torch.full((H, 10), 0.1, device=DEV)
        for _ in range(200):
            w = q * Py[:, ytr].T; w = w / w.sum(1, keepdim=True).clamp(min=1e-30)
            Py = (w.T @ Y + 1e-3) / (w.sum(0)[:, None] + 1e-2)
        lPy = torch.log(Py / Py.sum(1, keepdim=True))
        lam = (lPy.max(0).values - lPy.min(0).values)
        log(event="cal_model", test_bpd=bpd, per_height_bits=per, kappa=[round(float(v), 4) for v in model.kappa()[0]],
            lambda_y=[round(float(v), 2) for v in lam], certainty_eps=certainty(model, Xte),
            root_entropy=round(float(-(torch.exp(lq_te) * lq_te).sum(1).mean()), 3))
        R_tr, R_te = torch.cat([lq_tr, t_tr], 1), torch.cat([lq_te, t_te], 1)
        log(event="grid", c="root", n_nodes=1,
            thumb=dict(lin=mlp_head(t_tr, ytr, t_te, yte, lin=True), mlp=mlp_head(t_tr, ytr, t_te, yte)),
            state=dict(lin=mlp_head(R_tr, ytr, R_te, yte, lin=True), mlp=mlp_head(R_tr, ytr, R_te, yte)))
        for c in (3, 2, 1):
            th_tr, ms_tr = grids(model, Xtr, c); th_te, ms_te = grids(model, Xte, c)
            S_tr, S_te = torch.cat([th_tr, ms_tr], 1), torch.cat([th_te, ms_te], 1)
            log(event="grid", c=c, n_nodes=4 ** (model.L - c),
                thumb=dict(lin=mlp_head(th_tr, ytr, th_te, yte, lin=True), mlp=mlp_head(th_tr, ytr, th_te, yte)),
                msgs=dict(lin=mlp_head(ms_tr, ytr, ms_te, yte, lin=True), mlp=mlp_head(ms_tr, ytr, ms_te, yte)),
                state=dict(lin=mlp_head(S_tr, ytr, S_te, yte, lin=True), mlp=mlp_head(S_tr, ytr, S_te, yte)))
            del th_tr, ms_tr, S_tr
    elif mode == "run":
        runs = [(cs, P) for cs in (5, 4, 3) for P in (0, 1000, 50000)]
        for k, (cs, P) in enumerate(runs):
            if left() < 400:
                log(event="budget_stop", remaining=len(runs) - k); break
            model = it.ImageTree(H, seed=0, cstar=cs, C=10).to(DEV)
            fit_joint(model, Xtr, ytr, P, tag=f"ST{cs}_P{P}")
            evaluate(model, Xte, yte, Xtr, ytr, tag=f"ST16_c{cs}_P{P}", P=P)
    log(event="done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cal")
