"""A root that keeps the configuration: the state grid and its readout (srhm.tex Sec. grid; G1-G5 fixed there first).

Zero-label tied IT(16) (3000 steps) on N = 50,000 and N = 1024 unlabelled images.  For c = 3, 2, 1 the grid features of
an image are, per grid node v, the normalized message from its subtree e~_v (H numbers) and its thumbnail t_v (12).
Heads (P labelled images = the first P of a fixed random order):
  untied logistic   on (e~_v, t_v)_v                           (conditional ML of GP(c), position-dependent)
  tied logistic     on (sum_v e~_v, sum_v t_v)                 (GP(c) with a position-free head)
  Bayes grid head   generative ML of GP(c): rho_v(h|y) from soft counts, class means + pooled covariance of t_v
Head to head at c = 2 (claim 2): the same tree + untied linear head trained end to end by backpropagating the
cross-entropy from P labels, from the same initialization; lr best of 3 on 5000 held-out labelled images (the last
5000 of the order, never in P), early stopping on them.
"""
import os, sys, math, time, copy
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(globals().get("__file__", "./x"))))
import imtree as it
import exp_cifar as ex

DEV, LN2, log, left = ex.DEV, ex.LN2, ex.log, ex.left
H, STEPS = 16, int(os.environ.get("GRID_STEPS", "3000"))
PS = (100, 300, 1000, 3000, 10000, 50000)
BP_STEPS = int(os.environ.get("GRID_BP_STEPS", "3000"))


def grid_feats(model, P, c):
    """Differentiable grid features: (B, n_c, H) normalized subtree messages and (B, n_c, 12) thumbnails."""
    inc = None
    for k in range(c):
        ak = model.emis(k, P["tp"][k], P["det"][k])
        ak = ak if inc is None else ak + inc
        B, n, Hh = ak.shape
        mx = ak.max(-1, keepdim=True).values
        e = torch.exp(ak - mx).reshape(B, n // 4, 4, Hh)
        T = torch.exp(model.logT(k))
        m = torch.log(torch.einsum("ixy,bpiy->bpix", T, e) + 1e-38) + mx.reshape(B, n // 4, 4, 1)
        inc = m.sum(2)
    return torch.softmax(inc, -1), P["tp"][c - 1][:, ::4]


@torch.no_grad()
def features(model, X, c, bs=1000):
    g = torch.Generator(device=DEV).manual_seed(7)
    E, Tt = [], []
    for s in range(0, X.shape[0], bs):
        e, t = grid_feats(model, it.pyramid(ex.deq(X[s:s + bs], g)), c)
        E.append(e); Tt.append(t)
    return torch.cat(E), torch.cat(Tt)                                   # (N, n, H), (N, n, 12)


def untied(E, T):
    return torch.cat([E, T], -1).reshape(E.shape[0], -1)


def tied(E, T):
    return torch.cat([E.sum(1), T.sum(1)], -1)


@torch.no_grad()
def bayes_grid(Etr, Ttr, ytr, Ete, Tte, yte, P, alpha=0.1, ridge=1e-4):
    E, T, y = Etr[:P], Ttr[:P], ytr[:P]
    Y = torch.nn.functional.one_hot(y, 10).float()                       # (P, 10)
    cnt = torch.einsum("pc,pnh->cnh", Y, E) + alpha                        # (10, n, H)
    lrho = torch.log(cnt / cnt.sum(-1, keepdim=True))
    Ny = Y.sum(0).clamp(min=1)
    mu = torch.einsum("pc,pnd->cnd", Y, T) / Ny[:, None, None]            # (10, n, 12)
    R = T - mu[y]                                                          # residuals
    S = torch.einsum("pnd,pne->nde", R, R) / max(P - 10, 1) + ridge * torch.eye(12, device=DEV)
    Si = torch.linalg.inv(S)                                               # (n, 12, 12)
    lp = torch.log(Ny / Ny.sum())
    out = lp[None].expand(Ete.shape[0], -1).clone()
    for cl in range(10):
        out[:, cl] += torch.log((Ete * torch.exp(lrho[cl])[None]).sum(-1).clamp(min=1e-30)).sum(1)
        a = torch.einsum("nde,ne->nd", Si, mu[cl])                         # Sigma^-1 mu
        out[:, cl] += (Tte * a[None]).sum((1, 2)) - 0.5 * (mu[cl] * a).sum()
    return round(float((out.argmax(1) == yte).float().mean()), 4)


def heads(model, Xtr, ytr, Xte, yte, c, tag, Ps=PS, which=("untied", "tied", "bayes")):
    Etr, Ttr = features(model, Xtr, c); Ete, Tte = features(model, Xte, c)
    res = dict(event="heads", tag=tag, c=c, n_regions=Etr.shape[1],
               msg_certainty_eps=round(float((1 - Ete.max(-1).values).mean()), 4))
    if "untied" in which:
        Ftr, Fte = untied(Etr, Ttr), untied(Ete, Tte)
        res["untied"] = {P: round(ex.logistic_head(Ftr, ytr, Fte, yte, P), 4) for P in Ps}
    if "tied" in which:
        Ftr, Fte = tied(Etr, Ttr), tied(Ete, Tte)
        res["tied"] = {P: round(ex.logistic_head(Ftr, ytr, Fte, yte, P), 4) for P in Ps}
    if "bayes" in which:
        res["bayes"] = {P: bayes_grid(Etr, Ttr, ytr, Ete, Tte, yte, P) for P in Ps}
    log(**res)
    return res


# ------------------------------------------------------------------ claim 2: backprop through the same architecture
def backprop(Xtr, ytr, Xval, yval, Xte, yte, P, lr, c=2, steps=BP_STEPS, batch=128):
    torch.manual_seed(0)
    model = it.ImageTree(H, seed=0).to(DEV)                               # the same initialization as EM
    n = 4 ** (model.L - c)
    head = torch.nn.Linear(n * (H + 12), 10).to(DEV)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))
    g = torch.Generator(device=DEV).manual_seed(3)

    def logits(X, gen=None):
        e, t = grid_feats(model, it.pyramid(ex.deq(X, gen)), c)
        return head(torch.cat([e, (t - 0.47) / 0.25], -1).reshape(X.shape[0], -1))

    @torch.no_grad()
    def acc(X, y):
        gg = torch.Generator(device=DEV).manual_seed(11)
        return float(sum(int((logits(X[s:s + 1000], gg).argmax(1) == y[s:s + 1000]).sum())
                         for s in range(0, X.shape[0], 1000)) / X.shape[0])

    best, best_state = -1.0, None
    for s in range(steps):
        idx = torch.randint(0, P, (min(batch, P),), device=DEV, generator=g)
        loss = torch.nn.functional.cross_entropy(logits(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (s + 1) % 250 == 0 or s == steps - 1:
            v = acc(Xval, yval)
            if v > best:
                best = v; best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict()))
    model.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    return best, acc(Xte, yte)


def main():
    log(event="start", dev=DEV, budget=ex.BUDGET, steps=STEPS)
    torch.manual_seed(0)
    Xtr, ytr, Xte, yte = ex.load_cifar()
    perm = torch.randperm(Xtr.shape[0], generator=torch.Generator().manual_seed(0)).to(DEV)
    Xtr, ytr = Xtr[perm], ytr[perm]
    # zero-label features
    model = it.ImageTree(H, seed=0).to(DEV)
    ex.fit(model, Xtr, STEPS, 128, tag="IT16_N50000")
    for c in (3, 2, 1):
        heads(model, Xtr, ytr, Xte, yte, c, tag="N50000")
    small = it.ImageTree(H, seed=0).to(DEV)
    ex.fit(small, Xtr[:1024], STEPS, 128, tag="IT16_N1024")
    heads(small, Xtr, ytr, Xte, yte, 2, tag="N1024", Ps=(50000,), which=("untied", "tied"))
    # claim 2 head to head at c = 2
    Xval, yval = Xtr[-5000:], ytr[-5000:]
    for P in (300, 1000, 3000, 10000, 45000):
        if left() < 600:
            log(event="budget_stop", P=P); break
        runs = {}
        for lr in (1e-3, 3e-3, 1e-2):
            t = time.time()
            v, te = backprop(Xtr, ytr, Xval, yval, Xte, yte, P, lr)
            runs[lr] = (round(v, 4), round(te, 4))
            log(event="bp_run", P=P, lr=lr, val=round(v, 4), test=round(te, 4), secs=round(time.time() - t, 1))
        lr_best = max(runs, key=lambda k: runs[k][0])
        log(event="bp", P=P, lr=lr_best, val=runs[lr_best][0], test=runs[lr_best][1])
    log(event="done")


if __name__ == "__main__":
    main()
