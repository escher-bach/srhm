"""CIFAR-10 calibration before the image tree (srhm.tex section 10).  Measures two DATA facts the
predictions depend on; no learner of the theory is run here.

(a) Class information in the thumbnail pyramid: test accuracy of multinomial logistic regression and of
    a small MLP on the k x k block-average thumbnail, k = 1, 2, 4, 8, 32.  A root whose state is only a
    2x2 thumbnail cannot beat the k = 2 numbers (data processing).
(b) Myhill--Nerode witness: in a tree whose state is the parent thumbnail only, sibling children are
    conditionally independent given it.  We regress each child's log detail energy on its parent's
    thumbnail (MLP, per child position) and report the residual correlation between SIBLINGS and, as a
    control, between COUSINS (same position, children of two sibling parents) -- per height.
(c) Per-height detail statistics (Haar energy, kurtosis): how self-similar CIFAR is across its 5 octaves.
"""
import os, json, time
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BUDGET = float(os.environ.get("BUDGET_SECONDS", "1e9")); T0 = time.time()


def log(**k):
    k["t"] = round(time.time() - T0, 1); print(json.dumps(k), flush=True)


def load():
    import torchvision
    tr = torchvision.datasets.CIFAR10("/content/cifar", train=True, download=True)
    te = torchvision.datasets.CIFAR10("/content/cifar", train=False, download=True)
    f = lambda d: (torch.tensor(d.data).float().add(0.5).div(256.0), torch.tensor(d.targets))
    return f(tr), f(te)


def thumb(x, k):                       # x (N,32,32,3) -> (N,k,k,3) block average
    b = 32 // k
    return x.reshape(x.shape[0], k, b, k, b, 3).mean((2, 4))


def fit_clf(Xtr, ytr, Xte, yte, hidden=0, epochs=60, wd=1e-4):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = ((Xtr - mu) / sd).to(DEV), ((Xte - mu) / sd).to(DEV)
    ytr, yte = ytr.to(DEV), yte.to(DEV)
    D = Xtr.shape[1]
    net = (torch.nn.Linear(D, 10) if hidden == 0 else
           torch.nn.Sequential(torch.nn.Linear(D, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                               torch.nn.ReLU(), torch.nn.Linear(hidden, 10))).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    n = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, 512):
            idx = perm[s:s + 512]
            loss = torch.nn.functional.cross_entropy(net(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float((net(Xte).argmax(1) == yte).float().mean())


def haar(block):                       # (..., 2, 2, 3) -> mean (...,3), detail (..., 9)
    a, b, c, d = block[..., 0, 0, :], block[..., 0, 1, :], block[..., 1, 0, :], block[..., 1, 1, :]
    m = (a + b + c + d) / 4
    det = torch.cat([(a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2], -1)
    return m, det


def pyramid_nodes(x, h):
    """Nodes at height h (h = 0: 2x2 pixel patches): thumbnails (N, g, g, 2, 2, 3), g = 16 / 2^h."""
    I = thumb(x, 32 // 2 ** h)                           # 2^(5-h) resolution
    g = I.shape[1] // 2
    return I.reshape(x.shape[0], g, 2, g, 2, 3).permute(0, 1, 3, 2, 4, 5)


def mn_witness(x, h_child, n_img=20000):
    """Children at height h_child, parents at h_child + 1."""
    x = x[:n_img]
    C = pyramid_nodes(x, h_child)                        # (N, g, g, 2,2,3)
    P = pyramid_nodes(x, h_child + 1)                    # (N, g/2, g/2, 2,2,3)
    _, det = haar(C)
    e = torch.log((det ** 2).sum(-1) + 1e-6)             # (N, g, g)
    N, g = e.shape[0], e.shape[1]
    gp = g // 2
    e = e.reshape(N, gp, 2, gp, 2).permute(0, 1, 3, 2, 4).reshape(N, gp, gp, 4)   # child pos i = 2r+c
    par = P.reshape(N, gp, gp, 12)
    X = par.reshape(-1, 12).to(DEV); Y = e.reshape(-1, 4).to(DEV)
    net = torch.nn.Sequential(torch.nn.Linear(12, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128),
                              torch.nn.ReLU(), torch.nn.Linear(128, 4)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ntr = X.shape[0] // 2                                 # fit on half, residuals on the other half
    for it in range(3000):
        idx = torch.randint(0, ntr, (1024,), device=DEV)
        loss = ((net(X[idx]) - Y[idx]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        R = (Y - net(X)).reshape(N, gp, gp, 4)[N // 2:]
        Rn = (R - R.mean()) / R.std()
        sib = float((Rn[..., 0] * Rn[..., 3]).mean() + (Rn[..., 0] * Rn[..., 1]).mean()) / 2
        out = dict(h_child=h_child, sibling_corr=round(sib, 4), resid_var_frac=round(float(R.var() / Y.var()), 4))
        if gp >= 2:   # cousins: same child position, parents horizontally adjacent within a grandparent
            cou = float((Rn[:, :, 0::2, 1] * Rn[:, :, 1::2, 0]).mean())
            out["cousin_corr_adjacent"] = round(cou, 4)
            far = float((Rn[:, 0::2, 0::2, 0] * Rn[:, 1::2, 1::2, 3]).mean())
            out["cousin_corr_diag_far"] = round(far, 4)
    return out


def detail_stats(x, h_child, n_img=20000):
    C = pyramid_nodes(x[:n_img], h_child)
    _, det = haar(C)
    d = det.reshape(-1, 9)
    k = ((d - d.mean(0)) ** 4).mean(0) / d.var(0) ** 2
    return dict(h_child=h_child, energy=round(float((d ** 2).sum(1).mean()), 6),
                kurtosis_mean=round(float(k.mean()), 2))


def main():
    (xtr, ytr), (xte, yte) = load()
    log(event="loaded", n_train=len(ytr), n_test=len(yte), dev=DEV)
    for h in range(4):
        log(event="detail_stats", **detail_stats(xtr, h))
    for h in range(4):
        log(event="mn_witness", **mn_witness(xtr, h))
    for k in (1, 2, 4, 8, 32):
        Xtr, Xte = thumb(xtr, k).reshape(len(ytr), -1), thumb(xte, k).reshape(len(yte), -1)
        lin = fit_clf(Xtr, ytr, Xte, yte, 0, epochs=40)
        mlp = fit_clf(Xtr, ytr, Xte, yte, 512 if k <= 8 else 1024, epochs=40)
        log(event="thumb_acc", k=k, linear=round(lin, 4), mlp=round(mlp, 4))
        if time.time() - T0 > BUDGET - 60:
            break
    log(event="done")


if __name__ == "__main__":
    main()
