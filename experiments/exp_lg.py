"""Continuous (linear-Gaussian) experiments.  python verify/exp_lg.py {zero|bp|untied}

zero   : zero-label learner on the tied model: R^2 vs depth L and unlabelled trees U
bp     : backprop with root labels (tied linear SRG and untied linear CNN): R^2 vs labels P
untied : untied model (fresh random maps per level): zero-label learner, and labels needed
Model: d=3, n=3, A_i = 0.75 Q_i (Q_i random orthogonal), Psi = (1-0.75^2) I, Sigma0 = I.
"""
import json, os, sys, time
import numpy as np
from lgtree import LGTree, random_model, kalman_root, learn, r2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_lg_results.jsonl")
D, N, LAM = 3, 3, 0.75


def log(rec):
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def tied_tree(L, seed):
    A, Psi, S0 = random_model(D, N, LAM, np.random.default_rng(seed))
    return LGTree({h: A for h in range(1, L + 1)}, Psi, S0, L)


def run_zero(models=8):
    for L in [4, 5, 6, 7, 8, 9]:
        for U in [1, 2, 4, 8, 16, 32, 64]:
            if U * N ** (L - 1) > 200000:
                continue
            gaps = []
            for g in range(models):
                T = tied_tree(L, g)
                Xt, yt = T.sample(2000, np.random.default_rng(99))
                orc = r2(kalman_root(Xt, T.maps, T.Psi, T.Sigma0)[0], yt)
                X, _ = T.sample(U, np.random.default_rng(1000 * g + U))
                try:
                    maps, Ph, Sh = learn(X, N, L)
                    raw = kalman_root(Xt, maps, Ph, Sh)[0]
                    got = max(r2(raw, yt), r2(-raw, yt))          # global sign: see note
                except np.linalg.LinAlgError:
                    got = -np.inf
                gaps.append(orc - got)
            gaps = np.array(gaps)
            log(dict(exp="zero", L=L, U=U, bottom_grids=U * N ** (L - 1),
                     success=float((gaps <= 0.05).mean()), median_gap=float(np.median(gaps))))


def run_bp(models=2):
    import torch
    torch.set_num_threads(4)

    class Net(torch.nn.Module):
        def __init__(self, L, tied):
            super().__init__()
            self.L, self.tied = L, tied
            if tied:
                self.cell = torch.nn.Linear(N * D, D, bias=False)
            else:
                self.cells = torch.nn.ModuleList([torch.nn.Linear(N * D, D, bias=False) for _ in range(L)])
            self.out = torch.nn.Linear(D, D, bias=False)

        def forward(self, x):
            for l in range(self.L):
                x = x.reshape(x.shape[0], -1, N * D)
                x = (self.cell if self.tied else self.cells[l])(x)
            return self.out(x[:, 0])

    for L in [4, 6, 8]:
        for tied in [True, False]:
            for P in [8, 16, 32, 64, 128, 256, 512, 1024, 4096]:
                gaps = []
                for g in range(models):
                    T = tied_tree(L, g)
                    Xt, yt = T.sample(2000, np.random.default_rng(99))
                    orc = r2(kalman_root(Xt, T.maps, T.Psi, T.Sigma0)[0], yt)
                    X, y = T.sample(P, np.random.default_rng(1000 * g + P))
                    torch.manual_seed(g)
                    net = Net(L, tied)
                    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
                    Xs, ys = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
                    for step in range(3000):
                        loss = ((net(Xs) - ys) ** 2).mean()
                        opt.zero_grad(); loss.backward(); opt.step()
                    with torch.no_grad():
                        pred = net(torch.tensor(Xt, dtype=torch.float32)).numpy()
                    gaps.append(orc - r2(pred, yt))
                log(dict(exp="bp", L=L, tied=tied, P=P, median_gap=float(np.median(gaps)),
                         gaps=[round(x, 3) for x in gaps]))


def run_untied(models=8):
    """Fresh random maps at every level. (a) zero-label tied learner; (b) labels needed when
    every level below the top is known exactly (oracle) and only the top map is fitted."""
    for L in [6, 8]:
        for g in range(models):
            rng = np.random.default_rng(500 + g)
            maps = {h: random_model(D, N, LAM, rng)[0] for h in range(1, L + 1)}
            _, Psi, S0 = random_model(D, N, LAM, rng)
            T = LGTree(maps, Psi, S0, L)
            Xt, yt = T.sample(2000, np.random.default_rng(99))
            orc = r2(kalman_root(Xt, maps, Psi, S0)[0], yt)
            X, _ = T.sample(64, np.random.default_rng(g))
            mh, Ph, Sh = learn(X, N, L)
            raw = kalman_root(Xt, mh, Ph, Sh)[0]
            rec = dict(exp="untied_zero", L=L, model=g, oracle=orc,
                       r2_zero_label=float(max(r2(raw, yt), r2(-raw, yt))))
            # (b) everything known except the root's basis (a Haar rotation U, invisible in
            # unlabelled data because the root prior is isotropic).  The learner sees
            # w = U E[z|leaves] and fits the rotation back from P labels (orthogonal
            # Procrustes).  Theorem: R^2 <= (P/d) * oracle for P < d.
            U = np.linalg.qr(rng.normal(size=(D, D)))[0]
            wt = kalman_root(Xt, maps, Psi, S0)[0] @ U.T
            for P in [1, 2, 3, 4, 8, 16]:
                Xl, yl = T.sample(P, np.random.default_rng(7000 + P + g))
                wl = kalman_root(Xl, maps, Psi, S0)[0] @ U.T
                Uo, _, Vt = np.linalg.svd(wl.T @ yl)                 # orthogonal Procrustes
                W = Uo @ Vt
                rec[f"r2_labels_{P}"] = float(r2(wt @ W, yt))
            log(rec)


if __name__ == "__main__":
    {"zero": run_zero, "bp": run_bp, "untied": run_untied}[sys.argv[1]]()
