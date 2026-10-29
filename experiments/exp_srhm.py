"""Experiments for the tied Random Hierarchy Model (srhm.tex).

  python verify/exp_srhm.py ctx     # zero-label bottom-level learner: P* vs depth
  python verify/exp_srhm.py cnn     # backprop (root labels): untied CNN and tied CNN (SRG)
  python verify/exp_srhm.py consts  # channel constants lambda, lambda_sib, q; |Aut(D)|
Grammar: v=8 symbols, m=4 rules per symbol, s=2 (fraction of valid tuples f = 1/2).
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from srhm import Grammar, ctx_learn, n_eff

V, M, S = 8, 4, 2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_srhm_results.jsonl")


def log(rec):
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def run_ctx(seeds=8):
    for L in range(5, 12):
        for P in [1, 2, 4, 8, 16, 32]:
            if P * S ** (L - 1) > 8192:          # far past the threshold, skip
                continue
            ok = []
            for seed in range(seeds):
                G = Grammar(V, M, S, np.random.default_rng(1000 + seed))
                r = np.random.default_rng(seed)
                X, _ = G.sample(P, L, r)
                tab, n = ctx_learn(G, X, restarts=15)
                Xt, yt = G.sample(1000, L, r)
                ok.append(float((G.parse(Xt, tab) == yt).mean()))
            log(dict(exp="ctx", L=L, P=P, bottom_grids=P * S ** (L - 1),
                     success=float(np.mean([a == 1.0 for a in ok])), acc=float(np.mean(ok))))


def run_cnn(seeds=2, Ls=(2, 3, 4, 5, 6), tieds=(False, True)):
    import torch
    torch.set_num_threads(2)

    class Net(torch.nn.Module):
        def __init__(self, L, H, tied):
            super().__init__()
            self.L, self.tied = L, tied
            if tied:    # scale-tied SRG: one cell map applied at every level
                self.emb = torch.nn.Conv1d(V, H, 1)
                self.cell = torch.nn.Conv1d(H, H, S, stride=S)
            else:       # standard deep CNN: kernel = stride = s, separate weights per layer
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Conv1d(V if l == 0 else H, H, S, stride=S) for l in range(L)])
            self.out = torch.nn.Linear(H, V)

        def forward(self, x):
            if self.tied:
                x = torch.relu(self.emb(x))
                for _ in range(self.L):
                    x = torch.relu(self.cell(x))
            else:
                for layer in self.layers:
                    x = torch.relu(layer(x))
            return self.out(x[:, :, 0])

    def onehot(z):
        return torch.nn.functional.one_hot(torch.tensor(z), V).float().permute(0, 2, 1)

    for L in Ls:
        G = Grammar(V, M, S, np.random.default_rng(1000))
        Xt, yt = G.sample(4000, L, np.random.default_rng(99))
        Xt, yt = onehot(Xt), torch.tensor(yt)
        for tied in tieds:
            for P in [2 ** k for k in range(5, 17)]:
                accs = []
                for seed in range(seeds):
                    torch.manual_seed(seed)
                    X, y = G.sample(P, L, np.random.default_rng(seed))
                    X, y = onehot(X), torch.tensor(y)
                    net = Net(L, 256, tied)
                    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
                    bs = min(128, P); steps = 0; t0 = time.time()
                    max_steps = max(2000, 40 * P // bs)
                    while steps < max_steps:
                        perm = torch.randperm(P)
                        for i in range(0, P, bs):
                            idx = perm[i:i + bs]
                            loss = torch.nn.functional.cross_entropy(net(X[idx]), y[idx])
                            opt.zero_grad(); loss.backward(); opt.step(); steps += 1
                        with torch.no_grad():   # stopping check on a training subset
                            sub = perm[:2048]
                            if (net(X[sub]).argmax(1) == y[sub]).all() and loss.item() < 1e-3:
                                break
                    with torch.no_grad():
                        accs.append(float((net(Xt).argmax(1) == yt).float().mean()))
                log(dict(exp="cnn", tied=tied, L=L, P=P, acc=float(np.mean(accs)),
                         secs=round(time.time() - t0, 1)))
                if np.mean(accs) > 0.99:
                    break


def run_consts():
    for seed in range(8):
        G = Grammar(V, M, S, np.random.default_rng(1000 + seed))
        lam, ls = G.channel_constants(8)
        ne = {L: round(n_eff(L, S, lam, ls)[0], 2) for L in [6, 8, 10, 12]}
        log(dict(exp="consts", seed=seed, lam=lam, lam_sib=ls, q=S * lam ** 2,
                 aut_D=G.automorphisms_of_D(), n_eff_bound=ne))


if __name__ == "__main__":
    if sys.argv[1] == "cnn" and len(sys.argv) > 2:      # e.g.  cnn 5 tied
        run_cnn(Ls=(int(sys.argv[2]),), tieds=(sys.argv[3] == "tied",))
    else:
        {"ctx": run_ctx, "cnn": run_cnn, "consts": run_consts}[sys.argv[1]]()
