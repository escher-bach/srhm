"""Claim-2 test (srhm.tex section 9): does backprop's label cost follow the linear KS number q_1?

Predictions, fixed before running (verify/check_claim2.py):
  grammar 0 (q_1 ~ 0.40): ||A_L|| = .542 .194 .088 .033 .013 .0053 (L=2..7) -> P*_bp grows ~2.4x/level
  grammar 2 (q_1 ~ 0.93): ||A_L|| = .805 .712 .639 .579 .529 .487          -> P*_bp nearly flat (~1.1x/level)
  weaker (shared constant, P* ~ 300/||A_L||): grammar 2 P* ~ 400-600 at every depth.

Protocol = srhm Table 3 (exp_srhm.py run_cnn), untied deep CNN, kernel = stride = s = 2, width 256,
Adam lr 1e-3, batch min(128,P), max(2000, 40P/bs) steps, early stop when a training subset is fit.
3 seeds per point.  P* = smallest P (powers of 2) with mean test accuracy >= 0.9.
Self-contained (Colab `run` ships one file).  Prints one JSON line per (grammar, L, P).
"""
import json, os, sys, time
import numpy as np
import torch

V, M, S = 8, 4, 2
T0 = time.time()
BUDGET = int(os.environ.get("BUDGET_SECONDS", "3600")) - 240      # leave margin for teardown
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Grammar:                       # identical sampling to verify/srhm.py
    def __init__(self, rng):
        tuples = rng.choice(V ** S, size=V * M, replace=False)
        self.R = tuples.reshape(V, M)
        c = np.arange(V ** S)
        self.digits = np.stack([(c // V ** (S - 1 - i)) % V for i in range(S)], -1)

    def sample(self, P, L, rng):
        z = rng.integers(0, V, size=(P, 1)); y = z[:, 0].copy()
        for _ in range(L):
            j = rng.integers(0, M, size=z.shape)
            z = self.digits[self.R[z, j]].reshape(P, -1)
        return z, y


class Net(torch.nn.Module):
    def __init__(self, L, H=256):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Conv1d(V if l == 0 else H, H, S, stride=S) for l in range(L)])
        self.out = torch.nn.Linear(H, V)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.out(x[:, :, 0])


def onehot(z):
    return torch.nn.functional.one_hot(torch.tensor(z), V).float().permute(0, 2, 1).to(DEV)


def train_eval(G, L, P, seed, Xt, yt, epochs=40, trace=None):
    torch.manual_seed(seed)
    X, y = G.sample(P, L, np.random.default_rng(seed))
    X, y = onehot(X), torch.tensor(y, device=DEV)
    net = Net(L).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bs = min(128, P); steps = 0; max_steps = max(2000, epochs * P // bs); ep = 0
    while steps < max_steps:
        perm = torch.randperm(P, device=DEV)
        for i in range(0, P, bs):
            idx = perm[i:i + bs]
            loss = torch.nn.functional.cross_entropy(net(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); steps += 1
        ep += 1
        if trace is not None and ep % 10 == 0:
            with torch.no_grad():
                trace.append((ep, round(float((net(Xt).argmax(1) == yt).float().mean()), 4)))
            if time.time() - T0 > BUDGET:
                break
        with torch.no_grad():
            sub = perm[:2048]
            if (net(X[sub]).argmax(1) == y[sub]).all() and loss.item() < 1e-3:
                break
    with torch.no_grad():
        return float((net(Xt).argmax(1) == yt).float().mean()), steps


def main():
    print(json.dumps(dict(event="start", device=DEV,
                          gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else None,
                          budget=BUDGET)), flush=True)
    for L in range(2, 8):
        for g in (0, 2):
            G = Grammar(np.random.default_rng(1000 + g))
            Xt, yt = G.sample(4000, L, np.random.default_rng(99))
            Xt, yt = onehot(Xt), torch.tensor(yt, device=DEV)
            for P in [2 ** k for k in range(5, 18)]:
                if time.time() - T0 > BUDGET:
                    print(json.dumps(dict(event="budget_stop", L=L, grammar=g, P=P)), flush=True)
                    return
                t = time.time(); accs, steps = [], []
                for seed in range(3):
                    a, st = train_eval(G, L, P, seed, Xt, yt); accs.append(a); steps.append(st)
                print(json.dumps(dict(grammar=g, L=L, P=P, acc=round(float(np.mean(accs)), 4),
                                      accs=[round(a, 4) for a in accs], steps=steps,
                                      secs=round(time.time() - t, 1))), flush=True)
                if np.mean(accs) >= 0.99:
                    break
    print(json.dumps(dict(event="done", secs=round(time.time() - T0))), flush=True)


def main_long():
    """Is grammar 0 at L=7 label-limited or plateau-limited?  5x the epoch budget, with traces."""
    G = Grammar(np.random.default_rng(1000))
    Xt, yt = G.sample(4000, 7, np.random.default_rng(99))
    Xt, yt = onehot(Xt), torch.tensor(yt, device=DEV)
    for P in (16384, 32768):
        for seed in range(3):
            if time.time() - T0 > BUDGET:
                return
            tr = []; t = time.time()
            a, st = train_eval(G, 7, P, seed, Xt, yt, epochs=200, trace=tr)
            print(json.dumps(dict(mode="long", grammar=0, L=7, P=P, seed=seed, acc=round(a, 4),
                                  steps=st, trace=tr, secs=round(time.time() - t, 1))), flush=True)


if __name__ == "__main__":
    main_long() if sys.argv[1:] == ["long"] else main()
