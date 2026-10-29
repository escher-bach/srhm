"""Generic baseline for where.tex sec. prog-latent: a decoder-only transformer trained by backpropagation on the same
units in pre-order (a lossless serialisation of the AST).  Position i receives the previous symbol, the context t_i
and the parent depth of the current slot, and predicts x_i.  Summed log-loss = log-likelihood of the tree, in the
same nats per unit as the tree model.  Capacity (size) and early stopping chosen on the same 20 validation files.
Run locally:  python exp_code_tf.py [smoke]
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import astdata

BUDGET = float(os.environ.get("BUDGET_SECONDS", "1e9")); T0 = time.time()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
W, STRIDE, BATCH, N_VAL, MAXD = 512, 256, 32, 20, 31
SIZES = {"tiny": (2, 128, 4), "small": (4, 256, 8), "medium": (6, 384, 6)}


def log(**k):
    k["t"] = round(time.time() - T0, 1); print(json.dumps(k), flush=True)
    if os.environ.get("PROGRESS_FILE"):
        with open(os.environ["PROGRESS_FILE"], "a") as f:
            f.write(json.dumps(k) + "\n")


class Seqs:
    def __init__(self, d, files, C):
        sel = np.isin(d["ufile"], files)
        f = d["ufile"][sel]; x = d["x"][sel].astype(np.int64); t = d["t"][sel].astype(np.int64)
        dp = np.minimum(d["dp"][sel], MAXD).astype(np.int64)
        cut = np.nonzero(np.diff(f))[0] + 1
        self.files = [(a, b, c) for a, b, c in zip(np.split(x, cut), np.split(t, cut), np.split(dp, cut))]
        self.C = C; self.n_units = int(sel.sum())
        self.lens = np.array([len(a) for a, _, _ in self.files])

    def window(self, fi, start):
        x, t, dp = self.files[fi]
        xs = x[start:start + W]; n = len(xs)
        prev = np.concatenate([[self.C] if start == 0 else [x[start - 1]], xs[:-1]])
        pad = lambda a, v: np.concatenate([a, np.full(W - n, v, np.int64)])
        return pad(prev, self.C), pad(t[start:start + W], 0), pad(dp[start:start + W], 0), pad(xs, -100)

    def sample_batch(self, rng):
        p = self.lens / self.lens.sum()
        out = []
        for fi in rng.choice(len(self.files), BATCH, p=p):
            start = int(rng.integers(0, max(1, self.lens[fi] - W + 1)))
            out.append(self.window(fi, start))
        return [torch.as_tensor(np.stack(a), device=DEV) for a in zip(*out)]

    def eval_windows(self):
        """(fi, start, score_from): every unit scored once, with >= W - STRIDE context after the first window"""
        for fi, L in enumerate(self.lens):
            yield fi, 0, 0
            s = STRIDE
            while s + W - STRIDE < L:
                yield fi, s, W - STRIDE
                s += STRIDE


class TF(nn.Module):
    def __init__(self, C, T, size, drop):
        super().__init__()
        L, dm, nh = SIZES[size]
        self.emb_x = nn.Embedding(C + 1, dm); self.emb_t = nn.Embedding(T, dm)
        self.emb_d = nn.Embedding(MAXD + 1, dm); self.emb_p = nn.Embedding(W, dm)
        layer = nn.TransformerEncoderLayer(dm, nh, 4 * dm, drop, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, L)
        self.norm = nn.LayerNorm(dm); self.out = nn.Linear(dm, C); self.drop = nn.Dropout(drop)
        self.register_buffer("mask", torch.triu(torch.full((W, W), float("-inf")), 1))

    def forward(self, prev, t, dp):
        h = self.emb_x(prev) + self.emb_t(t) + self.emb_d(dp) + self.emb_p.weight[None, : prev.shape[1]]
        h = self.blocks(self.drop(h), mask=self.mask[: prev.shape[1], : prev.shape[1]], is_causal=True)
        return self.out(self.norm(h))


@torch.no_grad()
def evaluate(model, seqs):
    model.eval(); tot = 0.0; cnt = 0; buf = []
    def flush():
        nonlocal tot, cnt
        prev, t, dp, y, frm = [torch.as_tensor(np.stack(a), device=DEV) for a in zip(*buf)]
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            logits = model(prev, t, dp).float()
        lp = -F.cross_entropy(logits.transpose(1, 2), y.clamp(min=0), reduction="none")
        keep = (y >= 0) & (torch.arange(W, device=DEV)[None] >= frm[:, None])
        tot += float(lp[keep].double().sum()); cnt += int(keep.sum()); buf.clear()
    for fi, s, frm in seqs.eval_windows():
        buf.append(seqs.window(fi, s) + (np.int64(frm),))
        if len(buf) == 64: flush()
    if buf: flush()
    assert cnt == seqs.n_units, (cnt, seqs.n_units)
    model.train()
    return -tot / seqs.n_units


def train(tr, val, C, T, size, seed, max_steps, drop=0.1, lr=5e-4, eval_every=100, patience=6):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = TF(C, T, size, drop).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.98))
    sched = lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, max_steps) / max_steps))
    best, best_state, bad, step = math.inf, None, 0, 0
    for step in range(max_steps):
        for g in opt.param_groups: g["lr"] = lr * sched(step)
        prev, t, dp, y = tr.sample_batch(rng)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            loss = F.cross_entropy(model(prev, t, dp).float().transpose(1, 2), y, ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % eval_every == 0 or step == max_steps - 1:
            v = evaluate(model, val)
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
            else:
                bad += 1
            if bad >= patience or time.time() - T0 > BUDGET - 400: break
    model.load_state_dict(best_state)
    return model, best, step + 1


def main():
    smoke = "smoke" in sys.argv or os.environ.get("SMOKE") == "1"
    d = astdata.load()
    C, T = len(d["symbols"]), len(d["contexts"])
    test, pool = astdata.split(d)
    rs = np.random.default_rng(7)
    val = rs.choice(pool, N_VAL, replace=False); pool = np.setdiff1d(pool, val)
    if smoke: test = test[:40]
    val_s, te_s = Seqs(d, val, C), Seqs(d, test, C)
    log(dev=DEV, pool=len(pool), test_units=te_s.n_units, val_units=val_s.n_units)
    plan = ([(2, ["tiny"]), (4, ["tiny"])] if smoke else
            [(1, ["tiny", "small"]), (4, ["tiny", "small"]), (16, ["tiny", "small"]), (64, ["tiny", "small"]),
             (256, ["small", "medium"]), (len(pool), ["small", "medium"])])
    for n, sizes in plan:
        for s in range(1 if n == len(pool) else 2):
            files = np.random.default_rng(100 * n + s).choice(pool, n, replace=False)   # same files as exp_code_latent
            tr = Seqs(d, files, C)
            res = {}
            for size in sizes:
                if time.time() - T0 > BUDGET - 400:
                    log(stopped="budget"); return
                steps = 20 if smoke else 6000
                model, v, done = train(tr, val_s, C, T, size, s, steps, eval_every=10 if smoke else 100)
                res[size] = dict(val=round(v, 5), steps=done)
                res[size]["test"] = round(evaluate(model, te_s), 5)
                log(stage=True, files=n, seed=s, size=size, **res[size])
                del model
                if DEV == "cuda": torch.cuda.empty_cache()
            best = min(res, key=lambda k: res[k]["val"])
            log(files=n, seed=s, train_units=tr.n_units, by_size=res, best_size=best, best_test=res[best]["test"])


if __name__ == "__main__":
    main()
