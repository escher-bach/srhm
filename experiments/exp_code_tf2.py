"""Stronger generic baseline (hist.tex sec. robust): the transformer of exp_code_tf.py, plus
  * the two context features of the combined model as inputs (previous sibling's symbol, parent's context),
  * a search over size (up to 8 x 512), dropout and learning rate at each N, selected on the validation files,
  * longer training (up to 20000 steps) with early stopping.
Same units, same training subsets, same validation and test files as every other model; nats per unit.
env: CORPUS (stdlib | scipy), STAGE (small: N = 1, 4, 16, 64 | mid: N = 256 | full: all files), SMOKE=1.
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
SIZES = {"tiny": (2, 128, 4), "small": (4, 256, 8), "medium": (6, 384, 6), "large": (8, 512, 8)}


def log(**k):
    k["t"] = round(time.time() - T0, 1); print(json.dumps(k), flush=True)
    if os.environ.get("PROGRESS_FILE"):
        with open(os.environ["PROGRESS_FILE"], "a") as f:
            f.write(json.dumps(k) + "\n")


class Seqs:
    def __init__(self, d, files, C, prev, tp):
        sel = np.isin(d["ufile"], files)
        f = d["ufile"][sel]; cut = np.nonzero(np.diff(f))[0] + 1
        cols = [d["x"][sel].astype(np.int64), d["t"][sel].astype(np.int64), np.minimum(d["dp"][sel], MAXD).astype(np.int64),
                prev[sel], tp[sel]]
        self.files = list(zip(*[np.split(c, cut) for c in cols]))
        self.C = C; self.n_units = int(sel.sum())
        self.lens = np.array([len(a[0]) for a in self.files])

    def window(self, fi, start):
        x, t, dp, pv, tq = self.files[fi]
        xs = x[start:start + W]; n = len(xs)
        last = np.concatenate([[self.C] if start == 0 else [x[start - 1]], xs[:-1]])
        pad = lambda a, v: np.concatenate([a, np.full(W - n, v, np.int64)])
        sl = slice(start, start + W)
        return pad(last, self.C), pad(t[sl], 0), pad(dp[sl], 0), pad(pv[sl], 0), pad(tq[sl], 0), pad(xs, -100)

    def sample_batch(self, rng):
        p = self.lens / self.lens.sum(); out = []
        for fi in rng.choice(len(self.files), BATCH, p=p):
            out.append(self.window(fi, int(rng.integers(0, max(1, self.lens[fi] - W + 1)))))
        return [torch.as_tensor(np.stack(a), device=DEV) for a in zip(*out)]

    def eval_windows(self):
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
        self.emb_x = nn.Embedding(C + 1, dm); self.emb_t = nn.Embedding(T, dm); self.emb_d = nn.Embedding(MAXD + 1, dm)
        self.emb_pv = nn.Embedding(C + 1, dm); self.emb_tp = nn.Embedding(T + 1, dm); self.emb_p = nn.Embedding(W, dm)
        layer = nn.TransformerEncoderLayer(dm, nh, 4 * dm, drop, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dm); self.out = nn.Linear(dm, C); self.drop = nn.Dropout(drop)
        self.register_buffer("mask", torch.triu(torch.full((W, W), float("-inf")), 1))

    def forward(self, last, t, dp, pv, tq):
        n = last.shape[1]
        h = self.emb_x(last) + self.emb_t(t) + self.emb_d(dp) + self.emb_pv(pv) + self.emb_tp(tq) + self.emb_p.weight[None, :n]
        h = self.blocks(self.drop(h), mask=self.mask[:n, :n], is_causal=True)
        return self.out(self.norm(h))


@torch.no_grad()
def evaluate(model, seqs):
    model.eval(); tot = 0.0; cnt = 0; buf = []
    def flush():
        nonlocal tot, cnt
        *inp, y, frm = [torch.as_tensor(np.stack(a), device=DEV) for a in zip(*buf)]
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            logits = model(*inp).float()
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


def train(tr, val, C, T, size, drop, lr, seed, max_steps, eval_every, patience):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = TF(C, T, size, drop).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.98))
    sched = lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, max_steps) / max_steps))
    best, best_state, bad, step = math.inf, None, 0, 0
    for step in range(max_steps):
        for g in opt.param_groups: g["lr"] = lr * sched(step)
        *inp, y = tr.sample_batch(rng)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            loss = F.cross_entropy(model(*inp).float().transpose(1, 2), y, ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % eval_every == 0 or step == max_steps - 1:
            v = evaluate(model, val)
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
            else:
                bad += 1
            if bad >= patience or time.time() - T0 > BUDGET - 500: break
    model.load_state_dict(best_state)
    return model, best, step + 1


def grid(n, smoke):
    if smoke: return [("tiny", 0.1, 1e-3)], 30, 10, 2
    if n <= 16: return [(s, dr, lr) for s in ("tiny", "small") for dr in (0.1, 0.3) for lr in (1e-3, 3e-4)], 3000, 50, 6
    if n <= 64: return [(s, dr, lr) for s in ("tiny", "small", "medium") for dr in (0.1, 0.3) for lr in (1e-3, 3e-4)], 5000, 100, 6
    if n <= 256: return [(s, dr, 5e-4) for s in ("small", "medium", "large") for dr in (0.1, 0.3)], 12000, 200, 6
    # a 4th entry overrides the schedule length: the first transformer's best run used a 6000-step cosine, and on the
    # standard library the 20000-step schedule did worse (0.896 against 0.862)
    return [("medium", 0.1, 5e-4, 6000), ("medium", 0.3, 5e-4, 8000), ("small", 0.3, 5e-4), ("medium", 0.3, 5e-4),
            ("large", 0.3, 5e-4, 8000)], 20000, 250, 8


def main():
    smoke = os.environ.get("SMOKE") == "1"; stage = os.environ.get("STAGE", "small")
    d = astdata.load()
    C, T = len(d["symbols"]), len(d["contexts"])
    prev, tp = astdata.features(d)
    test, pool = astdata.split(d)
    val = np.random.default_rng(7).choice(pool, N_VAL, replace=False); pool = np.setdiff1d(pool, val)
    if smoke: test = test[:30]
    val_s, te_s = Seqs(d, val, C, prev, tp), Seqs(d, test, C, prev, tp)
    log(dev=DEV, corpus=astdata.CORPUS, stage=stage, pool=len(pool), test_units=te_s.n_units, val_units=val_s.n_units, C=C, T=T)
    ns = {"small": [1, 4, 16, 64], "mid": [256], "full": [len(pool)]}[stage]
    if smoke: ns = [2]
    for n in ns:
        n = min(n, len(pool)); best_cfg = None
        for s in range(1 if n == len(pool) else 2):
            files = np.random.default_rng(100 * n + s).choice(pool, n, replace=False)   # the files of every other model
            tr = Seqs(d, files, C, prev, tp)
            cfgs, max_steps, eval_every, patience = grid(n, smoke)
            cfgs = cfgs[:int(os.environ.get("GRID_LIMIT", "99"))]
            if s > 0: cfgs = [best_cfg]                                                  # later seeds: the first seed's best
            best = None
            for cfg in cfgs:
                size, drop, lr = cfg[:3]; ms = cfg[3] if len(cfg) > 3 else max_steps
                if time.time() - T0 > BUDGET - 500:
                    log(stopped="budget"); return
                model, v, done = train(tr, val_s, C, T, size, drop, lr, s, ms, eval_every, patience)
                rec = dict(files=n, seed=s, size=size, drop=drop, lr=lr, max_steps=ms, val=round(v, 5), steps=done)
                if best is None or v < best["val"]:
                    rec["test"] = round(evaluate(model, te_s), 5); best = rec
                    if s == 0: best_cfg = cfg
                log(cfg=True, **rec)
                del model
                if DEV == "cuda": torch.cuda.empty_cache()
            log(result=True, corpus=astdata.CORPUS, files=n, seed=s, train_units=tr.n_units, **{k: best[k] for k in ("size", "drop", "lr", "val", "test", "steps")})


if __name__ == "__main__":
    main()
