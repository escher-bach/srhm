"""Baseline for lang.tex: a decoder-only transformer trained by backpropagation on the same bytes, the same training
articles, the same validation and test articles as lang.py.  Position i receives the previous byte (a start symbol at
the head of an article) and predicts byte i from up to 256 bytes of history.  Summed log-loss = log-likelihood of the
article, in the same nats per byte.  Size, dropout and stopping point are chosen on the validation articles.
Runs on Colab (L4): colab_safe_run3.sh L4 exp_lang_tf_bundle.py MIN OUT -      SMOKE=1 for a pipeline check.
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import langdata

BUDGET = float(os.environ.get("BUDGET_SECONDS", "1e9")); T0 = time.time()
SMOKE = os.environ.get("SMOKE") == "1"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
W, STRIDE, BATCH, C = 256, 128, 64, 256
SIZES = {"tiny": (2, 128, 4), "small": (4, 256, 8), "medium": (6, 384, 6), "large": (8, 512, 8)}


def log(**k):
    k["t"] = round(time.time() - T0, 1); print(json.dumps(k), flush=True)
    if os.environ.get("PROGRESS_FILE"):
        with open(os.environ["PROGRESS_FILE"], "a") as f:
            f.write(json.dumps(k) + "\n")


class Seqs:
    def __init__(self, D, files):
        self.files = []
        for f in files:
            a, b = np.searchsorted(D["fid"], [f, f + 1]); self.files.append(D["x"][a:b])
        self.lens = np.array([len(a) for a in self.files]); self.n_units = int(self.lens.sum())

    def window(self, fi, start):
        x = self.files[fi]; xs = x[start:start + W]; n = len(xs)
        prev = np.concatenate([[C] if start == 0 else [x[start - 1]], xs[:-1]])
        pad = lambda a, v: np.concatenate([a, np.full(W - n, v, np.int64)])
        return pad(prev, C), pad(xs, -100)

    def sample_batch(self, rng):
        p = self.lens / self.lens.sum(); out = []
        for fi in rng.choice(len(self.files), BATCH, p=p):
            out.append(self.window(fi, int(rng.integers(0, max(1, self.lens[fi] - W + 1)))))
        return [torch.as_tensor(np.stack(a), device=DEV) for a in zip(*out)]

    def eval_windows(self):
        """(file, start, score_from): every byte scored once, with at least W - STRIDE bytes of history after the first window"""
        for fi, L in enumerate(self.lens):
            yield fi, 0, 0
            s = STRIDE
            while s + W - STRIDE < L:
                yield fi, s, W - STRIDE
                s += STRIDE


class TF(nn.Module):
    def __init__(self, size, drop):
        super().__init__()
        L, dm, nh = SIZES[size]
        self.emb_x = nn.Embedding(C + 1, dm); self.emb_p = nn.Embedding(W, dm)
        layer = nn.TransformerEncoderLayer(dm, nh, 4 * dm, drop, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, L)
        self.norm = nn.LayerNorm(dm); self.out = nn.Linear(dm, C); self.drop = nn.Dropout(drop)
        self.register_buffer("mask", torch.triu(torch.full((W, W), float("-inf")), 1))

    def forward(self, prev):
        h = self.emb_x(prev) + self.emb_p.weight[None, : prev.shape[1]]
        h = self.blocks(self.drop(h), mask=self.mask[: prev.shape[1], : prev.shape[1]], is_causal=True)
        return self.out(self.norm(h))


@torch.no_grad()
def evaluate(model, seqs):
    model.eval(); tot = 0.0; cnt = 0; buf = []
    def flush():
        nonlocal tot, cnt
        prev, y, frm = [torch.as_tensor(np.stack(a), device=DEV) for a in zip(*buf)]
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            logits = model(prev).float()
        lp = -F.cross_entropy(logits.transpose(1, 2), y.clamp(min=0), reduction="none")
        keep = (y >= 0) & (torch.arange(W, device=DEV)[None] >= frm[:, None])
        tot += float(lp[keep].double().sum()); cnt += int(keep.sum()); buf.clear()
    for fi, s, frm in seqs.eval_windows():
        buf.append(seqs.window(fi, s) + (np.int64(frm),))
        if len(buf) == 256: flush()
    if buf: flush()
    assert cnt == seqs.n_units, (cnt, seqs.n_units)
    model.train()
    return -tot / seqs.n_units


def train(tr, val, size, drop, seed, max_steps, cap, lr=6e-4, eval_every=250, patience=6):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); t_run = time.time()
    model = TF(size, drop).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.98))
    sched = lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, max_steps) / max_steps))
    best, best_state, bad, step, why = math.inf, None, 0, 0, "steps"
    for step in range(max_steps):
        for g in opt.param_groups: g["lr"] = lr * sched(step)
        prev, y = tr.sample_batch(rng)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            loss = F.cross_entropy(model(prev).float().transpose(1, 2), y, ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % eval_every == 0 or step == max_steps - 1:
            v = evaluate(model, val)
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
            else:
                bad += 1
            if bad >= patience: why = "patience"; break
            if time.time() - t_run > cap or time.time() - T0 > BUDGET - 500: why = "time"; break
    model.load_state_dict(best_state)
    return model, best, step + 1, why


def main():
    D = langdata.load()
    if SMOKE: D["val"], D["test"] = D["val"][:3], D["test"][:5]
    val_s, te_s = Seqs(D, D["val"]), Seqs(D, D["test"]); nall = len(D["train"])
    log(dev=DEV, train_files=nall, test_units=te_s.n_units, val_units=val_s.n_units)
    # (files, [(size, dropout)], max steps, minutes per run, evaluation interval)
    plan = ([(1, [("tiny", 0.1)], 30, 1, 10), (4, [("tiny", 0.1)], 30, 1, 10)] if SMOKE else
            [(1, [("tiny", 0.1), ("tiny", 0.3), ("small", 0.3)], 3000, 4, 50),
             (4, [("tiny", 0.1), ("tiny", 0.3), ("small", 0.3)], 3000, 4, 50),
             (16, [("tiny", 0.1), ("small", 0.1), ("small", 0.3)], 4000, 5, 100),
             (64, [("small", 0.1), ("small", 0.3), ("medium", 0.2)], 8000, 9, 250),
             (256, [("small", 0.1), ("medium", 0.1), ("medium", 0.2)], 12000, 16, 250),
             (nall, [("medium", 0.1), ("large", 0.1)], 20000, 38, 500)])
    if os.environ.get("PLAN"):                      # a second pass with longer schedules: JSON list of the same tuples (0 = all files)
        plan = [(n or nall, [tuple(c) for c in cf], st, mi, ev) for n, cf, st, mi, ev in json.loads(os.environ["PLAN"])]
    for n, configs, steps, minutes, every in plan:
        tr = Seqs(D, np.sort(langdata.subset(D, n, 0))); res = {}
        for size, drop in configs:
            if time.time() - T0 > BUDGET - 600:
                log(stopped="budget"); return
            model, v, done, why = train(tr, val_s, size, drop, 0, steps, 60 * minutes, eval_every=every)
            key = "%s/%.1f" % (size, drop)
            res[key] = dict(val=round(v, 5), steps=done, stop=why, test=round(evaluate(model, te_s), 5))
            log(stage=True, files=n, config=key, **res[key])
            del model
            if DEV == "cuda": torch.cuda.empty_cache()
        best = min(res, key=lambda k: res[k]["val"])
        log(files=n, train_units=tr.n_units, by_config=res, best_config=best, best_test=res[best]["test"])


if __name__ == "__main__":
    main()
