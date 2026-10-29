"""Head to head (srhm.tex section 8.9): the SAME hard-quantized codebook tree (v = 8, 1-d nonlinear
Gaussian tree, kernel 0), trained by two rules on the same P trees of depth L:
  local  -- EM (grid-local statistics), labels discarded (coarse-to-fine 4 -> 8);
  bp     -- backpropagation of the root-label loss (squared error of E[c_root | X] vs the true root)
            through every level; Adam; lr chosen per setting on a separate 300-tree labelled validation
            set that is NOT counted in P (backprop's best case).
Same quantizer (fitted on the P trees' unlabelled leaves), same random initial kernel, same test set
(400 trees of depth L).  Output: one JSON line per (rule, L, P, seed).
"""
import os, sys, json, time, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(globals().get("__file__", "./x"))))
import ngt, codebook_torch as cbt

V = 8
LS = (4, 6, 8)
PS = (2, 8, 32, 128, 512)
LRS = (0.03, 0.1, 0.3)


def r2(p, r):
    return float(1 - ((p - r) ** 2).mean() / r.var())


def predict(logits, c, edges, X, chunk=64):
    M = torch.softmax(logits, dim=1)                      # M[i, a', a]: distribution over a'
    cb = cbt.QCodebook(M, c.unsqueeze(1), edges)
    return cbt.q_root_value(cb, X, chunk)


def train_bp(X, r, edges, c0, M0, Xv, rv, lr, steps=600, batch=64, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.log(torch.as_tensor(M0, dtype=torch.float32, device=cbt.DEV)).clone().requires_grad_(True)
    c = torch.as_tensor(c0[:, 0], dtype=torch.float32, device=cbt.DEV).clone().requires_grad_(True)
    opt = torch.optim.Adam([logits, c], lr=lr)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=cbt.DEV)
    rt = torch.as_tensor(r, dtype=torch.float32, device=cbt.DEV)
    best = (-math.inf, None)
    for step in range(steps):
        idx = torch.randint(0, X.shape[0], (min(batch, X.shape[0]),), generator=g).to(cbt.DEV)
        M = torch.softmax(logits, dim=1)
        cb = cbt.QCodebook(M, c.unsqueeze(1), edges)
        ups, _, _, L, _ = cbt._with_q(cbt.upward)(cb, Xt[idx])
        post = ups[L] * cb.stationary()
        post = post / post.sum(1, keepdim=True)
        loss = ((post @ c - rt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 49 or step == steps - 1:            # model selection on validation (not in P)
            with torch.no_grad():
                val = r2(predict(logits, c, edges, Xv), rv)
            if val > best[0]:
                best = (val, (logits.detach().clone(), c.detach().clone()))
    return best


def main():
    print(json.dumps(dict(event="start", device=cbt.DEV)), flush=True)
    true = ngt.random_true_kernel(np.random.default_rng(100))
    for L in LS:
        Xte, rte = ngt.sample_trees(true, 400, L, np.random.default_rng(7))
        Xv, rv = ngt.sample_trees(true, 300, L, np.random.default_rng(8))
        r2o = r2(ngt.root_posterior_mean(true, Xte), rte)
        for P in PS:
            for seed in range(2):
                X, r = ngt.sample_trees(true, P, L, np.random.default_rng(10_000 * L + 100 * P + seed))
                edges, c0 = cbt.quantize(X, V)
                M0 = np.random.default_rng(seed).dirichlet(np.ones(V), size=(3, V)).transpose(0, 2, 1)
                t = time.time()
                fits = cbt.q_fit_coarse_to_fine(X, V, np.random.default_rng(seed))
                cb, _ = fits[V]
                rec = dict(rule="local", L=L, P=P, seed=seed, labels=0, r2=round(r2(cbt.q_root_value(cb, Xte), rte), 4),
                           r2_oracle=round(r2o, 4), secs=round(time.time() - t, 1))
                print(json.dumps(rec), flush=True)
                t = time.time(); best = None
                for lr in LRS:
                    val, params = train_bp(X, r, torch.as_tensor(edges, dtype=torch.float32, device=cbt.DEV),
                                           c0, M0, Xv, rv, lr, seed=seed)
                    if best is None or val > best[0]:
                        best = (val, params, lr)
                val, (logits, c), lr = best
                rec = dict(rule="bp", L=L, P=P, seed=seed, labels=P, lr=lr, val_r2=round(val, 4),
                           r2=round(r2(predict(logits, c, torch.as_tensor(edges, dtype=torch.float32, device=cbt.DEV), Xte), rte), 4),
                           r2_oracle=round(r2o, 4), secs=round(time.time() - t, 1))
                print(json.dumps(rec), flush=True)
    print(json.dumps(dict(event="done")), flush=True)


if __name__ == "__main__":
    main()
