"""Hard-quantized codebook tree for d > 1 (srhm.tex section 8.8, V2).

Leaves are quantized by tree-structured vector quantization (TSVQ): cell a at depth k splits into cells
2a, 2a+1 at depth k+1 by 2-means (started along the cell's principal axis).  Cells are nested, so the
coarse-to-fine symbol split of codebook_torch (new symbols 2a, 2a+1 refine a) stays exact.  No emission
noise: a leaf's symbol is its cell, a symbol's value is its cell mean (design rule of Q4).
"""
import copy
import numpy as np
import torch
import codebook_torch as cbt


class TSVQ:
    def __init__(self, X, depth_max, iters=15):
        X = np.asarray(X, float)
        self.cent = {0: X.mean(0, keepdims=True)}
        idx = np.zeros(len(X), dtype=np.int64)
        for k in range(1, depth_max + 1):
            new = np.zeros((2 ** k, X.shape[1]))
            for a in range(2 ** (k - 1)):
                pts = X[idx == a]
                if len(pts) < 2:
                    new[2 * a] = new[2 * a + 1] = self.cent[k - 1][a]
                    continue
                mu = pts.mean(0)
                direc = np.linalg.svd(pts - mu, full_matrices=False)[2][0]
                lab = (pts - mu) @ direc > 0
                for _ in range(iters):
                    c0 = pts[~lab].mean(0) if (~lab).any() else mu
                    c1 = pts[lab].mean(0) if lab.any() else mu
                    lab = ((pts - c1) ** 2).sum(1) < ((pts - c0) ** 2).sum(1)
                new[2 * a], new[2 * a + 1] = c0, c1
            self.cent[k] = new
            c0 = new[2 * idx]; c1 = new[2 * idx + 1]
            idx = 2 * idx + (((X - c1) ** 2).sum(1) < ((X - c0) ** 2).sum(1))
        self.X, self.depth_max = X, depth_max
        self.cent_t = {k: torch.tensor(v, dtype=torch.float32, device=cbt.DEV) for k, v in self.cent.items()}

    def assign(self, Xt, depth):
        """Xt torch (N, d) -> cell index at `depth` (descend the tree)."""
        idx = torch.zeros(Xt.shape[0], dtype=torch.long, device=Xt.device)
        for k in range(1, depth + 1):
            c = self.cent_t[k]
            d0 = ((Xt - c[2 * idx]) ** 2).sum(1); d1 = ((Xt - c[2 * idx + 1]) ** 2).sum(1)
            idx = 2 * idx + (d1 < d0).long()
        return idx

    def means(self, depth):
        idx = self.assign(torch.tensor(self.X, dtype=torch.float32, device=cbt.DEV), depth).cpu().numpy()
        c = self.cent[depth].copy()
        for a in range(2 ** depth):
            if (idx == a).any():
                c[a] = self.X[idx == a].mean(0)
        return c


class VQCodebook(cbt.QCodebook):
    def __init__(self, M, c, quant, depth):
        super().__init__(M, c, np.zeros(1))
        self.quant, self.depth = quant, depth


def vq_fit_coarse_to_fine(X, v_final, rng, v0=4, restarts=4, iters=300, tol=1e-8):
    X = np.asarray(X, float)
    Xf = X.reshape(-1, X.shape[-1])
    dmax = int(round(np.log2(v_final))); d0 = int(round(np.log2(v0)))
    quant = TSVQ(Xf, dmax)
    best = None
    for _ in range(restarts):
        M = rng.dirichlet(np.ones(v0), size=(3, v0)).transpose(0, 2, 1)
        cb, ll = cbt.q_run_em(VQCodebook(M, quant.means(d0), quant, d0), X, iters, tol)
        if best is None or ll > best[1]:
            best = (cb, ll)
    cb, ll = best
    out = {v0: (cb, ll)}
    for depth in range(d0 + 1, dmax + 1):
        v = cb.v
        M = cb.M.cpu().numpy()
        w = 0.5 + 0.05 * rng.uniform(-1, 1, size=(3, v, v))
        order = np.repeat(np.arange(v), 2)
        M2 = np.zeros((3, 2 * v, 2 * v))
        for i in range(3):
            child = np.stack([M[i][order[a2]] * (w[i][order[a2]] if a2 % 2 == 0 else 1 - w[i][order[a2]])
                              for a2 in range(2 * v)])
            M2[i] = child[:, order]
        cb, ll = cbt.q_run_em(VQCodebook(M2, quant.means(depth), quant, depth), X, iters, tol)
        out[2 * v] = (cb, ll)
    return out
