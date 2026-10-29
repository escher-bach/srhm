"""Structured (smooth) kernel on the adaptive codebook (srhm.tex section 8.10).

  log M_i[a' | a] = -||c_a' - phi_i(c_a; theta)||^2 / (2 psi^2) + log w_a' - log Z_i(a)

phi_i(z) = b_i + L_i z + B_i tanh(A_i z + a_i)  (small network in LEAF coordinates, tied across heights).
Leaves: hard tree-structured quantization (codebook_vq.TSVQ); c = cell means.  Beliefs over the v symbols
are exact (sum-product, codebook_torch).  Learning = generalized EM: exact E-step (pooled grid-local pair
statistics), M-step = gradient steps on sum_{i,a,a'} pair_i[a,a'] log M_i[a'|a] over (theta, psi, w).
Refining the codebook (larger v) reuses theta unchanged.
"""
import copy
import math
import numpy as np
import torch
import codebook_torch as cbt
import codebook_vq as vq

S = 3
DEV = cbt.DEV


class SmoothKernel(torch.nn.Module):
    def __init__(self, d, hidden=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        r = lambda *sh, s=1.0: (torch.randn(*sh, generator=g) * s)
        self.b = torch.nn.Parameter(r(S, d, s=0.1))
        self.Lm = torch.nn.Parameter(r(S, d, d, s=0.7 / math.sqrt(d)))
        self.A = torch.nn.Parameter(r(S, hidden, d, s=1.0 / math.sqrt(d)))
        self.a = torch.nn.Parameter(r(S, hidden, s=0.1))
        self.B = torch.nn.Parameter(r(S, d, hidden, s=0.3 / math.sqrt(hidden)))
        self.log_psi = torch.nn.Parameter(torch.tensor(math.log(0.6)))
        self.log_w = None

    def phi(self, z):
        """z (n, d) -> (S, n, d)"""
        h = torch.tanh(torch.einsum("shd,nd->snh", self.A, z) + self.a[:, None, :])
        return self.b[:, None, :] + torch.einsum("sed,nd->sne", self.Lm, z) + torch.einsum("seh,snh->sne", self.B, h)

    def M(self, c):
        """c (v, d) codebook -> M (S, v, v) with M[i, a', a] = P(child a' | parent a)."""
        v = c.shape[0]
        if self.log_w is None or self.log_w.shape[0] != v:
            self.log_w = torch.nn.Parameter(torch.zeros(v, device=c.device))
        mu = self.phi(c)                                                     # (S, v_parent, d)
        d2 = ((c[None, :, None, :] - mu[:, None, :, :]) ** 2).sum(-1)        # (S, a', a)
        logits = -d2 / (2 * torch.exp(2 * self.log_psi)) + self.log_w[None, :, None]
        return torch.softmax(logits, dim=1)


def fit(X, v_list, seed=0, em_iters=60, m_steps=25, lr=0.03, restarts=3, tol=1e-6):
    """Generalized EM, coarse-to-fine over v_list (theta reused).  Returns {v: (kernel, codebook, ll)}."""
    X = np.asarray(X, float)
    d = X.shape[-1] if X.ndim == 3 else 1
    Xf = X.reshape(-1, d)
    quant = vq.TSVQ(Xf, int(round(math.log2(max(v_list)))))
    out, best_state = {}, None
    for vi, v in enumerate(v_list):
        depth = int(round(math.log2(v)))
        c = torch.tensor(quant.means(depth), dtype=torch.float32, device=DEV)
        starts = range(restarts) if vi == 0 else [None]
        best = None
        for rs in starts:
            ker = SmoothKernel(d, seed=seed * 100 + rs).to(DEV) if rs is not None else best_state
            ker.M(c)                                                         # (re)create cell weights
            opt = torch.optim.Adam(ker.parameters(), lr=lr)
            prev = -math.inf
            for it in range(em_iters):
                with torch.no_grad():
                    cb = vq.VQCodebook(ker.M(c), c, quant, depth)
                    pair, _, _, ll = cbt.q_estep(cb, X)
                pair = pair.float()
                for _ in range(m_steps):                                      # M-step: local statistics only
                    Mx = ker.M(c)
                    obj = -(pair.transpose(1, 2) * torch.log(Mx + 1e-30)).sum() / pair.sum()
                    opt.zero_grad(); obj.backward(); opt.step()
                if ll - prev < tol * abs(ll):
                    break
                prev = ll
            with torch.no_grad():
                ll = cbt.q_estep(vq.VQCodebook(ker.M(c), c, quant, depth), X)[3]
            if best is None or ll > best[0]:
                best = (ll, ker)
        best_state = best[1]
        with torch.no_grad():
            out[v] = (copy.deepcopy(best_state), vq.VQCodebook(best_state.M(c), c, quant, depth), best[0])
        best_state.log_w = None                                               # re-sized at next v
    return out


def phi_error(ker, true_phi, Z, psi):
    """Mean over positions of RMS ||phi_hat_i(z) - phi_i(z)|| over latent samples Z, in units of psi."""
    with torch.no_grad():
        ph = ker.phi(torch.tensor(Z, dtype=torch.float32, device=DEV)).cpu().numpy()   # (S, n, d)
    tr = true_phi(Z)                                                                   # (n, S, d)
    return float(np.mean([np.sqrt(((ph[i] - tr[:, i]) ** 2).sum(-1).mean()) for i in range(S)]) / psi)


def fit_best(X, v_list, seed=0, n_restarts=8, em_iters=40):
    """Maximum likelihood with restarts SELECTED AT THE FINEST CODEBOOK: each restart is carried through
    the whole coarse-to-fine ladder, and the final log-likelihood chooses (srhm.tex section 8.10: at a
    coarse v the likelihood cannot separate the truth from near-automorphism solutions)."""
    best = None
    for r in range(n_restarts):
        out = fit(X, v_list, seed=seed * 1000 + r, em_iters=em_iters, restarts=1)
        if best is None or out[max(v_list)][2] > best[max(v_list)][2]:
            best = out
    return best
