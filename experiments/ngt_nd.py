"""Bounded nonlinear Gaussian tied tree in d dimensions, and its grid oracle (torch).

Model: z_root ~ stationary law; children z_{wi} = phi_i(z_w) + psi * eps (eps ~ N(0, I_d)), same phi at
every height; phi_i(z) = b_i + W_i sig(z) + U_i tanh(3 sig(z)), sig(z) = z / (1 + |z|) coordinatewise.
Oracle: belief propagation with the TRUE kernel on a regular grid over [-lim, lim]^d.  The Gaussian
transition is applied as a separable blur (child-side) plus multilinear interpolation at phi_i(grid)
(upward), and as a multilinear splat plus blur (for the stationary law).
"""
import math
import numpy as np
import torch
import torch.nn.functional as Fn

S = 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sig(z):
    return z / (1.0 + np.abs(z))


class NDKernel:
    def __init__(self, b, W, U, psi):
        self.b, self.W, self.U, self.psi = np.asarray(b), np.asarray(W), np.asarray(U), float(psi)
        self.d = self.b.shape[1]

    def phi(self, z):
        """z (..., d) -> (..., S, d)"""
        p = sig(z); t = np.tanh(3 * p)
        return np.stack([self.b[i] + p @ self.W[i].T + t @ self.U[i].T for i in range(S)], -2)


def random_kernel(d, rng, psi=0.35, sv_range=(1.3, 2.0)):
    b, W, U = [], [], []
    for _ in range(S):
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        sv = rng.uniform(*sv_range, d)
        Wi = Q @ np.diag(sv)
        Ui = rng.normal(size=(d, d)); Ui *= 0.3 * sv.min() / np.linalg.norm(Ui, 2)
        b.append(rng.uniform(-0.3, 0.3, d)); W.append(Wi); U.append(Ui)
    return NDKernel(np.array(b), np.array(W), np.array(U), psi)


def sample_trees(ker, U, L, rng, burn=40):
    d = ker.d
    z = np.zeros((U, d))
    for _ in range(burn):
        i = rng.integers(S, size=U)
        z = ker.phi(z)[np.arange(U), i] + ker.psi * rng.normal(size=(U, d))
    root = z.copy()
    z = z[:, None, :]
    for _ in range(L):
        m = ker.phi(z)                                             # (U, nodes, S, d)
        z = (m + ker.psi * rng.normal(size=m.shape)).reshape(U, -1, d)
    return z, root


class GridOracle:
    def __init__(self, ker, n=81, lim=4.0):
        self.ker, self.n, self.lim, self.d = ker, n, lim, ker.d
        axis = np.linspace(-lim, lim, n)
        mesh = np.stack(np.meshgrid(*([axis] * self.d), indexing="ij"), -1).reshape(-1, self.d)
        self.P = torch.tensor(mesh, dtype=torch.float32, device=DEV)            # (n^d, d)
        ph = ker.phi(mesh)                                                      # (n^d, S, d)
        self.phiP = [torch.tensor(ph[:, i], dtype=torch.float32, device=DEV) for i in range(S)]
        h = 2 * lim / (n - 1)
        r = int(math.ceil(4 * ker.psi / h))
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=DEV, dtype=torch.float32) * h / ker.psi) ** 2)
        self.k1 = (k / k.sum()).view(1, 1, -1)
        self.pi = self._stationary()

    # --- grid operators -------------------------------------------------------------
    def _blur(self, f):
        """f (N, n^d) -> Gaussian-blurred along every axis (zero padding: mass leaving the box is lost)."""
        N, n, d = f.shape[0], self.n, self.d
        g = f.view(N, *([n] * d))
        for ax in range(d):
            g = g.movedim(ax + 1, -1)
            sh = g.shape
            g = Fn.conv1d(g.reshape(-1, 1, n), self.k1, padding=self.k1.shape[-1] // 2).reshape(sh)
            g = g.movedim(-1, ax + 1)
        return g.reshape(N, -1)

    def _interp(self, f, pts):
        """f (N, n^d) grid values, pts (M, d) -> values at pts (N, M), multilinear, 0 outside."""
        N, n, d = f.shape[0], self.n, self.d
        inp = f.view(N, 1, *([n] * d))
        g = (pts / self.lim).flip(-1)                         # grid_sample wants (x=last axis, ...)
        g = g.view(1, *([1] * (d - 1)), -1, d).expand(N, *([1] * (d - 1)), -1, d)
        out = Fn.grid_sample(inp, g, mode="bilinear", padding_mode="zeros", align_corners=True)
        return out.reshape(N, -1)

    def _splat(self, w, pts):
        """Adjoint of _interp: put mass w (N, M) at locations pts (M, d) onto the grid (N, n^d)."""
        n, d = self.n, self.d
        u = (pts + self.lim) / (2 * self.lim) * (n - 1)
        lo = torch.floor(u).long(); fr = u - lo
        out = torch.zeros((w.shape[0], n ** d), device=DEV)
        for corner in range(2 ** d):
            off = torch.tensor([(corner >> k) & 1 for k in range(d)], device=DEV)
            idx = lo + off
            wt = torch.prod(torch.where(off.bool(), fr, 1 - fr), -1)
            ok = ((idx >= 0) & (idx < n)).all(-1)
            flat = torch.zeros_like(idx[:, 0])
            for k in range(d):
                flat = flat * n + idx[:, k].clamp(0, n - 1)
            out.index_add_(1, flat[ok], w[:, ok] * wt[ok])
        return out

    def _stationary(self, iters=300):
        pi = torch.full((1, self.n ** self.d), 1.0 / self.n ** self.d, device=DEV)
        for _ in range(iters):
            pi = sum(self._blur(self._splat(pi, self.phiP[i])) for i in range(S)) / S
            pi = pi / pi.sum()
        return pi[0] + 1e-30

    # --- inference --------------------------------------------------------------------
    def root_mean(self, X, chunk=10):
        """X (T, n_leaves, d) numpy -> posterior mean of the root (T, d)."""
        out = []
        for a in range(0, X.shape[0], chunk):
            Xc = torch.tensor(X[a:a + chunk], dtype=torch.float32, device=DEV)
            T, nl, d = Xc.shape
            L = int(round(math.log(nl) / math.log(S)))
            x = Xc.reshape(-1, S, d)                                               # height-1 children
            lb = torch.zeros((x.shape[0], self.P.shape[0]), device=DEV)
            for i in range(S):
                lb += -((x[:, i, None, :] - self.phiP[i][None]) ** 2).sum(-1) / (2 * self.ker.psi ** 2)
            b = torch.exp(lb - lb.max(1, keepdim=True).values)
            u = b / b.sum(1, keepdim=True)
            for h in range(2, L + 1):
                uc = self._blur(u).view(-1, S, u.shape[1])
                b = torch.ones((uc.shape[0], u.shape[1]), device=DEV)
                for i in range(S):
                    m = self._interp(uc[:, i], self.phiP[i]).clamp_min(0)
                    b = b * (m / (m.max(1, keepdim=True).values + 1e-30))
                u = (b + 1e-30) / (b + 1e-30).sum(1, keepdim=True)
            post = u * self.pi
            post = post / post.sum(1, keepdim=True)
            out.append((post @ self.P).cpu().numpy())
        return np.concatenate(out)
