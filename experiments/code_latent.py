"""Latent tied tree model of Python ASTs (where.tex sec. prog-latent), fitted by EM with zero labels.

Every node n carries a hidden type h_n in {0..H-1}.  A child slot u of parent p, with context t_u and parent-depth
group g_u, draws its symbol and (if the child is an internal node) the child's type:
    P(x_u | g_u, t_u, h_p) * Q(h_u | g_u, x_u, h_p)
<STOP>/<NONE> slots and leaf children draw only the symbol (a leaf's type has no observable consequence).
Roots draw h from pi.  Groups: 'partial' = parent depth 0, 1, 2 own rules, >= 3 one tied rule; 'tied' = one rule.
H = 1 is exactly the observed-rule model of code_claim1.py.

E-step: one upward pass (log messages, level by level) and one downward pass, both exact (sum-product on a tree).
M-step: closed form, with the smoothing of code_claim1.py for P and a small Dirichlet for Q.
"""
import math
import numpy as np
import torch

BANDS_DP_GROUP = {"partial": lambda dp: np.minimum(dp, 3), "tied": lambda dp: np.zeros_like(dp)}


class Corpus:
    """the units and nodes of a set of files, re-indexed contiguously, as torch tensors on `dev`"""
    def __init__(self, d, files, dev, grouping="partial"):
        um = np.isin(d["ufile"], files); nm = np.isin(d["node_file"], files)
        gidx = np.nonzero(nm)[0]
        loc = np.full(len(d["node_depth"]), -1, np.int64); loc[gidx] = np.arange(len(gidx))
        node = d["node"][um].astype(np.int64)
        node_l = np.where(node >= 0, loc[np.maximum(node, 0)], -1)
        internal = np.where(node >= 0, d["node_internal"][np.maximum(node, 0)], 0).astype(bool)
        dp = d["dp"][um].astype(np.int64)
        ndepth = d["node_depth"][nm]
        roots = np.nonzero(ndepth == 0)[0]
        nfile = d["node_file"][nm]
        rootpos = {f: i for i, f in enumerate(nfile[roots])}
        T = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=dev)
        self.n_units, self.n_nodes, self.n_files = int(um.sum()), int(nm.sum()), len(roots)
        self.t, self.x = T(d["t"][um].astype(np.int64)), T(d["x"][um].astype(np.int64))
        self.g = T(BANDS_DP_GROUP[grouping](dp).astype(np.int64))
        self.par, self.node = T(loc[d["par"][um]]), T(node_l)
        self.internal = T(internal)
        self.roots = T(roots.astype(np.int64))
        self.node_fidx = T(np.array([rootpos[f] for f in nfile], np.int64))
        self.lev = [T(np.nonzero(dp == k)[0]) for k in range(int(dp.max()) + 1)]
        self.xcount = np.bincount(d["x"][um], minlength=len(d["symbols"]))


class LatentTree:
    def __init__(self, G, T, C, H, dev, a=1.0, b=0.01):
        self.G, self.T, self.C, self.H, self.dev, self.a, self.b = G, T, C, H, dev, a, b
        self.logP = torch.full((G, T, C, H), -math.log(C), device=dev)
        self.logQ = torch.full((G, C, H, H), -math.log(H), device=dev)
        self.logpi = torch.full((H,), -math.log(H), device=dev)
        self.m = None                                             # training marginal of x, for smoothing
        self.max_elems = 1.5e8                                    # cap on the (m, H, H) blocks of the passes

    def chunk(self):
        return max(1, int(self.max_elems // (self.H * self.H)))

    # ------------------------------------------------------------------ inference
    def upward(self, c):
        logb = torch.zeros(c.n_nodes, self.H, device=self.dev)
        LM = torch.empty(c.n_units, self.H, device=self.dev)
        for u in reversed(c.lev):
            if len(u) == 0: continue
            g, t, x = c.g[u], c.t[u], c.x[u]
            lm = self.logP[g, t, x].clone()                          # (n, H) over the parent's type
            ii = torch.nonzero(c.internal[u])[:, 0]
            for j in ii.split(self.chunk()):                         # (m, H, H) blocks, bounded memory
                q = self.logQ[g[j], x[j]]
                lm[j] = lm[j] + torch.logsumexp(q + logb[c.node[u[j]]][:, None, :], dim=2)
            LM[u] = lm
            logb.index_add_(0, c.par[u], lm)
        Z = torch.logsumexp(self.logpi + logb[c.roots], dim=1)       # log-likelihood per file
        return logb, LM, Z

    def loglik(self, c):
        return float(self.upward(c)[2].double().sum())

    def estep(self, c):
        logb, LM, Z = self.upward(c)
        Zn = Z[c.node_fidx]
        logo = torch.full((c.n_nodes, self.H), -math.inf, device=self.dev)
        logo[c.roots] = self.logpi
        cP = torch.zeros_like(self.logP); cQ = torch.zeros_like(self.logQ)
        for u in c.lev:
            if len(u) == 0: continue
            g, t, x, p = c.g[u], c.t[u], c.x[u], c.par[u]
            op = logo[p] + logb[p]                                     # parent: outside + all its children
            cP.index_put_((g, t, x), torch.exp(op - Zn[p][:, None]), accumulate=True)
            ii = torch.nonzero(c.internal[u])[:, 0]
            for j in ii.split(self.chunk()):
                uu = u[j]; gi, xi, pi_ = g[j], x[j], p[j]
                rest = op[j] - LM[uu] + self.logP[gi, t[j], xi]           # parent side, this child's symbol drawn
                q = self.logQ[gi, xi]
                pair = rest[:, :, None] + q + logb[c.node[uu]][:, None, :] - Zn[pi_][:, None, None]
                cQ.index_put_((gi, xi), torch.exp(pair), accumulate=True)
                logo[c.node[uu]] = torch.logsumexp(rest[:, :, None] + q, dim=1)
        cpi = torch.exp(self.logpi + logb[c.roots] - Z[:, None]).sum(0)
        return cP, cQ, cpi, float(Z.double().sum())

    # ------------------------------------------------------------------ learning
    def mstep(self, cP, cQ, cpi):
        m = self.m[None, None, :, None]
        self.logP = torch.log((cP + self.a * m) / (cP.sum(2, keepdim=True) + self.a))
        self.logQ = torch.log((cQ + self.b) / (cQ.sum(3, keepdim=True) + self.H * self.b))
        self.logpi = torch.log((cpi + 1e-3) / (cpi.sum() + self.H * 1e-3))

    def set_marginal(self, c):
        m = torch.as_tensor(c.xcount + 1.0, device=self.dev, dtype=torch.float32)
        self.m = m / m.sum()

    def em(self, c, iters, tol=1e-6, log=None):
        prev = -math.inf; hist = []
        for it in range(iters):
            cP, cQ, cpi, ll = self.estep(c)
            self.mstep(cP, cQ, cpi)
            hist.append(ll)
            if log: log(it=it, H=self.H, train_ll_per_unit=ll / c.n_units)
            if ll - prev < tol * abs(ll): break
            prev = ll
        return hist

    def split(self, rng, eps=0.1):
        """H -> 2H: duplicate every type, perturb the copies' symbol laws, split each child type's mass in two"""
        H2 = 2 * self.H
        noise = lambda shape: torch.as_tensor(rng.normal(0, eps, shape), device=self.dev, dtype=torch.float32)
        P = self.logP.repeat_interleave(2, dim=3) + noise(self.logP.shape[:3] + (H2,))
        self.logP = P - torch.logsumexp(P, dim=2, keepdim=True)
        Q = self.logQ.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3) - math.log(2)
        Q = Q + noise(Q.shape)
        self.logQ = Q - torch.logsumexp(Q, dim=3, keepdim=True)
        pi = self.logpi.repeat_interleave(2) - math.log(2)
        self.logpi = pi - torch.logsumexp(pi, 0)
        self.H = H2

    def state(self):
        return (self.logP.clone(), self.logQ.clone(), self.logpi.clone(), self.H)

    def load_state(self, s):
        self.logP, self.logQ, self.logpi, self.H = s[0].clone(), s[1].clone(), s[2].clone(), s[3]
