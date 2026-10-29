"""Image tree (srhm.tex section 10): the tied structured kernel on the thumbnail pyramid, with hidden
context types.  Exact sum-product, grid-local generalized EM, zero labels.

Representation (exact, invertible).  An image of side 2^(L+1) is a quadtree of height L.  The node at
height c owns a 2^(c+1)-square region; its STATE is (t, h): t = the 2x2x3 block-average thumbnail of its
region (observed: parent-determined, Thm mng(4)), h in {0..H-1} a hidden context type (Thm mng(1)-(3)).
A child at position i (0..3 = TL,TR,BL,BR) has mean colour t_parent[i]; what it adds is its orthonormal
Haar detail delta in R^9.  So the image <-> (root thumbnail, all details), and
    log p(y) = log p(t_R) + sum_children [ log p(delta) ] - n_children * 3 log 2.

Tied kernel (same at every height; `untied=True` gives every parent height its own copy; `cstar=c` unties
only the parents at heights >= c, the semantic top ST(H, c*) of srhm.tex Def. semtop):
    h'_i | h         ~ T_i[h, h']
    delta_i | t, h'  ~ N( A_{i,h'} g_i(t) + b_{i,h'},  (U_{i,h'}^T U_{i,h'})^{-1} ),
    g_i(t) = tanh(W_i f(t) + c_i)          (the nonlinear structured part: a small net in leaf coordinates)
Root:  h_R ~ pi,  t_R | h_R ~ N(m_h, (V_h^T V_h)^{-1}).
Leaves (height 0, 2x2 pixel patches) are generated exactly like every other child (design rule, Q4).
Label leaf (C > 0): y | h_R ~ l(y | h_R), a fifth child of the root; observed on labelled images (y >= 0),
summed out on the others (y = -1).  Learning is the same generalized EM on the joint likelihood.
"""
import math
import numpy as np
import torch

LOG2 = math.log(2.0)
ULMAX, VLMAX = math.log(2000.0), math.log(1000.0)   # noise floors (sd >= 5e-4, 1e-3): ML for
# mixtures is unbounded without a floor; dequantization noise alone has sd 1.1e-3 per Haar coefficient.


# ---------------------------------------------------------------- pyramid <-> image
def morton(g):
    """Permutation of a g x g grid (row-major index) into quadtree (Morton) order, row bit high."""
    r, c = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
    k = np.zeros_like(r)
    for b in range(int(math.log2(g)) if g > 1 else 0):
        k |= ((r >> b) & 1) << (2 * b + 1)
        k |= ((c >> b) & 1) << (2 * b)
    order = np.empty(g * g, dtype=np.int64)
    order[k.ravel()] = np.arange(g * g)
    return torch.tensor(order)


def blocks(I):
    """I (B, 2g, 2g, 3) -> (B, g*g [Morton], 2, 2, 3)"""
    B, S, _, C = I.shape
    g = S // 2
    X = I.reshape(B, g, 2, g, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, g * g, 2, 2, C)
    return X[:, morton(g).to(I.device)]


def haar(bl):
    a, b, c, d = bl[..., 0, 0, :], bl[..., 0, 1, :], bl[..., 1, 0, :], bl[..., 1, 1, :]
    return (a + b + c + d) / 4, torch.cat([(a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2], -1)


def ihaar(m, det):
    H, V, D = det[..., 0:3], det[..., 3:6], det[..., 6:9]
    a = m + (H + V + D) / 2; b = m + (-H + V - D) / 2; c = m + (H - V - D) / 2; d = m + (-H - V + D) / 2
    return torch.stack([torch.stack([a, b], -2), torch.stack([c, d], -2)], -3)


def pyramid(y):
    """y (B, S, S, 3), S = 2^(L+1) -> dict: tR (B,12); for child height c = 0..L-1:
    tp[c] (B, n_c, 12) parent thumbnails, det[c] (B, n_c, 9); child k's parent is node k//4, position k%4."""
    B, S = y.shape[0], y.shape[1]
    L = int(round(math.log2(S))) - 1
    levels = [y]
    for _ in range(L):
        I = levels[-1]
        levels.append(I.reshape(B, I.shape[1] // 2, 2, I.shape[2] // 2, 2, 3).mean((2, 4)))
    th = [blocks(I) for I in levels]                     # th[c]: (B, n_c, 2,2,3), node thumbnails at height c
    out = dict(L=L, tR=th[L].reshape(B, 12), tp={}, det={})
    for c in range(L):
        _, det = haar(th[c])
        out["det"][c] = det
        out["tp"][c] = th[c + 1].reshape(B, -1, 12).repeat_interleave(4, 1)
    return out


def unpyramid(tR, dets, L):
    """Inverse: root thumbnail (B,12) and dets[c] (B, n_c, 9) -> image (B, S, S, 3)."""
    B = tR.shape[0]
    th = tR.reshape(B, 1, 2, 2, 3)
    for c in range(L - 1, -1, -1):
        means = th.reshape(B, -1, 3)                     # pixel i of parent k -> child 4k+i (Morton)
        th = ihaar(means, dets[c])                       # (B, n_c, 2, 2, 3)
    g = int(round(math.sqrt(th.shape[1])))
    inv = torch.argsort(morton(g).to(th.device))
    X = th[:, inv].reshape(B, g, g, 2, 2, 3).permute(0, 1, 3, 2, 4, 5)
    return X.reshape(B, 2 * g, 2 * g, 3)


# ---------------------------------------------------------------- model
def _tri(n):
    return torch.tril_indices(n, n, -1)


class ImageTree(torch.nn.Module):
    def __init__(self, H, L=4, hid=32, untied=False, seed=0, det_sd=0.03, root_sd=0.2, cstar=None, C=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        r = lambda *s, sc=1.0: torch.nn.Parameter(torch.randn(*s, generator=g) * sc)
        cstar = (1 if untied else L + 1) if cstar is None else cstar
        assert 1 <= cstar <= L + 1
        self.H, self.L, self.hid, self.cstar, self.C = H, L, hid, cstar, C
        self.untied = cstar == 1
        nl = (1 if cstar > 1 else 0) + (L - cstar + 1)            # shared copy + one per untied height
        self.nl = nl
        self.W = r(nl, 4, hid, 12, sc=1.0 / math.sqrt(12))
        self.cb = r(nl, 4, hid, sc=0.1)
        self.A = r(nl, 4, H, 9, hid, sc=0.3 * det_sd / math.sqrt(hid))
        self.b = r(nl, 4, H, 9, sc=0.1 * det_sd)
        self.Uoff = torch.nn.Parameter(torch.zeros(nl, 4, H, 36))
        self.Ulog = torch.nn.Parameter(torch.full((nl, 4, H, 9), -math.log(det_sd))
                                       + 0.7 * torch.randn(nl, 4, H, 1, generator=g))   # types start as scales
        self.Tlog = torch.nn.Parameter(2.0 * torch.eye(H).expand(nl, 4, H, H).clone()
                                       + 0.3 * torch.randn(nl, 4, H, H, generator=g))
        self.pilog = torch.nn.Parameter(torch.zeros(H))
        self.m = r(H, 12, sc=root_sd); self.m.data += 0.47
        self.Voff = torch.nn.Parameter(torch.zeros(H, 66))
        self.Vlog = torch.nn.Parameter(torch.full((H, 12), -math.log(root_sd)))
        if C > 0:
            self.ylog = torch.nn.Parameter(torch.zeros(H, C))                    # label kernel, uniform start

    # -- pieces
    def lvl(self, c):                                    # parameter index for parents at height c+1
        ph = c + 1
        if ph < self.cstar:
            return 0
        return (1 if self.cstar > 1 else 0) + ph - self.cstar

    def logl(self):
        return torch.log_softmax(self.ylog, -1)                         # (H, C)

    def label_term(self, y):
        """(B, H): log l(y|h) for labelled images (y >= 0), 0 for unlabelled (y = -1)."""
        lab = self.logl()[:, y.clamp(min=0)].T
        return lab * (y >= 0).to(lab.dtype)[:, None]

    def logT(self, c):
        return torch.log_softmax(self.Tlog[self.lvl(c)], -1)          # (4, H, H) [i, parent, child]

    def logpi(self):
        return torch.log_softmax(self.pilog, 0)

    def U(self, c):
        k = self.lvl(c)
        U = torch.diag_embed(torch.exp(self.Ulog[k].clamp(max=ULMAX))).clone()          # (4, H, 9, 9), precision Cholesky
        ti = _tri(9).to(U.device)
        U[:, :, ti[0], ti[1]] = self.Uoff[k]
        return U

    def mean(self, c, tp):
        """tp (B, n, 12) parent thumbnails of children at height c -> means (B, n, H, 9)."""
        k = self.lvl(c)
        B, n, _ = tp.shape
        f = ((tp - 0.47) / 0.25).reshape(B, n // 4, 4, 12)
        gz = torch.tanh(torch.einsum("ihd,bpid->bpih", self.W[k], f) + self.cb[k])       # (B, n/4, 4, hid)
        mu = torch.einsum("ixeh,bpih->bpixe", self.A[k], gz) + self.b[k]              # (B, n/4, 4, H, 9)
        return mu.reshape(B, n, self.H, 9)

    def emis(self, c, tp, det):
        """log N(det | .) for every type: (B, n, H)."""
        B, n, _ = tp.shape
        mu = self.mean(c, tp)
        U = self.U(c)                                                   # (4, H, 9, 9)
        res = (det[:, :, None, :] - mu).reshape(B, n // 4, 4, self.H, 9)
        z = torch.einsum("ixjk,bpixk->bpixj", U, res)
        ld = self.Ulog[self.lvl(c)].clamp(max=ULMAX)                    # (4, H, 9)  log diag
        ll = -0.5 * (z ** 2).sum(-1) + ld.sum(-1) - 4.5 * math.log(2 * math.pi)
        return ll.reshape(B, n, self.H)

    def root_emis(self, tR):
        V = torch.diag_embed(torch.exp(self.Vlog.clamp(max=VLMAX))).clone()
        ti = _tri(12).to(V.device); V[:, ti[0], ti[1]] = self.Voff
        z = torch.einsum("xjk,bxk->bxj", V, tR[:, None, :] - self.m)
        return -0.5 * (z ** 2).sum(-1) + self.Vlog.clamp(max=VLMAX).sum(-1) - 6 * math.log(2 * math.pi)

    # -- exact sum-product over types
    def upward(self, P, lls, cmin=0, y=None):
        """lls[c] (B, n_c, H) emission log-lik.  Truncated at child height cmin (nodes below ignored).
        Returns a[c] (log up-message incl. own emission), msgs[c] (B, n_c, H) child->parent log messages,
        root log-message and log Z (B,)."""
        a, msgs = {}, {}
        inc = None
        for c in range(cmin, P["L"]):
            ac = lls[c] if inc is None else lls[c] + inc
            a[c] = ac
            B, n, H = ac.shape
            mx = ac.max(-1, keepdim=True).values
            e = torch.exp(ac - mx).reshape(B, n // 4, 4, H)
            T = torch.exp(self.logT(c))                                # (4, H, H)
            m = torch.log(torch.einsum("ixy,bpiy->bpix", T, e) + 1e-38) + mx.reshape(B, n // 4, 4, 1)
            msgs[c] = m.reshape(B, n, H)
            inc = m.sum(2)                                             # (B, n/4, H) incoming at parents
        aR = self.root_emis(P["tR"]) + inc[:, 0]
        if y is not None:
            aR = aR + self.label_term(y)
        logZ = torch.logsumexp(self.logpi() + aR, -1)
        return a, msgs, aR, logZ

    def posteriors(self, P, lls, y=None, pooled=True):
        """Exact node marginals q[c] (B, n_c, H), root qR (B, H), and pair counts xi[c]: pooled (4, H, H), or
        per node (B, n_c, H_parent, H_child) if pooled=False.  With y: conditioned on the observed labels."""
        a, msgs, aR, logZ = self.upward(P, lls, y=y)
        L = P["L"]
        qR = torch.softmax(self.logpi() + aR, -1)
        D_par = (self.logpi() + aR)[:, None, :]                       # parent (down + up) at the root
        q, xi = {}, {}
        for c in range(L - 1, -1, -1):
            B, n, H = a[c].shape
            full = D_par.repeat_interleave(4, 1)                       # parent's total log-belief per child
            r = full - msgs[c]                                         # exclude this child's own message
            logT = self.logT(c)
            rm = r.max(-1, keepdim=True).values
            A = torch.exp(r - rm).reshape(B, n // 4, 4, H)
            am = a[c].max(-1, keepdim=True).values
            Bv = torch.exp(a[c] - am).reshape(B, n // 4, 4, H)
            T = torch.exp(logT)
            TB = torch.einsum("ixy,bpiy->bpix", T, Bv)                 # (B, n/4, 4, H_parent)
            Z = (A * TB).sum(-1, keepdim=True)
            if pooled:
                xi[c] = T * torch.einsum("bpix,bpiy->ixy", A / Z, Bv)
            else:
                xi[c] = (T[None, None] * (A / Z)[..., :, None] * Bv[..., None, :]).reshape(B, n, H, H)
            down = torch.log(torch.einsum("bpix,ixy->bpiy", A, T) + 1e-38).reshape(B, n, H) + rm
            tot = down + a[c]
            q[c] = torch.softmax(tot, -1)
            D_par = tot
        return q, qR, xi, logZ

    def logp_image(self, P, lls=None, cmin=0, y=None):
        """log density of the pyramid truncated at child height cmin (cmin=0: the whole image); with y, the
        joint log p(image, y) for labelled entries."""
        if lls is None:
            lls = {c: self.emis(c, P["tp"][c], P["det"][c]) for c in range(cmin, P["L"])}
        logZ = self.upward(P, lls, cmin, y=y)[3]
        nchild = sum(P["det"][c].shape[1] for c in range(cmin, P["L"]))
        return logZ - nchild * 3 * LOG2

    # -- learning: grid-local statistics only
    def q_objective(self, P, y=None):
        """Expected complete-data log-lik under the exact posterior (Fisher identity: its gradient equals
        the gradient of the marginal log-likelihood).  Returns (-Q per image, log p per image).  With labels y
        (-1 = unlabelled) the likelihood is the joint one: log p(image, y) where y is observed."""
        lls = {c: self.emis(c, P["tp"][c], P["det"][c]) for c in range(P["L"])}
        with torch.no_grad():
            q, qR, xi, logZ = self.posteriors(P, {c: v.detach() for c, v in lls.items()}, y=y)
        Q = (qR * (self.logpi() + self.root_emis(P["tR"]))).sum()
        if y is not None:
            Q = Q + (qR * self.label_term(y)).sum()
        for c in range(P["L"]):
            Q = Q + (q[c] * lls[c]).sum() + (xi[c] * self.logT(c)).sum()
        B = P["tR"].shape[0]
        nchild = sum(P["det"][c].shape[1] for c in range(P["L"]))
        return -Q / B, (logZ - nchild * 3 * LOG2)

    # -- root state for the head
    @torch.no_grad()
    def root_state(self, P):
        lls = {c: self.emis(c, P["tp"][c], P["det"][c]) for c in range(P["L"])}
        a, msgs, aR, logZ = self.upward(P, lls)
        return torch.log_softmax(self.logpi() + aR, -1), P["tR"]

    @torch.no_grad()
    def state_grid(self, P, c):
        """The state grid G_c (srhm.tex Cor. reach, item 2): thumbnails (B, n_c, 12) of the nodes at height c and
        the log-messages they receive from their children (B, n_c, H), normalized; 1 <= c <= L-1."""
        lls = {k: self.emis(k, P["tp"][k], P["det"][k]) for k in range(c)}
        a, msgs = {}, {}
        inc = None
        for k in range(c):
            ak = lls[k] if inc is None else lls[k] + inc
            B, n, H = ak.shape
            mx = ak.max(-1, keepdim=True).values
            e = torch.exp(ak - mx).reshape(B, n // 4, 4, H)
            T = torch.exp(self.logT(k))
            m = torch.log(torch.einsum("ixy,bpiy->bpix", T, e) + 1e-38) + mx.reshape(B, n // 4, 4, 1)
            inc = m.sum(2)
        return P["tp"][c - 1][:, ::4], torch.log_softmax(inc, -1)

    def kappa(self):
        """Birkhoff contraction kappa(T) = tanh(Delta(T)/4) for every stored kernel copy and position: (nl, 4)."""
        lT = torch.log_softmax(self.Tlog, -1)                          # (nl, 4, H, H) [.., parent, child]
        # Delta = max over parents a,a' and children b,b' of lT[a,b]+lT[a',b']-lT[a,b']-lT[a',b]
        r = lT[..., :, None, :] - lT[..., None, :, :]                  # (nl, 4, a, a', b): lT[a,b]-lT[a',b]
        D = (r.max(-1).values + (-r).max(-1).values).flatten(-2).max(-1).values
        return torch.tanh(D / 4)

    # -- generation: ancestral sampling, top-down
    @torch.no_grad()
    def sample(self, B, gen=None):
        dev = self.m.device
        H, L = self.H, self.L
        hR = torch.multinomial(torch.exp(self.logpi()), B, replacement=True, generator=gen)
        V = torch.diag_embed(torch.exp(self.Vlog.clamp(max=VLMAX))).clone()
        ti = _tri(12).to(dev); V[:, ti[0], ti[1]] = self.Voff
        eps = torch.randn(B, 12, 1, device=dev, generator=gen)
        tR = self.m[hR] + torch.linalg.solve_triangular(V[hR], eps, upper=False).squeeze(-1)
        th, types, dets = tR.reshape(B, 1, 12), hR.reshape(B, 1), {}
        for c in range(L - 1, -1, -1):
            n = th.shape[1] * 4
            T = torch.exp(self.logT(c))                               # (4, H, H)
            pos = torch.arange(4, device=dev).repeat(th.shape[1])
            par = types.repeat_interleave(4, 1)                       # (B, n)
            probs = T[pos[None, :].expand(B, n), par]                 # (B, n, H)
            ch = torch.multinomial(probs.reshape(-1, H), 1, generator=gen).reshape(B, n)
            tp = th.repeat_interleave(4, 1)
            mu = self.mean(c, tp)                                     # (B, n, H, 9)
            mu = torch.gather(mu, 2, ch[:, :, None, None].expand(B, n, 1, 9)).squeeze(2)
            U = self.U(c)[pos[None, :].expand(B, n), ch]              # (B, n, 9, 9)
            eps = torch.randn(B, n, 9, 1, device=dev, generator=gen)
            det = mu + torch.linalg.solve_triangular(U, eps, upper=False).squeeze(-1)
            dets[c] = det
            means = th.reshape(B, -1, 3)
            th = ihaar(means, det).reshape(B, n, 12)
            types = ch
        return unpyramid(tR, dets, L), dets, tR


def nparams(model):
    return sum(p.numel() for p in model.parameters())
