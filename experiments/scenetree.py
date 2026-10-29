"""Scene grammar SG(H, M) with K views (srhm.tex Def. sg, Sec. views).

A scene type z in {0..M-1} ~ pi is the common parent of K views; each view is a whole image tree whose nodes have
state (t, h, z), z copied down (parent-determined, hidden).  Given z a view is IT(H) (imtree.ImageTree) with root
type prior pi_z(h) and tied type kernel T_z; emissions (details given parent thumbnail and type; root thumbnail
given root type) are shared by all z.  Optional label leaf l(y | z) at the scene node.  Exact sum-product: per z,
the type passes of imtree, batched over z.  Learning: generalized EM (grid-local expected statistics under the exact
posterior; Fisher identity), zero labels unless y is given.

Images of a batch are scene-major: B = S * K, image s*K + k is view k of scene s.
"""
import math
import torch
import imtree as it

LOG2 = it.LOG2


class SceneGrammar(torch.nn.Module):
    def __init__(self, H, M, L=4, hid=32, seed=0, C=0, zeps=0.005):
        super().__init__()
        self.base = it.ImageTree(H, L=L, hid=hid, seed=seed)          # shared emissions
        self.base.Tlog.requires_grad_(False); self.base.pilog.requires_grad_(False)   # replaced by T_z, pi_z
        g = torch.Generator().manual_seed(seed + 1000)
        self.H, self.M, self.L, self.C = H, M, L, C
        # near-symmetric start: a shared kernel plus a small per-z perturbation, so that every scene type starts
        # with responsibility ~1/M (a mixture component that starts with none never receives a gradient)
        shared = 2.0 * torch.eye(H).expand(4, H, H) + 0.3 * torch.randn(4, H, H, generator=g)
        self.TlogZ = torch.nn.Parameter(shared.expand(M, 4, H, H).clone() + zeps * torch.randn(M, 4, H, H, generator=g))
        self.pilogZ = torch.nn.Parameter(zeps * torch.randn(M, H, generator=g))
        self.zlog = torch.nn.Parameter(torch.zeros(M))
        if C > 0:
            self.ylog = torch.nn.Parameter(torch.zeros(M, C))

    # -- pieces
    def logTz(self):
        return torch.log_softmax(self.TlogZ, -1)                        # (M, 4, H, H) [z, i, parent, child]

    def logpiZ(self):
        return torch.log_softmax(self.pilogZ, -1)                       # (M, H)

    def logpz_prior(self):
        return torch.log_softmax(self.zlog, 0)

    def logl(self):
        return torch.log_softmax(self.ylog, -1)                         # (M, C)

    def emissions(self, P):
        lls = {c: self.base.emis(c, P["tp"][c], P["det"][c]) for c in range(P["L"])}
        return lls, self.base.root_emis(P["tR"])

    # -- exact sum-product over types, for every z
    def upward(self, lls, lr):
        """lls[c] (B, n_c, H), lr (B, H) -> a[c], msgs[c] (B, M, n_c, H), aR (B, M, H), log p_z(view) (B, M)
        (without the Jacobian constant)."""
        T = torch.exp(self.logTz())
        a, msgs, inc = {}, {}, None
        for c in range(len(lls)):
            ac = lls[c][:, None] if inc is None else lls[c][:, None] + inc
            if ac.shape[1] == 1:
                ac = ac.expand(-1, self.M, -1, -1)
            a[c] = ac
            B, M, n, H = ac.shape
            mx = ac.max(-1, keepdim=True).values
            e = torch.exp(ac - mx).reshape(B, M, n // 4, 4, H)
            m = torch.log(torch.einsum("zixy,bzpiy->bzpix", T, e) + 1e-38) + mx.reshape(B, M, n // 4, 4, 1)
            msgs[c] = m.reshape(B, M, n, H)
            inc = m.sum(3)
        aR = lr[:, None, :] + inc[:, :, 0]
        return a, msgs, aR, torch.logsumexp(self.logpiZ()[None] + aR, -1)

    def njac(self, P):
        return sum(P["det"][c].shape[1] for c in range(P["L"])) * 3 * LOG2

    def scene_logits(self, logpz, K, y=None):
        """(S, M): log pi(z) + sum_k log p_z(view k) [+ log l(y|z) for labelled scenes]."""
        S = logpz.shape[0] // K
        Lz = self.logpz_prior()[None] + logpz.reshape(S, K, self.M).sum(1)
        if y is not None:
            lab = self.logl()[:, y.clamp(min=0)].T
            Lz = Lz + lab * (y >= 0).to(lab.dtype)[:, None]
        return Lz

    def logp_scenes(self, P, K, y=None):
        """log p(view_1..view_K [, y]) per scene (S,), with the Jacobian constant."""
        lls, lr = self.emissions(P)
        logpz = self.upward(lls, lr)[3]
        return torch.logsumexp(self.scene_logits(logpz, K, y), -1) - K * self.njac(P)

    @torch.no_grad()
    def z_posterior(self, P):
        """Single-view posterior of the scene type, log w(z) (B, M): the scene state read from one image."""
        lls, lr = self.emissions(P)
        logpz = self.upward(lls, lr)[3]
        return torch.log_softmax(self.logpz_prior()[None] + logpz, -1)

    # -- learning
    def q_objective(self, P, K, y=None):
        """-Q per scene (expected complete-data log-lik under the exact posterior; its gradient equals that of
        log p(views [, y]) by the Fisher identity), and log p per scene."""
        lls, lr = self.emissions(P)
        with torch.no_grad():
            ld = {c: v.detach() for c, v in lls.items()}
            a, msgs, aR, logpz = self.upward(ld, lr.detach())
            Lz = self.scene_logits(logpz, K, y)
            W = torch.softmax(Lz, -1)                                    # (S, M) scene posterior
            logp = torch.logsumexp(Lz, -1) - K * self.njac(P)
            Wb = W.repeat_interleave(K, 0)                                # (B, M) for each view
            T = torch.exp(self.logTz())
            logpiZ = self.logpiZ()
            qR = torch.softmax(logpiZ[None] + aR, -1)                     # (B, M, H)
            qRbar = (Wb[..., None] * qR).sum(1)                           # (B, H)
            cR = (Wb[..., None] * qR).sum(0)                              # (M, H)
            D_par = (logpiZ[None] + aR)[:, :, None, :]
            qbar, xi = {}, {}
            for c in range(P["L"] - 1, -1, -1):
                B, M, n, H = a[c].shape
                r = D_par.repeat_interleave(4, 2) - msgs[c]
                rm = r.max(-1, keepdim=True).values
                A = torch.exp(r - rm).reshape(B, M, n // 4, 4, H)
                am = a[c].max(-1, keepdim=True).values
                Bv = torch.exp(a[c] - am).reshape(B, M, n // 4, 4, H)
                TB = torch.einsum("zixy,bzpiy->bzpix", T, Bv)
                Z = (A * TB).sum(-1, keepdim=True)
                xi[c] = T * torch.einsum("bz,bzpix,bzpiy->zixy", Wb, A / Z, Bv)
                down = torch.log(torch.einsum("bzpix,zixy->bzpiy", A, T) + 1e-38).reshape(B, M, n, H) + rm
                tot = down + a[c]
                qbar[c] = (Wb[..., None, None] * torch.softmax(tot, -1)).sum(1)   # (B, n, H)
                D_par = tot
        Q = (W * self.logpz_prior()[None]).sum() + (cR * self.logpiZ()).sum() + (qRbar * lr).sum()
        lTz = self.logTz()
        for c in range(P["L"]):
            Q = Q + (qbar[c] * lls[c]).sum() + (xi[c] * lTz).sum()
        if y is not None:
            lab = self.logl()[:, y.clamp(min=0)].T * (y >= 0).float()[:, None]
            Q = Q + (W * lab).sum()
        return -Q / W.shape[0], logp


def nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
