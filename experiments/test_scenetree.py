"""Correctness checks for scenetree.py (CPU, float64):
1. log p(views) against brute-force enumeration of z and every type configuration of every view;
2. Fisher identity: gradient of -Q (grid-local expected statistics) == autodiff gradient of -log p, with and without
   labels (mixed labelled / unlabelled scenes), K = 1, 2, 3;
3. agreement identity (Prop. agree): log p(views) = sum_k log p(view k) + log sum_z pi prod_k (w_k / pi);
4. M = 1 reduces to imtree.ImageTree with the same kernel.
"""
import itertools, math
import torch
import imtree as it
import scenetree as st

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, info)


def brute_view(m, P, b, z):
    """log p_z(view b) by enumeration (L = 1: root + 4 children), without the Jacobian."""
    lls, lr = m.emissions(P)
    H = m.H
    lT = m.logTz()[z]; lpi = m.logpiZ()[z]
    tot = []
    for hr in range(H):
        for ch in itertools.product(range(H), repeat=4):
            v = lpi[hr] + lr[b, hr]
            for i, h in enumerate(ch):
                v = v + lT[i, hr, h] + lls[0][b, i, h]
            tot.append(v)
    return torch.logsumexp(torch.stack(tot), 0)


def randomize(m):
    with torch.no_grad():
        m.TlogZ.normal_(); m.pilogZ.normal_(); m.zlog.normal_(); m.base.b.mul_(20)
        if m.C:
            m.ylog.normal_(0, 1.5)


# 1. brute force, L = 1, H = 2, M = 3, K = 2
m = st.SceneGrammar(2, 3, L=1, hid=4, seed=1).double(); randomize(m)
K = 2
P = it.pyramid(torch.rand(2 * K, 4, 4, 3))
lp = m.logp_scenes(P, K)
ref = []
for s in range(2):
    ref.append(torch.logsumexp(torch.stack([m.logpz_prior()[z] + sum(brute_view(m, P, s * K + k, z) for k in range(K))
                                            for z in range(3)]), 0) - K * m.njac(P))
ref = torch.stack(ref)
check("brute force log p(views)", torch.allclose(lp, ref, atol=1e-9), (lp.tolist(), ref.tolist()))

# 2. Fisher identity
for K in (1, 2, 3):
    for lab in (False, True):
        m = st.SceneGrammar(3, 4, L=2, hid=5, seed=2, C=5 if lab else 0).double(); randomize(m)
        S = 3
        P = it.pyramid(torch.rand(S * K, 8, 8, 3))
        y = torch.tensor([2, -1, 4]) if lab else None
        negQ, lp = m.q_objective(P, K, y)
        params = [p for p in m.parameters() if p.requires_grad]
        g1 = torch.autograd.grad(negQ, params)
        lp2 = m.logp_scenes(P, K, y)
        g2 = torch.autograd.grad(-lp2.mean(), params)
        err = max(float((a - b).abs().max()) for a, b in zip(g1, g2))
        scale = max(float(b.abs().max()) for b in g2)
        check(f"fisher identity K={K} labels={lab}", err < 1e-8 * max(scale, 1), (err, scale))
        check(f"logp consistent K={K} labels={lab}", torch.allclose(lp, lp2.detach(), atol=1e-9))

# 3. agreement identity
m = st.SceneGrammar(3, 4, L=2, hid=5, seed=3).double(); randomize(m)
K = 3
P = it.pyramid(torch.rand(2 * K, 8, 8, 3))
joint = m.logp_scenes(P, K)
single = m.logp_scenes(P, 1).reshape(2, K)                             # each view as its own scene
lw = m.z_posterior(P).reshape(2, K, -1)
lpi = m.logpz_prior()
agree = torch.logsumexp(lpi[None] + (lw - lpi[None, None]).sum(1), -1)
check("agreement identity", torch.allclose(joint, single.sum(1) + agree, atol=1e-9))

# 4. M = 1 equals imtree with the same kernel and root prior
m = st.SceneGrammar(3, 1, L=2, hid=5, seed=4).double(); randomize(m)
b = m.base
with torch.no_grad():
    b.Tlog.copy_(m.TlogZ[0][None]); b.pilog.copy_(m.pilogZ[0])
P = it.pyramid(torch.rand(3, 8, 8, 3))
check("M=1 == ImageTree", torch.allclose(m.logp_scenes(P, 1), b.logp_image(P), atol=1e-9))
print("ALL OK" if ok else "SOME FAILED")
