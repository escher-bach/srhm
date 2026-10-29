"""Correctness checks for imtree.py (run locally, CPU, float64):
1. pyramid <-> image round trip, and the change-of-variables constant (log|det J| = -3 log2 per child);
2. exact sum-product vs brute-force enumeration of all type configurations (log Z, root/leaf/internal
   marginals, pair counts sum = #edges);
3. Fisher identity: gradient of -Q (grid-local statistics) == autodiff gradient of -log p;
4. sampler: empirical mean of log p under own samples ~ -entropy is finite; untied runs.
5. semantic top (srhm.tex Sec. semtop): partial tying cstar, label leaf (brute force, Fisher identity with
   mixed labelled/unlabelled batches), state grids, kappa, and the label-reach bound (Thm. reach) at every node.
"""
import itertools, math
import torch
import imtree as it

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, info)


# 1. round trip and Jacobian
y = torch.rand(3, 16, 16, 3)
P = it.pyramid(y)
y2 = it.unpyramid(P["tR"], P["det"], P["L"])
check("roundtrip", torch.allclose(y, y2, atol=1e-12), float((y - y2).abs().max()))
y1 = torch.rand(1, 8, 8, 3)
f = lambda v: torch.cat([it.pyramid(v.reshape(1, 8, 8, 3))["tR"].ravel()] +
                        [it.pyramid(v.reshape(1, 8, 8, 3))["det"][c].ravel() for c in range(2)])
J = torch.autograd.functional.jacobian(f, y1.ravel())
ld = torch.linalg.slogdet(J)[1]
check("jacobian", abs(float(ld) + 20 * 3 * math.log(2)) < 1e-9, (float(ld), -60 * math.log(2)))


# 2. brute force
def brute(model, P, y=None):
    L, H = P["L"], model.H
    lls = {c: model.emis(c, P["tp"][c], P["det"][c])[0] for c in range(L)}
    lr = model.root_emis(P["tR"])[0] + model.logpi()
    if y is not None:
        lr = lr + model.logl()[:, y]
    ns = [P["det"][c].shape[1] for c in range(L)]
    total = 1 + sum(ns)
    C = torch.cartesian_prod(*[torch.arange(H)] * total)               # (H^total, total)
    logs = lr[C[:, 0]]
    idx, par = 1, C[:, 0:1]
    for c in range(L - 1, -1, -1):
        tc = C[:, idx: idx + ns[c]]; idx += ns[c]
        lT = model.logT(c)
        for k in range(ns[c]):
            logs = logs + lT[k % 4, par[:, k // 4], tc[:, k]] + lls[c][k, tc[:, k]]
        par = tc
    logZ = torch.logsumexp(logs, 0)
    w = torch.exp(logs - logZ)
    qR = torch.stack([w[C[:, 0] == h].sum() for h in range(H)])
    q0 = torch.stack([w[C[:, -1] == h].sum() for h in range(H)])        # last leaf
    return logZ, qR, q0


for (L, H) in [(1, 3), (2, 2)]:
    for untied in (False, True):
        m = it.ImageTree(H, L=L, hid=5, untied=untied, seed=1).double()
        with torch.no_grad():
            m.Tlog.normal_(); m.pilog.normal_(); m.b.mul_(20)
        S = 2 ** (L + 1)
        P = it.pyramid(torch.rand(1, S, S, 3))
        lls = {c: m.emis(c, P["tp"][c], P["det"][c]) for c in range(L)}
        q, qR, xi, logZ = m.posteriors(P, lls)
        bZ, bqR, bq0 = brute(m, P)
        check(f"logZ L={L} H={H} untied={untied}", abs(float(logZ[0] - bZ)) < 1e-9, (float(logZ[0]), float(bZ)))
        check(f"root marginal L={L}", torch.allclose(qR[0], bqR, atol=1e-9))
        check(f"leaf marginal L={L}", torch.allclose(q[0][0, -1], bq0, atol=1e-9))
        nedges = sum(P["det"][c].shape[1] for c in range(L))
        check(f"pair mass = #edges L={L}", abs(float(sum(x.sum() for x in xi.values())) - nedges) < 1e-9)
        # pair marginals consistent with child marginals
        cons = all(torch.allclose(xi[c].sum(1).sum(0), q[c][0].sum(0), atol=1e-9) for c in range(L))
        check(f"pair/child consistency L={L}", cons)

# 3. Fisher identity
m = it.ImageTree(4, L=3, hid=6, seed=2).double()
P = it.pyramid(torch.rand(2, 16, 16, 3))
negQ, lp = m.q_objective(P)
g1 = torch.autograd.grad(negQ, list(m.parameters()), allow_unused=True)
lp2 = m.logp_image(P).mean()
g2 = torch.autograd.grad(-lp2, list(m.parameters()), allow_unused=True)
err = max(float((a - b).abs().max()) for a, b in zip(g1, g2) if a is not None)
scale = max(float(b.abs().max()) for b in g2 if b is not None)
check("fisher identity", err < 1e-8 * max(scale, 1), (err, scale))
check("logp consistent", abs(float(lp.mean() - lp2)) < 1e-9)

# 4. sampler + truncated likelihoods
x, dets, tR = m.sample(5, gen=torch.Generator().manual_seed(0))
P = it.pyramid(x)
check("sample pyramid consistent", torch.allclose(P["tR"], tR, atol=1e-10) and
      all(torch.allclose(P["det"][c], dets[c], atol=1e-10) for c in range(3)))
lps = [float(m.logp_image(P, cmin=c).mean()) for c in range(3)]
print("truncated logp", lps)

# 5a. partial tying: cstar=1 / L+1 reproduce untied / tied exactly; lvl map; brute force for cstar in between
for untied, cs in ((True, 1), (False, 4)):
    m1 = it.ImageTree(4, L=3, hid=6, seed=2, untied=untied).double()
    m2 = it.ImageTree(4, L=3, hid=6, seed=2, cstar=cs).double()
    same = all(torch.equal(a, b) for a, b in zip(m1.parameters(), m2.parameters()))
    Pq = it.pyramid(torch.rand(2, 16, 16, 3))
    check(f"cstar={cs} == untied={untied}", same and torch.allclose(m1.logp_image(Pq), m2.logp_image(Pq)))
m = it.ImageTree(3, L=3, hid=5, seed=3, cstar=3)
check("lvl map cstar=3, L=3", [m.lvl(c) for c in range(3)] == [0, 0, 1] and m.nl == 2, [m.lvl(c) for c in range(3)])
m = it.ImageTree(3, L=4, hid=5, seed=3, cstar=4)
check("lvl map cstar=4, L=4", [m.lvl(c) for c in range(4)] == [0, 0, 0, 1] and m.nl == 2)

# 5b. label leaf: brute force, posteriors, Fisher identity on a mixed batch
for cs in (2, 3):
    m = it.ImageTree(2, L=2, hid=5, seed=4, cstar=cs, C=3).double()
    with torch.no_grad():
        m.Tlog.normal_(); m.pilog.normal_(); m.b.mul_(20); m.ylog.normal_(0, 2)
    P = it.pyramid(torch.rand(1, 8, 8, 3))
    lls = {c: m.emis(c, P["tp"][c], P["det"][c]) for c in range(2)}
    for yv in (0, 2):
        y = torch.tensor([yv])
        q, qR, xi, logZ = m.posteriors(P, lls, y=y)
        bZ, bqR, bq0 = brute(m, P, y=yv)
        check(f"label logZ cstar={cs} y={yv}", abs(float(logZ[0] - bZ)) < 1e-9, (float(logZ[0]), float(bZ)))
        check(f"label root/leaf marginals cstar={cs} y={yv}",
              torch.allclose(qR[0], bqR, atol=1e-9) and torch.allclose(q[0][0, -1], bq0, atol=1e-9))
    q, qR, xi, logZ = m.posteriors(P, lls, y=torch.tensor([-1]))
    bZ, _, _ = brute(m, P)
    check(f"unlabelled (y=-1) == no label cstar={cs}", abs(float(logZ[0] - bZ)) < 1e-9)
m = it.ImageTree(4, L=3, hid=6, seed=5, cstar=3, C=10).double()
with torch.no_grad():
    m.ylog.normal_(0, 1)
P = it.pyramid(torch.rand(4, 16, 16, 3)); y = torch.tensor([3, -1, 7, -1])
negQ, lp = m.q_objective(P, y=y)
g1 = torch.autograd.grad(negQ, list(m.parameters()))
g2 = torch.autograd.grad(-m.logp_image(P, y=y).mean(), list(m.parameters()))
err = max(float((a - b).abs().max()) for a, b in zip(g1, g2))
check("fisher identity, joint likelihood, mixed batch", err < 1e-8 * max(float(max(b.abs().max() for b in g2)), 1), err)
check("ylog gets gradient", float(g1[-1].abs().max()) > 0)

# 5c. state grid messages == the upward pass's incoming messages; thumbnails == block averages
P = it.pyramid(torch.rand(2, 16, 16, 3))
lls = {c: m.emis(c, P["tp"][c], P["det"][c]) for c in range(3)}
a, msgs, aR, logZ = m.upward(P, lls)
okg = True
for c in (1, 2):
    th, inc = m.state_grid(P, c)
    okg &= torch.allclose(inc, torch.log_softmax(a[c] - lls[c], -1), atol=1e-10)
th1, _ = m.state_grid(P, 2)                                               # 4 nodes at height 2 (Morton order)
tR_from_grid = th1.reshape(2, 4, 4, 3).mean(2).reshape(2, 12)             # node i's mean colour = root pixel i
check("state grid messages", okg)
check("state grid thumbnails average to the root thumbnail", torch.allclose(tR_from_grid, P["tR"], atol=1e-12))

# 5d. kappa against a direct computation
m = it.ImageTree(3, L=2, hid=4, seed=6, cstar=2)
with torch.no_grad():
    m.Tlog.normal_()
kap = m.kappa()
lT = torch.log_softmax(m.Tlog, -1)
ref = torch.zeros(m.nl, 4)
for k in range(m.nl):
    for i in range(4):
        D = max(float(lT[k, i, a, b] + lT[k, i, a2, b2] - lT[k, i, a, b2] - lT[k, i, a2, b])
                for a in range(3) for a2 in range(3) for b in range(3) for b2 in range(3))
        ref[k, i] = math.tanh(D / 4)
check("kappa", torch.allclose(kap, ref, atol=1e-6))

# 5e. Theorem reach: d_H(q^y_v, q_v) <= lambda_y * prod kappa along the path, TV bounds, at every node
def dH(p, q):
    r = torch.log(p) - torch.log(q)
    return r.max(-1).values - r.min(-1).values
viol, tight = 0, 0.0
for seed in range(6):
    m = it.ImageTree(3, L=3, hid=5, seed=seed, cstar=2, C=4).double()
    with torch.no_grad():
        m.Tlog.normal_(0, 0.7); m.ylog.normal_(0, 1.5); m.b.mul_(5)
    P = it.pyramid(torch.rand(3, 16, 16, 3))
    lls = {c: m.emis(c, P["tp"][c], P["det"][c]) for c in range(3)}
    for yv in range(4):
        y = torch.full((3,), yv)
        q0, qR0, _, _ = m.posteriors(P, lls)
        q1, qR1, _, _ = m.posteriors(P, lls, y=y)
        lam = float(m.logl()[:, yv].max() - m.logl()[:, yv].min())
        kap = m.kappa()
        viol += int(float(dH(qR1, qR0).max()) > lam + 1e-9)
        for c in range(3):
            # path from the root to a node at height c: parents at heights L..c+1, positions from the Morton index
            n = q0[c].shape[1]
            for k in range(n):
                D, idx = lam, k
                for cc in range(c, 3):
                    D *= float(kap[m.lvl(cc), idx % 4]); idx //= 4
                d = float(dH(q1[c][:, k], q0[c][:, k]).max())
                tv = float(0.5 * (q1[c][:, k] - q0[c][:, k]).abs().sum(-1).max())
                viol += int(d > D + 1e-9) + int(tv > math.tanh(D / 4) + 1e-9)
                tight = max(tight, d / max(D, 1e-300))
check("label reach bound at every node (Thm reach)", viol == 0, f"violations={viol}, max ratio d/D={tight:.3f}")
print("ALL OK" if ok else "SOME FAILED")
