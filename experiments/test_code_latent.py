"""Checks for code_latent.py.
 1. E-step against brute-force enumeration of all hidden assignments on small random trees (log-likelihood, expected
    counts of P, Q, pi), with random parameters, H = 3, both groupings.
 2. H = 1 on the real corpus reproduces code_claim1.py's 'partial' model: 1.72930 nats per test unit.
 3. EM never decreases the training likelihood, and a split preserves it approximately (eps -> 0: exactly).
"""
import itertools, math, sys, os
import numpy as np
import torch
from code_latent import Corpus, LatentTree

rng = np.random.default_rng(0)
STOP = 5


def random_corpus(n_files=3, C=6, T=4, max_internal=4):
    uf, ut, ux, udp, upar, unode, nd, ni, nf = [], [], [], [], [], [], [], [], []
    for f in range(n_files):
        root = len(nd); nd.append(0); ni.append(1); nf.append(f)
        todo, n_int = [root], 1
        while todo:
            p = todo.pop(0)
            for _ in range(rng.integers(1, 4)):
                r = rng.random()
                uf.append(f); ut.append(int(rng.integers(T))); udp.append(nd[p]); upar.append(p)
                if r < 0.2:
                    ux.append(STOP); unode.append(-1); continue
                ux.append(int(rng.integers(STOP)))
                cid = len(nd); nd.append(nd[p] + 1); nf.append(f); unode.append(cid)
                if r < 0.55 and n_int < max_internal:
                    ni.append(1); n_int += 1; todo.append(cid)
                else:
                    ni.append(0)
            if not todo and n_int < 2:            # make sure the root has internal structure sometimes
                pass
    # nodes marked internal must have units: those appended to todo always get >= 1 unit above
    A = lambda v, t: np.array(v, t)
    return dict(ufile=A(uf, np.int32), t=A(ut, np.int16), x=A(ux, np.int16), dp=A(udp, np.int16),
                par=A(upar, np.int32), node=A(unode, np.int32), node_depth=A(nd, np.int16),
                node_internal=A(ni, np.int8), node_file=A(nf, np.int32), symbols=np.array([str(i) for i in range(C)]))


def brute(d, m, grouping):
    G = {"partial": lambda dp: min(dp, 3), "tied": lambda dp: 0}[grouping]
    P, Q, pi = [np.exp(v.double().cpu().numpy()) for v in (m.logP, m.logQ, m.logpi)]
    cP, cQ, cpi, ll = np.zeros_like(P), np.zeros_like(Q), np.zeros_like(pi), 0.0
    for f in np.unique(d["node_file"]):
        nodes = [i for i in np.nonzero(d["node_file"] == f)[0] if d["node_internal"][i] == 1]
        units = np.nonzero(d["ufile"] == f)[0]
        root = [i for i in nodes if d["node_depth"][i] == 0][0]
        Zf, cPf, cQf, cpif = 0.0, np.zeros_like(P), np.zeros_like(Q), np.zeros_like(pi)
        for assign in itertools.product(range(m.H), repeat=len(nodes)):
            h = dict(zip(nodes, assign))
            w = pi[h[root]]
            for u in units:
                g, t, x, p, c = G(int(d["dp"][u])), d["t"][u], d["x"][u], d["par"][u], d["node"][u]
                w *= P[g, t, x, h[p]]
                if c >= 0 and d["node_internal"][c]:
                    w *= Q[g, x, h[p], h[c]]
            Zf += w; cpif[h[root]] += w
            for u in units:
                g, t, x, p, c = G(int(d["dp"][u])), d["t"][u], d["x"][u], d["par"][u], d["node"][u]
                cPf[g, t, x, h[p]] += w
                if c >= 0 and d["node_internal"][c]:
                    cQf[g, x, h[p], h[c]] += w
        ll += math.log(Zf); cP += cPf / Zf; cQ += cQf / Zf; cpi += cpif / Zf
    return ll, cP, cQ, cpi


def test_brute():
    torch.set_default_dtype(torch.float64)
    for grouping in ("partial", "tied"):
        for trial in range(4):
            d = random_corpus()
            c = Corpus(d, np.unique(d["ufile"]), "cpu", grouping)
            m = LatentTree(4, 4, 6, 3, "cpu")
            m.logP = torch.log(torch.as_tensor(rng.dirichlet(np.ones(6), size=(4, 4, 3))).permute(0, 1, 3, 2).contiguous())
            m.logQ = torch.log(torch.as_tensor(rng.dirichlet(np.ones(3), size=(4, 6, 3))))
            m.logpi = torch.log(torch.as_tensor(rng.dirichlet(np.ones(3))))
            cP, cQ, cpi, ll = m.estep(c)
            bll, bP, bQ, bpi = brute(d, m, grouping)
            errs = [abs(ll - bll), np.abs(cP.numpy() - bP).max(), np.abs(cQ.numpy() - bQ).max(), np.abs(cpi.numpy() - bpi).max()]
            print(f"brute {grouping} trial {trial}: units {c.n_units} ll {ll:.6f} vs {bll:.6f}; max errs {['%.1e' % e for e in errs]}")
            assert max(errs) < 1e-9, errs
            # pair-weight sums: every internal non-root child contributes exactly 1 to cQ, every unit exactly 1 to cP
            assert abs(cP.sum().item() - c.n_units) < 1e-9
            assert abs(cQ.sum().item() - int(c.internal.sum())) < 1e-9
    torch.set_default_dtype(torch.float32)


def test_real_h1():
    import astdata
    d = astdata.load()
    test, pool = astdata.split(d)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, te = Corpus(d, pool, dev), Corpus(d, test, dev)
    m = LatentTree(4, len(d["contexts"]), len(d["symbols"]), 1, dev)
    m.set_marginal(tr)
    hist = m.em(tr, 3)
    nats = -m.loglik(te) / te.n_units
    print(f"H=1 real: test nats/unit {nats:.5f} (code_claim1 partial: 1.72930); train ll history {hist}")
    assert abs(nats - 1.72930) < 2e-5
    rng2 = np.random.default_rng(1)
    for H in (2, 4):
        before = m.loglik(tr)
        m.split(rng2, eps=1e-4)
        after = m.loglik(tr)
        h = m.em(tr, 4)
        print(f"split to H={m.H}: train ll {before:.1f} -> {after:.1f} after split; EM {['%.1f' % v for v in h]}")
        assert abs(after - before) < 1e-3 * abs(before)
        assert all(b >= a - 1e-4 * abs(a) for a, b in zip(h, h[1:]))


if __name__ == "__main__":
    test_brute()
    if "--real" in sys.argv:
        test_real_h1()
    print("ALL OK")
