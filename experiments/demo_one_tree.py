"""Thm 5.4 check: ONE unlabelled tree of depth L' yields n^(L'-1) conditionally independent
samples of the shared kernel.  Learn from a single tree, test at the root of depth-5 trees."""
import numpy as np
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import verify.demo_sshp as d

rng = np.random.default_rng(7)


def sample_depth(B, depth):
    z = rng.choice(d.k, size=(B, 1), p=d.pi)
    cum = np.cumsum(d.M, axis=0)
    for _ in range(depth):
        z = np.repeat(z, d.n, axis=1)
        z = (rng.random(z.shape)[..., None] > cum.T[z]).sum(-1)
    return z


Xte, yte = d.sample(5000)
print(f"oracle BP on depth-5 test set: {(d.bp(Xte, d.M, d.pi)[0].argmax(-1) == yte).mean():.3f}")
print("learn from ONE unlabelled tree of depth L' (zero labels), 5 seeds each")
for Lp in [5, 6, 7, 8, 9]:
    accs, errs, pinned = [], [], []
    for s in range(5):
        tree = sample_depth(1, Lp)
        Mh = d.learn_from_bottom(tree)
        errs.append(d.perm_err(Mh)); pinned.append(np.abs(Mh - d.M).max() < 0.2)
        accs.append((d.bp(Xte, Mh, d.pi)[0].argmax(-1) == yte).mean())
    print(f"    L'={Lp}: {d.n**Lp:6d} leaves, {d.n**(Lp-1):5d} bottom grids | kernel err (up to relabel)="
          f"{np.mean(errs):.3f}  labels pinned {sum(pinned)}/5  root acc={np.mean(accs):.3f}")
