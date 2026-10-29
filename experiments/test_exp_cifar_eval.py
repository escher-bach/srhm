"""Regression test: eval_density's bits/dim must equal -mean log p(image) computed image by image
(the per-batch Jacobian bug of the first CIFAR run subtracted 3n log2 once per batch)."""
import math, torch
src = open("exp_cifar.py").read(); g = {"__name__": "x"}
exec(compile(src, "exp_cifar.py", "exec"), g)
it = g["it"]
m = it.ImageTree(4, seed=0)
X = (torch.rand(1100, 32, 32, 3) * 255).byte()
bpd, per, lp = g["eval_density"](m, X)
gen = torch.Generator().manual_seed(123)
ref = []
with torch.no_grad():
    for s in range(0, 1100, 500):
        ref.append(m.logp_image(it.pyramid(g["deq"](X[s:s + 500], gen))).double())
ref = torch.cat(ref).mean()
print("eval", lp, "per-image", float(ref), "bpd", bpd, float(-ref / (3072 * math.log(2)) + 8))
assert abs(lp - float(ref)) < 1e-3 * abs(float(ref)), "eval_density normalizer wrong"
print("PASS")
