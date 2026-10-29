"""Demo for Section 5: in a self-similar hierarchical process the shared kernel can be
learned from the *bottom level only*, with *zero labels*, and deployed at the root.

Compared against backpropagation from root labels through two tied SRGs:
  (B1) the exact-BP architecture with a learnable kernel (ideal inductive bias),
  (B2) a generic neural tied SRG (MLP cell map).
Small CPU run (~1-3 minutes).  python verify/demo_sshp.py
"""
import itertools
import numpy as np
import torch

k, n, L = 3, 3, 5                    # alphabet, branching, depth  -> 243 leaves
rng = np.random.default_rng(1)
torch.manual_seed(0)
torch.set_num_threads(4)

# generic column-stochastic kernel M[a, z] = P(child = a | parent = z)
D = rng.dirichlet(np.ones(k), size=k).T
M = 0.72 * np.eye(k) + 0.28 * D
pi = np.ones(k) / k
lam2 = sorted(np.abs(np.linalg.eigvals(M)))[-2]


def degenerate(M):
    """Thm 5.2(b): spurious relabelling exists iff M R S = R^T M for R != I."""
    perms = [np.eye(k)[list(p)] for p in itertools.permutations(range(k))]
    return any(np.allclose(M @ R @ S, R.T @ M) for R in perms[1:] for S in perms)


def sample(B, M=M, pi=pi):
    z = rng.choice(k, size=(B, 1), p=pi)
    root = z[:, 0].copy()
    cum = np.cumsum(M, axis=0)                          # cum[a, z]
    for _ in range(L):
        z = np.repeat(z, n, axis=1)
        u = rng.random(z.shape)
        z = (u[..., None] > cum.T[z]).sum(-1)
    return z, root


def bp(leaves, M, pi):
    """Upward belief propagation = the tied SRG with f = normalise(prod_i M^T x_i)."""
    beta = np.eye(k)[leaves]                            # (B, m, k)
    loglik = np.zeros(len(leaves))
    while beta.shape[1] > 1:
        B_, m_, _ = beta.shape
        msg = (beta.reshape(-1, k) @ M).reshape(B_, m_ // n, n, k).prod(axis=2)
        s = msg.sum(-1, keepdims=True)
        loglik += np.log(s[..., 0]).sum(-1)
        beta = msg / s
    post = pi * beta[:, 0]
    loglik += np.log(post.sum(-1))
    return post / post.sum(-1, keepdims=True), loglik


def learn_from_bottom(leaves, em_iters=50):
    """Zero labels. Kernel from the height-1 grids only; labelling pinned by self-similarity."""
    Mh = estimate_kernel(leaves.reshape(-1, n), em_iters)
    # self-similarity pins the latent labelling: pick the relabelling with
    # the highest tied likelihood on the (unlabelled) full trees
    cands = [Mh[:, list(p)] for p in itertools.permutations(range(k))]
    lls = [bp(leaves, C, pi)[1].sum() for C in cands]
    return cands[int(np.argmax(lls))]


def estimate_kernel(trip, em_iters=50):
    """Method of moments (Jennrich) + EM on the latent-class model of bottom grids."""
    T = np.zeros((k, k, k))
    np.add.at(T, (trip[:, 0], trip[:, 1], trip[:, 2]), 1.0)
    T /= len(trip)
    # Jennrich: T(.,.,u) T(.,.,v)^{-1} = M diag(.) M^{-1}
    Ms = []
    for _ in range(5):
        u, v = rng.normal(size=k), rng.normal(size=k)
        Tu, Tv = T @ u, T @ v
        w, V = np.linalg.eig(Tu @ np.linalg.pinv(Tv))
        V = np.real(V)
        V = V * np.sign(V.sum(0))
        V = np.clip(V, 1e-6, None)
        Ms.append(V / V.sum(0))
    Mh = Ms[0]
    # align the 5 restarts to the first by column matching, then average
    for Mx in Ms[1:]:
        best = min(itertools.permutations(range(k)),
                   key=lambda p: np.abs(Mx[:, list(p)] - Ms[0]).sum())
        Mh = Mh + Mx[:, list(best)]
    Mh /= len(Ms)
    ph = np.ones(k) / k
    # EM refinement on the bottom-level latent-class model (still local, no backprop)
    oh = np.eye(k)[trip]                                # (N, n, k)
    for _ in range(em_iters):
        like = ph * np.prod(oh @ Mh, axis=1)            # (N, k)
        r = like / like.sum(-1, keepdims=True)
        ph = r.mean(0)
        Mh = np.einsum('nz,nia->az', r, oh) + 1e-9
        Mh /= Mh.sum(0)
    return Mh


# ---------------- backprop baselines (root labels) --------------------------
def train_bp_kernel(leaves, y, steps=200, restarts=3):
    """Backprop from root labels; best of `restarts` random inits (by training loss)."""
    X = torch.nn.functional.one_hot(torch.tensor(leaves), k).float()
    Y = torch.tensor(y)
    best = (np.inf, None)
    for _ in range(restarts):
        logits = torch.randn(k, k, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=0.1)
        for _ in range(steps):
            Mt = torch.softmax(logits, 0)
            beta = X
            while beta.shape[1] > 1:
                B_, m_, _ = beta.shape
                msg = (beta.reshape(-1, k) @ Mt).reshape(B_, m_ // n, n, k).prod(2)
                beta = msg / msg.sum(-1, keepdim=True)
            loss = torch.nn.functional.nll_loss(torch.log(beta[:, 0] + 1e-12), Y)
            opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best[0]:
            best = (loss.item(), torch.softmax(logits.detach().double(), 0).numpy())
    return best[1]


def perm_err(Mh):
    return min(np.abs(Mh[:, list(p)] - M).max() for p in itertools.permutations(range(k)))


def independent_grids(N):
    """N bottom grids with independent parents drawn from the height-1 marginal."""
    p1 = np.linalg.matrix_power(M, L - 1) @ pi
    z = rng.choice(k, size=N, p=p1)
    cum = np.cumsum(M, axis=0)
    return (rng.random((N, n))[..., None] > cum.T[z][:, None, :]).sum(-1)


class NeuralSRG(torch.nn.Module):
    def __init__(self, d=16, h=64):
        super().__init__()
        self.emb = torch.nn.Embedding(k, d)
        self.f = torch.nn.Sequential(torch.nn.Linear(n * d, h), torch.nn.Tanh(),
                                     torch.nn.Linear(h, d), torch.nn.LayerNorm(d))
        self.out = torch.nn.Linear(d, k)

    def forward(self, leaves):
        x = self.emb(leaves)
        while x.shape[1] > 1:
            B_, m_, d_ = x.shape
            x = self.f(x.reshape(B_, m_ // n, n * d_))
        return self.out(x[:, 0])


def train_neural(leaves, y, steps=1500):
    net = NeuralSRG()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    X, Y = torch.tensor(leaves), torch.tensor(y)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (min(256, len(X)),))
        loss = torch.nn.functional.cross_entropy(net(X[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return net


if __name__ == "__main__":
    print(f"k={k} n={n} L={L}  leaves={n**L}  lambda_2={lam2:.3f}  "
          f"q=n*lambda^2={n*lam2**2:.2f}  degenerate={degenerate(M)}")
    Xte, yte = sample(5000)
    acc = lambda post: (post.argmax(-1) == yte).mean()
    print(f"oracle BP (true kernel)     : {acc(bp(Xte, M, pi)[0]):.3f}")
    census = np.array([np.bincount(r, minlength=k).argmax() for r in Xte])
    print(f"census (leaf majority)      : {(census == yte).mean():.3f}")
    print(f"chance                      : {1/k:.3f}")

    print("\n(A) bottom-level learning, ZERO labels, U unlabelled trees")
    for U in [1, 5, 20, 50, 100, 200]:
        accs, errs = [], []
        for seed in range(3):
            Xtr, _ = sample(U)
            Mh = learn_from_bottom(Xtr)
            accs.append(acc(bp(Xte, Mh, pi)[0])); errs.append(np.abs(Mh - M).max())
        print(f"    U={U:4d} trees ({U*n**(L-1):5d} bottom grids): acc={np.mean(accs):.3f}"
              f"  max|M_hat-M|={np.mean(errs):.3f}")

    print("\n(A') Thm 5.3 ablation: kernel error from U trees vs the same number of")
    print("     INDEPENDENT bottom grids (error up to latent relabelling, mean of 5 seeds)")
    for U in [1, 5, 20, 100]:
        e_tree = np.mean([perm_err(estimate_kernel(sample(U)[0].reshape(-1, n))) for _ in range(5)])
        e_ind = np.mean([perm_err(estimate_kernel(independent_grids(U * n ** (L - 1)))) for _ in range(5)])
        print(f"    {U*n**(L-1):5d} grids: from {U:3d} trees err={e_tree:.3f} | independent err={e_ind:.3f}")

    print("\n(B1) backprop through the exact-BP tied architecture, P labelled trees")
    for P in [10, 30, 100, 300, 1000]:
        accs = []
        for seed in range(2):
            Xtr, ytr = sample(P)
            accs.append(acc(bp(Xte, train_bp_kernel(Xtr, ytr), pi)[0]))
        print(f"    P={P:5d}: acc={np.mean(accs):.3f}")

    print("\n(B2) backprop through a generic neural tied SRG, P labelled trees")
    for P in [100, 1000, 10000]:
        accs = []
        for seed in range(2):
            Xtr, ytr = sample(P)
            net = train_neural(Xtr, ytr)
            with torch.no_grad():
                accs.append((net(torch.tensor(Xte)).argmax(-1).numpy() == yte).mean())
        print(f"    P={P:5d}: acc={np.mean(accs):.3f}")
