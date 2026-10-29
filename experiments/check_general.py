"""Sanity check of srhm.tex section 8 in the soft (noisy) regime, which sections 3-4 never tested.

Tied tree process on Z={0,1,2}, branching s=3, with a generic *soft* kernel K(t|a) over the 27
tuples t, except that tuple t2 is made likelihood-proportional to t1: k(t2|.) = 0.5 k(t1|.).
Predictions:
  Lemma pooled / Thm mng(1): the empirical context law of t (sibling tuple at position 1 given
      tuple t at position 0, pooled over height-2 grids) converges to mu_t = Lambda_n beta_t, built
      from the *realized* sibling-latent law of the sample.
  Thm mng(2):  mu_{t1} = mu_{t2} exactly (proportional likelihoods), while other pairs differ.
  Thm depthg:  the estimation error depends on the data only through n_2 (total height-2 grids),
      not on how they are packaged into trees.
"""
import itertools
import numpy as np

k, s = 3, 3
TUP = np.array(list(itertools.product(range(k), repeat=s)))          # 27 x 3
T = len(TUP)
T1, T2 = 1, 22


def make_kernel(rng):
    """Generic soft kernel with parent->child persistence (children copy the parent with prob ~0.7),
    randomly perturbed so it has no symmetry; then tuple T2 is made likelihood-proportional to T1."""
    M = 0.7 * np.eye(k) + 0.3 * rng.dirichlet(np.ones(k), size=k).T     # column-stochastic
    K = np.ones((T, k))
    for i in range(s):
        K *= M[TUP[:, i]]                                              # CI product, K[t, a]
    K *= rng.uniform(0.7, 1.3, size=K.shape)                           # break CI / symmetry
    K /= K.sum(0, keepdims=True)
    others = [t for t in range(T) if t not in (T1, T2)]
    K[T2] = 0.5 * K[T1]
    K[others] *= (1 - K[T1] - K[T2]) / K[others].sum(0)
    return K                                                          # K[t, a] = k(t | a)


def noise_l1(p, N, rng, reps=200):
    """Expected L1 error of an empirical distribution of N draws from p (pure sampling noise)."""
    return np.mean([np.abs(rng.multinomial(N, p) / N - p).sum() for _ in range(reps)])


def sample_trees(K, U, L, rng):
    """Returns latents per height, height L (root) down to 0 (leaves), for U trees."""
    cum = np.cumsum(K, axis=0)
    z = rng.integers(k, size=U)
    levels = [z]
    for _ in range(L):
        u = rng.random(z.size)
        t = (u[None, :] > cum[:, z]).sum(0)                           # tuple index per node
        z = TUP[t].reshape(-1)                                        # children, in order
        levels.append(z)
    return levels[::-1]                                               # levels[h] = height h


def tuple_index(children):
    return (children.reshape(-1, s) * (k ** np.arange(s - 1, -1, -1))).sum(1)


def context_stats(K, levels):
    h1, h2 = levels[1], levels[2]
    t = tuple_index(levels[0])                                        # tuple of each height-1 grid
    t = t.reshape(-1, s)                                              # grouped by height-2 parent
    ti, tj = t[:, 0], t[:, 1]
    C = np.zeros((T, T))
    np.add.at(C, (ti, tj), 1)
    # realized sibling-latent law J^n (positions 0,1) from the height-2 latents
    K01 = np.zeros((k, k, k))                                         # [a, b, z]
    for idx, (a, b, _) in enumerate(TUP):
        K01[a, b] += K[idx]
    J = K01[:, :, h2].mean(-1)
    J1 = J.sum(1)
    nu = J / J1[:, None]
    beta = K * J1[None, :]
    beta /= beta.sum(1, keepdims=True)                                # beta[t, a]
    Lam = nu @ K.T                                                    # Lam[a, t'] = sum_b nu(b|a) K(t'|b)
    mu = beta @ Lam                                                   # mu[t, t']
    return C, mu


def main():
    rng = np.random.default_rng(0)
    K = make_kernel(rng)
    print("1) one deep tree (L=11, n_2=3^9=19683): empirical context vs Lambda_n beta_t")
    C, mu = context_stats(K, sample_trees(K, 1, 11, rng))
    N = C.sum(1)
    muhat = C / N[:, None]
    err = np.abs(muhat - mu).sum(1)
    base = np.array([noise_l1(mu[t], int(N[t]), rng) for t in range(T)])
    print(f"   mean L1(muhat_t, mu_t) = {err.mean():.4f};  pure-sampling-noise baseline = {base.mean():.4f}"
          f"  (ratio {err.mean()/base.mean():.2f})")
    D = np.abs(mu[:, None, :] - mu[None, :, :]).sum(-1)
    off = D[np.triu_indices(T, 1)]
    print(f"   theory:    L1(mu_t1, mu_t2) = {D[T1, T2]:.1e};  next-closest pair {np.sort(off)[1]:.4f} (near-synonyms: both beliefs are almost the same point mass), "
          f"median {np.median(off):.3f}")
    Dh = np.abs(muhat[:, None, :] - muhat[None, :, :]).sum(-1)
    d12 = Dh[T1, T2]
    # noise-only baseline for the difference of two independent empirical laws with the same mean
    b12 = np.mean([np.abs(rng.multinomial(int(N[T1]), mu[T1]) / N[T1]
                          - rng.multinomial(int(N[T2]), mu[T2]) / N[T2]).sum() for _ in range(500)])
    offh = Dh[np.triu_indices(T, 1)]
    print(f"   empirical: L1(muhat_t1, muhat_t2) = {d12:.3f} (noise-only expectation {b12:.3f}; N={N[T1]:.0f},{N[T2]:.0f});"
          f" pairs closer than t1,t2: {(offh < d12).sum()} of {offh.size}")

    print("2) packaging: fixed n_2 = 3^6 = 729 height-2 grids, split into U trees of depth L")
    for U, L in [(1, 8), (3, 7), (9, 6), (27, 5), (81, 4)]:
        errs = []
        for rep in range(40):
            C, mu = context_stats(K, sample_trees(K, U, L, rng))
            N = C.sum(1)
            ok = N > 0
            errs.append((np.abs(C[ok] / N[ok, None] - mu[ok]).sum(1) * np.sqrt(N[ok])).mean())
        print(f"   U={U:3d} trees, L={L}:  mean L1*sqrt(N_t) = {np.mean(errs):.3f} +- {np.std(errs)/np.sqrt(len(errs)):.3f}")


if __name__ == "__main__":
    main()
