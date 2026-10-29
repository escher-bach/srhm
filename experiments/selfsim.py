"""Self-similarity probe for images: is the parent->children rule the same at every scale?

Scales: exact pyramid, S_{k+1} = 2x2 block average of S_k.  A p x p patch at scale k+1 covers
exactly four p x p child patches at scale k (a quadtree grid, n = 4).
Symbols: contrast-normalised patches coded with ONE codebook shared by all scales, plus a
'flat' code for featureless patches.
Rule at scale k: K_k^i(child code | parent code), i = child position.
Tied  <=>  K_k = K_k' for all scales.  For the partially tied model, rho = 1 - E_a TV(K_k(.|a),
K_k'(.|a)), so we estimate
    rho_hat(k, k') = 1 - [ TV_across(k, k') - TV_within ]      (per position, parent-weighted)
with independent image halves for the two scales and a within-scale noise floor at matched
sample sizes.  MI_k = I(parent code; child code) measures how informative the rule is.
"""
import glob
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

P_SIZE, TAU = 4, 0.02


# ------------------------------------------------------------------ images and pyramid
def load_gray(path):
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return im @ np.array([0.299, 0.587, 0.114])


def pyramid(img, K, p=P_SIZE):
    unit = p * 2 ** K
    H, W = (img.shape[0] // unit) * unit, (img.shape[1] // unit) * unit
    S = [img[:H, :W]]
    for _ in range(K):
        x = S[-1]
        S.append(0.25 * (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2]))
    return S


def patches(S, p=P_SIZE):
    H, W = S.shape
    return S.reshape(H // p, p, W // p, p).transpose(0, 2, 1, 3).reshape(H // p, W // p, p * p)


def normalise(X):
    X = X - X.mean(-1, keepdims=True)
    sd = X.std(-1)
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12), sd < TAU


# ------------------------------------------------------------------ codebook and coding
def fit_codebook(pyrs, v, rng, per_scale=40000):
    """k-means on normalised non-flat patches, pooled EQUALLY over scales (shared alphabet)."""
    K = len(pyrs[0]) - 1
    pool = []
    for k in range(K + 1):
        Xk = []
        for S in pyrs:
            X, flat = normalise(patches(S[k]))
            Xk.append(X[~flat])
        Xk = np.concatenate(Xk)
        pool.append(Xk[rng.choice(len(Xk), size=min(per_scale, len(Xk)), replace=False)])
    km = MiniBatchKMeans(n_clusters=v - 1, random_state=0, n_init=5, batch_size=4096)
    km.fit(np.concatenate(pool))
    return km


def encode(S, km, p=P_SIZE):
    X, flat = normalise(patches(S, p))
    code = km.predict(X.reshape(-1, X.shape[-1])).reshape(flat.shape) + 1
    code[flat] = 0
    return code


def parent_children(codes_k, codes_k1):
    """Rows (parent, c_TL, c_TR, c_BL, c_BR) for every parent at scale k+1."""
    h, w = codes_k1.shape
    c = codes_k[:2 * h, :2 * w]
    return np.stack([codes_k1.ravel(), c[0::2, 0::2].ravel(), c[0::2, 1::2].ravel(),
                     c[1::2, 0::2].ravel(), c[1::2, 1::2].ravel()], 1)


def image_tables(pyrs, km):
    """tables[img][k] = (N, 5) parent/children codes for the scale pair (k -> k+1)."""
    out = []
    for S in pyrs:
        codes = [encode(s, km) for s in S]
        out.append([parent_children(codes[k], codes[k + 1]) for k in range(len(S) - 1)])
    return out


# ------------------------------------------------------------------ statistics
def _tv(ra, rb, v):
    n = len(ra)
    return np.mean([0.5 * np.abs(np.bincount(ra[:, i], minlength=v) -
                                 np.bincount(rb[:, i], minlength=v)).sum() / n
                    for i in range(1, 5)])


def rho_hat(tables, k, k2, v, rng, splits=20, n_min=30):
    """rho_hat = 1 - (TV_across - TV_within) between the rules of scale pairs (k -> k+1) and
    (k2 -> k2+1).  Images are split into independent halves A, B.  Per parent code a, all three
    samples (scale k on A, scale k2 on A, scale k2 on B) are subsampled to the SAME size n, so
    across = TV(k|A, k2|B) and the noise floor within = TV(k2|A, k2|B) have equal sampling noise."""
    m = len(tables)
    res = []
    for _ in range(splits):
        perm = rng.permutation(m)
        A, B = perm[: m // 2], perm[m // 2:]
        T1 = np.concatenate([tables[i][k] for i in A])
        T2A = np.concatenate([tables[i][k2] for i in A])
        T2B = np.concatenate([tables[i][k2] for i in B])
        acc, wit, den = 0.0, 0.0, 0
        for a in range(v):
            s = [T[T[:, 0] == a] for T in (T1, T2A, T2B)]
            n = min(len(x) for x in s)
            if n < n_min:
                continue
            s = [x[rng.choice(len(x), n, replace=False)] for x in s]
            acc += n * _tv(s[0], s[2], v); wit += n * _tv(s[1], s[2], v); den += n
        if den:
            res.append((1 - (acc - wit) / den, acc / den, wit / den))
    if not res:                                   # too few parents at this scale pair
        return np.full(3, np.nan), np.full(3, np.nan)
    r = np.array(res)
    return r.mean(0), r.std(0)


# ------------------------------------------------------------------ predictive transfer
def rule_samples(S_k, S_k1, p=P_SIZE):
    """(parent, detail) pairs for the rule scale k+1 -> k.  parent: normalised p x p patch at
    scale k+1; detail: the children's 2p x 2p region minus the upsampled parent (zero block
    means, so it is exactly what the rule adds), divided by the parent's norm (scale-free)."""
    par = patches(S_k1, p)                                              # (h, w, p*p)
    h, w, _ = par.shape
    region = patches(S_k[:2 * h * p, :2 * w * p], 2 * p)                # (h, w, 4p^2)
    up = patches(np.kron(S_k1[:h * p, :w * p], np.ones((2, 2))), 2 * p)
    pc = par.reshape(-1, p * p); pc = pc - pc.mean(1, keepdims=True)
    nrm = np.linalg.norm(pc, axis=1)
    keep = pc.std(1) >= TAU
    X = pc[keep] / nrm[keep, None]
    Y = (region - up).reshape(-1, 4 * p * p)[keep] / nrm[keep, None]
    return X, Y


def rule_samples_ctx(S_k, S_k1, p=P_SIZE):
    """Like rule_samples, but the input is the parent's 3x3 neighbourhood of patches at scale
    k+1 (3p x 3p pixels), centred on the parent's mean and divided by the parent's norm.
    Border parents are dropped.  Returns (context, parent-only, detail)."""
    par = patches(S_k1, p)
    h, w, _ = par.shape
    region = patches(S_k[:2 * h * p, :2 * w * p], 2 * p)
    up = patches(np.kron(S_k1[:h * p, :w * p], np.ones((2, 2))), 2 * p)
    det = (region - up)[1:-1, 1:-1].reshape(-1, 4 * p * p)
    ctx = np.concatenate([par[1 + dr:h - 1 + dr, 1 + dc:w - 1 + dc] for dr in (-1, 0, 1)
                          for dc in (-1, 0, 1)], axis=-1).reshape(-1, 9 * p * p)
    c = par[1:-1, 1:-1].reshape(-1, p * p)
    mean = c.mean(1, keepdims=True)
    nrm = np.linalg.norm(c - mean, axis=1)
    keep = (c - mean).std(1) >= TAU
    s = nrm[keep, None]
    return (ctx[keep] - mean[keep]) / s, (c[keep] - mean[keep]) / s, det[keep] / s


def transfer(pyrs, k, k2, rng, splits=10, n_max=4000, n_neighbors=32):
    """tau(k -> k2) = R2(rule learned at scale k, tested at k2) / R2(rule learned at k2, tested
    at k2), same training size, test on held-out images.  Also returns both R2 values."""
    from sklearn.neighbors import KNeighborsRegressor
    m = len(pyrs)
    out = []
    for _ in range(splits):
        perm = rng.permutation(m)
        A, B = perm[: m // 2], perm[m // 2:]
        get = lambda idx, kk: [np.concatenate(z) for z in
                               zip(*[rule_samples(pyrs[i][kk], pyrs[i][kk + 1]) for i in idx])]
        Xa, Ya = get(A, k); Xa2, Ya2 = get(A, k2); Xb2, Yb2 = get(B, k2)
        n = min(len(Xa), len(Xa2), n_max)
        ia, ia2 = rng.choice(len(Xa), n, replace=False), rng.choice(len(Xa2), n, replace=False)
        mu = Ya2[ia2].mean(0)
        null = ((Yb2 - mu) ** 2).sum()
        r2, sig = [], []
        for Xt, Yt in ((Xa[ia], Ya[ia]), (Xa2[ia2], Ya2[ia2])):
            f = KNeighborsRegressor(n_neighbors=n_neighbors).fit(Xt, Yt)
            pred = f.predict(Xb2)
            r2.append(1 - ((Yb2 - pred) ** 2).sum() / null)
            sig.append(((pred - mu) * (Yb2 - mu)).sum())      # signal captured on held-out data
        out.append((sig[0] / sig[1] if sig[1] > 0 else np.nan,
                    r2[0] / r2[1] if r2[1] > 0 else np.nan, r2[0], r2[1], n))
    o = np.array(out)
    return np.nanmean(o, 0), np.nanstd(o, 0)


# ------------------------------------------------------------------ calibration images
def cascade_images(num, levels, rho, rng, p=P_SIZE, v_gen=8, m=4, beta=0.8):
    """Multiplicative cascade with KNOWN tying.  A parent patch P gets the children region
    upsample(P) + beta*std(P)*D, with D a zero-block-mean detail field chosen uniformly among the
    m details of P's generator code.  Each (code, slot) detail keeps the base one at a level with
    probability rho, else is fresh.  Block-averaging the children returns P exactly, so the
    probe's pyramid reproduces the generator's scales."""
    def detail():
        D = rng.normal(size=(2 * p, 2 * p))
        bm = 0.25 * (D[0::2, 0::2] + D[1::2, 0::2] + D[0::2, 1::2] + D[1::2, 1::2])
        D = D - np.kron(bm, np.ones((2, 2)))
        return D / D.std()
    T = rng.normal(size=(v_gen, p * p)); T -= T.mean(1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    base = np.array([[detail() for _ in range(m)] for _ in range(v_gen)])
    dicts = []
    for _ in range(levels):
        dl = base.copy()
        for a in range(v_gen):
            for j in range(m):
                if rng.random() >= rho:
                    dl[a, j] = detail()
        dicts.append(dl)
    imgs = []
    for _ in range(num):
        img = rng.normal(size=(p, p))
        for lev in range(levels):
            Pt = patches(img, p)                                   # (h, w, p*p)
            h, w, _ = Pt.shape
            Xn, _ = normalise(Pt)
            code = (Xn.reshape(-1, p * p) @ T.T).argmax(1)
            j = rng.integers(0, m, size=h * w)
            sd = Pt.reshape(-1, p * p).std(1) + 1e-3
            D = dicts[lev][code, j] * (beta * sd)[:, None, None]  # (hw, 2p, 2p)
            up = np.kron(img, np.ones((2, 2)))
            det = D.reshape(h, w, 2 * p, 2 * p).transpose(0, 2, 1, 3).reshape(2 * h * p, 2 * w * p)
            img = up + det
        imgs.append(img)
    return imgs


def pink_noise_images(num, size, rng):
    """Gaussian random field with power spectrum 1/f^2: exactly scale-invariant in law."""
    f = np.fft.fftfreq(size)
    fr = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2); fr[0, 0] = 1.0
    out = []
    for _ in range(num):
        spec = (rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))) / fr
        spec[0, 0] = 0
        x = np.real(np.fft.ifft2(spec))
        out.append((x - x.min()) / (x.max() - x.min()))
    return out


def mutual_info(tables, k, v):
    T = np.concatenate([t[k] for t in tables])
    mis = []
    for i in range(1, 5):
        J = np.zeros((v, v)); np.add.at(J, (T[:, 0], T[:, i]), 1); J /= J.sum()
        pa, pc = J.sum(1, keepdims=True), J.sum(0, keepdims=True)
        nz = J > 0
        mis.append((J[nz] * np.log2(J[nz] / (pa @ pc)[nz])).sum())
    return float(np.mean(mis))


def kodak(K):
    return [pyramid(load_gray(f), K) for f in sorted(glob.glob("data/kodak/*.png"))]
