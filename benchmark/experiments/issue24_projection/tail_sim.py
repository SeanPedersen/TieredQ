"""Issue #24 go/no-go: tail (suffix-only ranks) score error of the current PCA-run code vs structured rotations.

Approximates the codec on OpenAI 100k: k-means residuals, PCA, prefix ranks [0, M1) left exact, tail ranks
[M1, MF) coded at the finer model's run widths. Schemes, same tail bits:
  pca      runs of equal bits, random rotation per run, per-run Lloyd on sigma-normalized coords, row LS scale
  exact    orthonormal basis of the tail subspace, one random rotation, uniform bits (query cost ~ d x M1)
  leaky    tail residual as a d-vector under a random d x d rotation, uniform bits over d coords (query cost d log d)
  hybrid   pca for the >= 4-bit runs, leaky for the rest
  capN     pca runs over only the first N tail ranks (greedy integer allocation of the same bits), the rest uncoded
Prints the tail score MSE <q, t - t_hat>^2 over (query, candidate) pairs relative to pca, and the tail signal for
scale. Usage: uv run --no-project --with numpy python -I tail_sim.py DATA_DIR (base.f32, queries.f32 of OpenAI-1536).
"""
import sys
import numpy as np

DATA = sys.argv[1]
D, M1, MF = 1536, 476, 1530
RUNS = [(59, 5), (367, 4), (347, 3), (246, 2), (35, 1)]
TRAIN, LISTS, QUERIES, CANDIDATES, LLOYD_ITERATIONS = 32768, 64, 300, 50, 30
rng = np.random.default_rng(0)

base = np.fromfile(f"{DATA}/base.f32", dtype=np.float32).reshape(-1, D)
queries = np.fromfile(f"{DATA}/queries.f32", dtype=np.float32).reshape(-1, D)[:QUERIES]
train = base[rng.choice(len(base), TRAIN, replace=False)]


def kmeans(x, k, iterations=10):
    centers = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iterations):
        assign = np.argmax(x @ centers.T - 0.5 * (centers**2).sum(1), axis=1)
        for c in range(k):
            members = x[assign == c]
            if len(members):
                centers[c] = members.mean(0)
    return centers


def residuals(x, centers):
    return x - centers[np.argmax(x @ centers.T - 0.5 * (centers**2).sum(1), axis=1)]


def lloyd(samples, bits):
    samples = np.sort(samples.ravel())[:: max(1, samples.size // 200000)]
    count = 1 << bits
    levels = np.quantile(samples, (np.arange(count) + 0.5) / count)
    for _ in range(LLOYD_ITERATIONS):
        cells = np.searchsorted((levels[1:] + levels[:-1]) / 2, samples)
        sums = np.bincount(cells, samples, count)
        counts = np.bincount(cells, minlength=count)
        levels = np.where(counts > 0, sums / np.maximum(counts, 1), levels)
        levels.sort()
    return levels


def quantize(values, levels):
    edges = (levels[1:] + levels[:-1]) / 2
    return levels[np.searchsorted(edges, values)]


def random_rotation(n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def uniform_widths(coords, bits):
    low = bits // coords
    widths = np.full(coords, low)
    widths[: bits - low * coords] += 1
    return widths


def code_groups(train_coords, coords, widths):
    """Per-coordinate widths (equal widths share Lloyd levels on sigma-normalized coords); row LS scale."""
    sigma = train_coords.std(0) + 1e-30
    out = np.empty_like(coords)
    for width in np.unique(widths):
        cols = widths == width
        levels = lloyd(train_coords[:, cols] / sigma[cols], int(width))
        out[:, cols] = quantize(coords[:, cols] / sigma[cols], levels) * sigma[cols]
    scale = (coords * out).sum(1) / np.maximum((out * out).sum(1), 1e-30)
    return out * scale[:, None]


def pca_scheme(train_tail, tail, runs):
    out, start = np.empty_like(tail), 0
    for width, bits in runs:
        rotation = random_rotation(width)
        cols = slice(start, start + width)
        rotated = code_groups(train_tail[:, cols] @ rotation, tail[:, cols] @ rotation, np.full(width, bits))
        out[:, cols] = rotated @ rotation.T
        start += width
    # One row scale over the whole tail, as the codec stores.
    scale = (tail * out).sum(1) / np.maximum((out * out).sum(1), 1e-30)
    return out * scale[:, None]


def rotated_scheme(train_tail, tail, bits):
    rotation = random_rotation(tail.shape[1])
    coded = code_groups(train_tail @ rotation, tail @ rotation, uniform_widths(tail.shape[1], bits))
    return coded @ rotation.T


centers = kmeans(train, LISTS)
train_res = residuals(train, centers)
mean = train_res.mean(0)
_, _, vt = np.linalg.svd(train_res - mean, full_matrices=False)
basis = vt.T  # d x d, PCA order

# Candidates: each query's top inner-product docs, coded as the rescorer sees them.
top = np.argsort(-(queries @ base.T), axis=1)[:, :CANDIDATES]
docs = base[top.ravel()]
doc_coords = (residuals(docs, centers) - mean) @ basis
train_coords = (train_res - mean) @ basis
q_rep = np.repeat(queries, CANDIDATES, axis=0)
q_coords = q_rep @ basis

tail_bits = sum(w * b for w, b in RUNS)
tail = slice(M1, MF)
results = {}
t_hat = pca_scheme(train_coords[:, tail], doc_coords[:, tail], RUNS)
results["pca"] = ((q_coords[:, tail] * (doc_coords[:, tail] - t_hat)).sum(1) ** 2).mean()
t_hat = rotated_scheme(train_coords[:, tail], doc_coords[:, tail], tail_bits)
results["exact"] = ((q_coords[:, tail] * (doc_coords[:, tail] - t_hat)).sum(1) ** 2).mean()


def leaky(cols, bits):
    """Tail part along `cols` as a d-vector, rotated in d dims and coded uniformly over all d coords."""
    train_vec, doc_vec = train_coords[:, cols] @ basis[:, cols].T, doc_coords[:, cols] @ basis[:, cols].T
    return rotated_scheme(train_vec, doc_vec, bits), doc_vec


hat, exact_vec = leaky(tail, tail_bits)
results["leaky"] = ((q_rep * (exact_vec - hat)).sum(1) ** 2).mean()
dense_ranks = sum(w for w, b in RUNS if b >= 4)
dense = slice(M1, M1 + dense_ranks)
dense_hat = pca_scheme(train_coords[:, dense], doc_coords[:, dense], [r for r in RUNS if r[1] >= 4])
hat, exact_vec = leaky(slice(M1 + dense_ranks, MF), sum(w * b for w, b in RUNS if b < 4))
error = (q_coords[:, dense] * (doc_coords[:, dense] - dense_hat)).sum(1) + (q_rep * (exact_vec - hat)).sum(1)
results["hybrid"] = (error**2).mean()



def greedy_runs(variances, bits, max_width=8):
    """Integer bits per rank by greedy marginal distortion (variance * 4^-b), grouped into runs of equal bits."""
    widths = np.zeros(len(variances), dtype=int)
    for _ in range(bits):
        gain = np.where(widths < max_width, variances * 4.0 ** (-widths.astype(float)), -1)
        if gain.max() < 0:
            break
        widths[np.argmax(gain)] += 1
    runs, start = [], 0
    while start < len(widths):
        end = start
        while end < len(widths) and widths[end] == widths[start]:
            end += 1
        runs.append((end - start, int(widths[start])))
        start = end
    return runs


variances = train_coords[:, tail].var(0)
for cap in (MF - M1, 768, 512, 384, 256):
    runs = [r for r in greedy_runs(variances[:cap], tail_bits) if r[1] > 0]
    coded = sum(w for w, _ in runs)
    kept = slice(M1, M1 + coded)
    t_hat = pca_scheme(train_coords[:, kept], doc_coords[:, kept], runs)
    error = (q_coords[:, kept] * (doc_coords[:, kept] - t_hat)).sum(1) + (q_coords[:, M1 + coded:MF] * doc_coords[:, M1 + coded:MF]).sum(1)
    results[f"cap{cap}"] = (error**2).mean()
    print(f"cap {cap}: runs {runs}", flush=True)

signal = ((q_coords[:, tail] * doc_coords[:, tail]).sum(1) ** 2).mean()
print(f"tail signal <q,t>^2 mean {signal:.3e}")
for name, mse in results.items():
    print(f"{name:7s} mse {mse:.3e}  x{mse / results['pca']:.2f} of pca")
