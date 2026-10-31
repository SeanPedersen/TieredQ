"""Issue #22 task 3.1: query energy on the data PCA ranks of trained TieredQ models (no codec change).

The eval's training PCA (DV_TIEREDQ_TRAIN_DUMP of `ivf_eval tieredq-nesting`: residual eigenvectors and eigenvalues of
the 32,768-row seed-0 sample at 256 centroids) and each budget's tier layout (the `nesting model` lines) give the
coded bits per rank. Per query set, mu_r = E[(q.u_r)^2] is the energy a query puts on rank r and lambda_r the data
variance; score error ~ sum_r mu_r lambda_r 2^(-2 b_r) (uncoded ranks keep all of lambda_r). Reports, per budget:
energy shares on uncoded ranks and per tier, mu/lambda share ratios per rank band, and that predicted relative error.
It then re-allocates each budget's bits greedily (Gaussian Lloyd-Max distortions, at most MAX_RANK_BITS per rank)
for rank weights lambda^1.5 (today's rule without the storage layout) and lambda * mu_hat of the FIT set (the
query-aware MSE weight), and predicts every query set's error under each, next to RaBitQ's 1-bit isotropic error
(a random rotation spreads every rank's variance and query energy evenly over the d coordinates).
Usage: query_energy.py DUMP_PREFIX NESTING_OUT D FIT NAME=queries.f32...
"""
import re
import sys
import numpy as np

LLOYD_MAX_DISTORTION = (1.0, 0.3634, 0.1175, 0.03454, 0.009497, 0.002499, 0.0006642, 0.0001700, 0.0000434)
MAX_RANK_BITS = 8
RANK_BANDS = ((0, 64), (64, 128), (128, 256), (256, 512), (512, 768), (768, 1024), (1024, 1536))


def load_spectrum(prefix, d):
    """Eigenvalues (descending) and the d x d basis with rank r in column r."""
    values = np.fromfile(f"{prefix}.eigenvalues.f64")[::-1].copy()
    vectors = np.fromfile(f"{prefix}.eigenvectors.f64").reshape(d, d)[::-1].T.copy()  # LAPACK: column-major, ascending
    return values, vectors


def parse_tiers(text):
    """{budget: [(bits, dimensions), ...]} from `nesting model` lines; tiers are listed in rank order."""
    layouts = {}
    for line in text.splitlines():
        match = re.match(r"nesting\tmodel\t(\d+)\t.*tiers (.*)$", line)
        if match:
            layouts[int(match[1])] = [tuple(map(int, tier.split("x"))) for tier in match[2].split()]
    return layouts


def rank_bits(tiers, d):
    bits = np.zeros(d)
    start = 0
    for tier_bits, dimensions in tiers:
        bits[start:start + dimensions] = tier_bits
        start += dimensions
    return bits


def energy(queries, basis):
    """mu_r = mean over queries of (q.u_r)^2."""
    return np.mean((queries @ basis) ** 2, axis=0)


def shares(values):
    return values / values.sum()


def relative_error(mu, variance, bits):
    return float(np.sum(mu * variance * 2.0 ** (-2 * bits)) / np.sum(mu * variance))


def lloyd_max_error(mu, variance, bits):
    distortion = np.asarray(LLOYD_MAX_DISTORTION)[bits.astype(int)]
    return float(np.sum(mu * variance * distortion) / np.sum(mu * variance))


def isotropic_error(mu, variance, bits_per_dim):
    """RaBitQ-like: every rotated coordinate carries sum(lambda)/d variance and sum(mu)/d query energy."""
    d = len(variance)
    return float(LLOYD_MAX_DISTORTION[bits_per_dim] * variance.sum() * mu.sum() / d / np.sum(mu * variance))


def greedy_bits(weights, total_bits):
    """Integer bits per rank minimizing sum w_r D(b_r): the distortion steps are decreasing, so greedy is optimal."""
    gains = -np.diff(np.asarray(LLOYD_MAX_DISTORTION))[:MAX_RANK_BITS]
    steps = (weights[:, None] * gains[None, :]).ravel()
    chosen = np.argsort(-steps, kind="stable")[:total_bits]
    return np.bincount(chosen // MAX_RANK_BITS, minlength=len(weights)).astype(float)


def allocation_report(variance, energies, layouts, fit):
    names = list(energies)
    rules = {"lambda^1.5": variance ** 1.5, f"lambda * mu_{fit}": variance * energies[fit]}
    lines = ["", "## Predicted relative error (Lloyd-Max) by allocation rule, equal total bits", "",
             "| Budget | Rule | Coded ranks | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
    for budget, tiers in sorted(layouts.items()):
        actual = rank_bits(tiers, len(variance))
        candidates = {"trained tiers": actual} | {
            rule: greedy_bits(weights, int(actual.sum())) for rule, weights in rules.items()}
        for rule, bits in candidates.items():
            lines.append(f"| {budget} | {rule} | {np.count_nonzero(bits)} | "
                         + " | ".join(f"{lloyd_max_error(energies[n], variance, bits):.4f}" for n in names) + " |")
    lines.append(f"| – | RaBitQ 1 bit, isotropic (approx.) | {len(variance)} | "
                 + " | ".join(f"{isotropic_error(energies[n], variance, 1):.4f}" for n in names) + " |")
    return "\n".join(lines)


def report(variance, energies, layouts, d):
    names = list(energies)
    data_share = shares(variance)
    query_shares = {name: shares(mu) for name, mu in energies.items()}
    lines = ["## Energy share by rank band (mu share / lambda share)", "",
             "| Ranks | data lambda | " + " | ".join(f"{n} mu | {n} mu/lambda" for n in names) + " |",
             "|---|---|" + "---|---|" * len(names)]
    for low, high in RANK_BANDS:
        if low >= d:
            break
        band = slice(low, min(high, d))
        cells = [f"{query_shares[n][band].sum():.3f} | {query_shares[n][band].sum() / data_share[band].sum():.2f}"
                 for n in names]
        lines.append(f"| {low}-{min(high, d) - 1} | {data_share[band].sum():.3f} | " + " | ".join(cells) + " |")
    for budget, tiers in sorted(layouts.items()):
        bits = rank_bits(tiers, d)
        coded = int(np.count_nonzero(bits))
        lines += ["", f"## Budget {budget}: {coded} coded ranks", "",
                  "| Tier | Ranks | data lambda | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
        start = 0
        for tier_bits, dimensions in tiers + [(0, d - coded)]:
            band = slice(start, start + dimensions)
            label = f"{tier_bits} bit" if tier_bits else "uncoded"
            lines.append(f"| {label} | {start}-{start + dimensions - 1} | {data_share[band].sum():.3f} | "
                         + " | ".join(f"{query_shares[n][band].sum():.3f}" for n in names) + " |")
            start += dimensions
        lines.append("| predicted relative error | | "
                     f"{relative_error(variance, variance, bits):.4f} | "
                     + " | ".join(f"{relative_error(energies[n], variance, bits):.4f}" for n in names) + " |")
    return "\n".join(lines)


def main(prefix, nesting_path, d, fit, query_args):
    d = int(d)
    variance, basis = load_spectrum(prefix, d)
    energies = {}
    for argument in query_args:
        name, path = argument.split("=", 1)
        energies[name] = energy(np.fromfile(path, dtype=np.float32).reshape(-1, d), basis)
    layouts = parse_tiers(open(nesting_path).read())
    print(report(variance, energies, layouts, d))
    print(allocation_report(variance, energies, layouts, fit))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
