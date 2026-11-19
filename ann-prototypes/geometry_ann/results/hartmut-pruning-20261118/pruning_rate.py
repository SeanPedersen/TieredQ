"""#45 check 2: how many restricted-assignment candidates an exact PCA lower bound rules out.

For held-out rows of the 8.84M Qwen/MS MARCO SQ8 store and trained 64K centroids, candidates are the members of the
row's nearest `probes` supers (256 supers, k-means of the centroids, as SuperRestriction groups them). With an
orthonormal basis P (d' rows), |x-c|^2 >= |P(x-c)|^2 (bound "proj"), and adding the residual norms' triangle term
(|r_x| - |r_c|)^2 (bound "proj+norm"). A candidate survives when its bound is below the best exact distance among the
m candidates with the smallest bound (m = 0 means the true nearest, the ideal). Reports survivors and the GEMM FLOP
ratio (2 d d' + 2 d' C + 2 d (S + m)) / (2 d C) against scoring all C candidates exactly (supers excluded on both sides).
Usage: pruning_rate.py CENTROIDS.f32 [ROWS]"""
import json
import sys

import numpy as np

Q = "/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107"
DIM, LISTS, SUPERS, SUPER_ITERATIONS = 1024, 65536, 256, 10
PCA_ROWS, PROJECTED, PROBES, BEST_OF = 100_000, (64, 128, 256), (4, 16), (0, 1, 4)


def load_rows(ids):
    codes = np.memmap(f"{Q}/base.i8", dtype=np.int8, mode="r").reshape(-1, DIM)
    scales, offsets = np.fromfile(f"{Q}/scales.f32", np.float32), np.fromfile(f"{Q}/offsets.f32", np.float32)
    return codes[np.sort(ids)].astype(np.float32) * scales + offsets


def squared_distances(x, c):
    return (x * x).sum(1)[:, None] + (c * c).sum(1)[None, :] - 2.0 * x @ c.T


def nearest(x, c, block=8192):
    return np.concatenate([squared_distances(x[i:i + block], c).argmin(1) for i in range(0, len(x), block)])


def group_supers(centroids, rng):
    means = centroids[rng.choice(LISTS, SUPERS, replace=False)]
    for _ in range(SUPER_ITERATIONS):
        owner = nearest(centroids, means)
        means = np.stack([centroids[owner == s].mean(0) if (owner == s).any() else means[s] for s in range(SUPERS)])
    return means, nearest(centroids, means)


def bases(sample, centroids):
    centered = sample - sample.mean(0)
    residual = sample - centroids[nearest(sample, centroids)]
    out = {}
    for name, rows in (("data", centered), ("residual", residual)):
        _, vectors = np.linalg.eigh(rows.T @ rows)
        out[name] = vectors[:, ::-1].T.copy()  # rows by decreasing variance
    return out


def survivors(x, centroids, members, basis, projected, best_of):
    p = basis[:projected]
    px, pc = x @ p.T, centroids @ p.T
    rx = np.sqrt(np.maximum((x * x).sum(1) - (px * px).sum(1), 0))
    rc = np.sqrt(np.maximum((centroids * centroids).sum(1) - (pc * pc).sum(1), 0))
    stats = {(bound, m): [] for bound in ("proj", "proj+norm") for m in best_of}
    for row, candidates in enumerate(members):
        exact = squared_distances(x[row:row + 1], centroids[candidates])[0]
        proj = squared_distances(px[row:row + 1], pc[candidates])[0]
        for bound, lower in (("proj", proj), ("proj+norm", proj + (rx[row] - rc[candidates]) ** 2)):
            order = np.argsort(lower)
            for m in best_of:
                best = exact.min() if m == 0 else exact[order[:m]].min()
                stats[bound, m].append(int((lower < best).sum()))
    return {key: np.array(value) for key, value in stats.items()}


def main(centroid_path, rows):
    rng = np.random.default_rng(45)
    centroids = np.fromfile(centroid_path, np.float32).reshape(LISTS, DIM)
    total = np.memmap(f"{Q}/base.i8", dtype=np.int8, mode="r").size // DIM
    picked = rng.choice(total, PCA_ROWS + rows, replace=False)
    sample, x = load_rows(picked[:PCA_ROWS]), load_rows(picked[PCA_ROWS:])
    super_means, owner = group_supers(centroids, rng)
    basis = bases(sample, centroids)
    order = np.argsort(squared_distances(x, super_means), axis=1)
    by_super = [np.flatnonzero(owner == s) for s in range(SUPERS)]
    for probes in PROBES:
        members = [np.concatenate([by_super[s] for s in order[r, :probes]]) for r in range(rows)]
        count = np.array([len(m) for m in members], dtype=np.float64).mean()
        for name, b in basis.items():
            for projected in PROJECTED:
                stats = survivors(x, centroids, members, b, projected, BEST_OF)
                for (bound, m), alive in stats.items():
                    flops = (DIM * projected + projected * count + DIM * (alive.mean() + m)) / (DIM * count)
                    print(json.dumps({"probes": probes, "candidates": round(count), "basis": name, "projected": projected,
                                      "bound": bound, "best_of": m, "survivors_mean": round(float(alive.mean()), 1),
                                      "survivors_p90": float(np.percentile(alive, 90)), "flop_ratio": round(flops, 3)}),
                          flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4000)
