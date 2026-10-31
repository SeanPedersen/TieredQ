"""Issue #19.2 shifted distribution: hold out whole k-means clusters of a base from codec training.

k-means (inner product on the normalized rows, seeded sample) splits the base into CLUSTERS topics; a seeded
HOLDOUT_FRACTION of them is held out. Writes, for an eval data directory (base.f32, queries.f32, truth.u32 and
optionally truth100.u32):
- OUT/excluded_rows.u32: base rows of held-out clusters, for ivf_eval's TRAIN_EXCLUDE_ROWS;
- OUT/heldout, OUT/seen: eval directories (base.f32 linked) with the queries whose true top 10 lie mostly
  (more than half) in held-out clusters, or not;
- OUT/manifest.json: clusters, held-out ids, row and query counts.
The base itself is unchanged: held-out rows are inserted with the model trained on the rest.
Usage: shift_data.py DATA_DIR D OUT [CLUSTERS] [HOLDOUT_FRACTION]
"""
import json
import sys
from pathlib import Path
import numpy as np

CLUSTERS = 64
HOLDOUT_FRACTION = 0.25
SEED = 0
KMEANS_SAMPLE = 65_536
KMEANS_ITERATIONS = 20
ASSIGN_CHUNK = 65_536
TOP_K = 10


def assign(rows, centroids):
    """Nearest centroid by Euclidean distance (argmax of x.c - |c|^2 / 2), in chunks."""
    half_norms = 0.5 * np.sum(centroids * centroids, axis=1)
    labels = np.empty(len(rows), dtype=np.int64)
    for start in range(0, len(rows), ASSIGN_CHUNK):
        labels[start:start + ASSIGN_CHUNK] = np.argmax(rows[start:start + ASSIGN_CHUNK] @ centroids.T - half_norms, axis=1)
    return labels


def kmeans(sample, clusters, iterations, rng):
    centroids = sample[rng.choice(len(sample), clusters, replace=False)].copy()
    for _ in range(iterations):
        labels = assign(sample, centroids)
        for cluster in range(clusters):
            members = sample[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
    return centroids


def heldout_queries(truth, row_heldout):
    """Queries with more than half of their true top 10 in held-out rows."""
    return row_heldout[truth[:, :TOP_K]].sum(axis=1) * 2 > TOP_K


def split(base, truth, clusters, fraction, seed):
    """(row labels, held-out cluster ids, held-out row mask, held-out query mask)."""
    rng = np.random.default_rng(seed)
    sample = base[np.sort(rng.choice(len(base), min(KMEANS_SAMPLE, len(base)), replace=False))]
    labels = assign(base, kmeans(sample, clusters, KMEANS_ITERATIONS, rng))
    heldout = np.sort(rng.choice(clusters, round(clusters * fraction), replace=False))
    row_heldout = np.isin(labels, heldout)
    return labels, heldout, row_heldout, heldout_queries(truth, row_heldout)


def write_subset(source, target, d, mask):
    target.mkdir(parents=True, exist_ok=True)
    if not (target / "base.f32").exists():
        (target / "base.f32").symlink_to((source / "base.f32").resolve())
    queries = np.fromfile(source / "queries.f32", dtype=np.float32).reshape(-1, d)
    queries[mask].tofile(target / "queries.f32")
    for name in ("truth.u32", "truth100.u32"):
        if (source / name).exists():
            truth = np.fromfile(source / name, dtype=np.uint32).reshape(len(queries), -1)
            truth[mask].tofile(target / name)


def main(data, d, out, clusters=CLUSTERS, fraction=HOLDOUT_FRACTION):
    data, out, d = Path(data), Path(out), int(d)
    base = np.fromfile(data / "base.f32", dtype=np.float32).reshape(-1, d)
    truth = np.fromfile(data / "truth.u32", dtype=np.uint32).reshape(-1, TOP_K).astype(np.int64)
    labels, heldout, row_heldout, query_heldout = split(base, truth, int(clusters), float(fraction), SEED)
    out.mkdir(parents=True, exist_ok=True)
    np.flatnonzero(row_heldout).astype(np.uint32).tofile(out / "excluded_rows.u32")
    write_subset(data, out / "heldout", d, query_heldout)
    write_subset(data, out / "seen", d, ~query_heldout)
    manifest = dict(data=str(data), clusters=int(clusters), holdout_fraction=float(fraction), seed=SEED,
                    heldout_clusters=heldout.tolist(), base_rows=len(base), heldout_rows=int(row_heldout.sum()),
                    cluster_sizes=np.bincount(labels, minlength=int(clusters)).tolist(), queries=len(truth),
                    heldout_queries=int(query_heldout.sum()),
                    truth_in_heldout_share=float(row_heldout[truth].mean()))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "cluster_sizes"}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
