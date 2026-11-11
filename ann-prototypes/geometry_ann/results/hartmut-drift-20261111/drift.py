"""Training on a shifted sample (#19 §2): does a TieredQ IVF whose codec and lists never saw part of the data lose
recall on it? VIBE arxiv 1.34M x 768. The base is cut into CLUSTERS clusters (a 100-list TieredQ IVF's assignment,
the same for every model). A model trains its codec and lists (train_lists) on SAMPLE_ROWS rows and then builds on
every row:
- holdout: the sample avoids HELD_OUT_SHARE of the clusters, drawn with the seed;
- random: a uniform sample of the same size, drawn with the seed;
- codec-holdout / lists-holdout: only the codec, or only the lists, train on the holdout sample, the other on the
  random one. The first train_lists() call trains the codec and the lists, a second retrains only the lists.
Recall@10 (VIBE's 1e-3 tolerance) is measured per nprobe over all queries and over the queries whose true nearest
neighbour lies in the held-out clusters (for random models, the clusters the same seed would hold out). Also records
the largest list and the share of rows in lists over 4x the mean.

Usage: BUILD_CORES=.. drift.py DATASET.hdf5 OUT.json MODE SEED"""
import json
import os
import sys
import time

import h5py
import numpy as np
from tieredq import TieredQIVFIndex

CLUSTERS, HELD_OUT_SHARE, CLUSTER_SEED = 100, 0.2, 2026
LISTS, BITS_PER_DIM, SAMPLE_ROWS = 2048, 1.75, 131072
NPROBES, K, RECALL_EPSILON = (8, 16, 32, 64), 10, 1e-3
CLUSTER_CACHE = "clusters.u32"


def clusters(train):
    """Every base row's cluster, from a 100-list IVF trained on a fixed sample (cached across models)."""
    if os.path.exists(CLUSTER_CACHE):
        return np.fromfile(CLUSTER_CACHE, dtype=np.uint32)
    index = TieredQIVFIndex(dimensions=train.shape[1], code_bits=train.shape[1], lists=CLUSTERS, metric="ip")
    pick = np.sort(np.random.default_rng(CLUSTER_SEED).choice(len(train), SAMPLE_ROWS, replace=False))
    index.train_lists(np.ascontiguousarray(train[pick]))
    index.build(train)
    labels = index.list_assignment().astype(np.uint32)
    labels.tofile(CLUSTER_CACHE)
    return labels


def recall(train, queries, true_distances, ids, rows):
    hits = []
    for q in rows:
        found = ids[q][ids[q] != 2**32 - 1]
        distances = 1.0 - train[found] @ queries[q]
        hits.append(np.sum(distances[:K] <= true_distances[q][K - 1] + RECALL_EPSILON))
    return float(np.mean(hits)) / K


def main():
    dataset, out, mode, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    os.sched_setaffinity(0, {int(c) for c in os.environ["BUILD_CORES"].split(",")})
    with h5py.File(dataset, "r") as f:
        train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        queries = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        neighbors, true_distances = f["neighbors"][:], f["distances"][:]
    labels = clusters(train)
    def after_clusters():
        """The seed's generator once it has drawn the held-out clusters (each sample draws from its own copy)."""
        rng = np.random.default_rng(seed)
        return rng, rng.choice(CLUSTERS, round(HELD_OUT_SHARE * CLUSTERS), replace=False)

    held_out = after_clusters()[1]
    shifted = np.isin(labels, held_out)
    holdout = np.ascontiguousarray(train[np.sort(after_clusters()[0].choice(np.flatnonzero(~shifted), SAMPLE_ROWS, replace=False))])
    uniform = np.ascontiguousarray(train[np.sort(after_clusters()[0].choice(len(train), SAMPLE_ROWS, replace=False))])
    samples = {"holdout": [holdout], "random": [uniform], "codec-holdout": [holdout, uniform],
               "lists-holdout": [uniform, holdout]}[mode]
    started = time.time()
    index = TieredQIVFIndex(dimensions=train.shape[1], code_bits=round(BITS_PER_DIM * train.shape[1]), lists=LISTS, metric="ip")
    for sample in samples:
        index.train_lists(sample)
    index.build(train)
    build_s = time.time() - started
    sizes = np.bincount(index.list_assignment(), minlength=LISTS)
    shifted_queries = np.flatnonzero(shifted[neighbors[:, 0]])
    points = []
    for nprobe in NPROBES:
        _, ids = index.search(queries, k=K, nprobe=nprobe)
        points.append(dict(nprobe=nprobe, recall_all=recall(train, queries, true_distances, ids, range(len(queries))),
                           recall_shifted=recall(train, queries, true_distances, ids, shifted_queries)))
    row = dict(mode=mode, seed=seed, held_out=sorted(int(c) for c in held_out), shifted_rows=int(shifted.sum()),
               shifted_queries=int(len(shifted_queries)), build_s=build_s, largest_list=int(sizes.max()),
               rows_in_lists_over_4x_mean=float(sizes[sizes > 4 * sizes.mean()].sum() / sizes.sum()),
               empty_lists=int((sizes == 0).sum()), points=points)
    with open(out, "w") as f:
        json.dump(row, f, indent=2)
    print(json.dumps(row))


if __name__ == "__main__":
    main()
