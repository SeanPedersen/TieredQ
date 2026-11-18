"""Issue #45: do native e-means lists beat the quantized TieredQIVFIndex's own Lloyd lists?

    uv run ivf_lists_check.py DATA.npy QUERY_TRUTH.npz OUT.jsonl --queries A.fvecs,B.fvecs --lists 2048 \
        --probes 8,16,32,64,128 [--seed 0] [--only name,...]

TieredQIVFIndex (L2, 2 bits per dimension, router off so both route exactly) is built on all rows with Lloyd list
training (TieredQCoarseTraining.lloyd: exact below 8,192 lists, from there 4 iterations on 40 rows per list with 256 x 16
restricted assignment), its automatic training ("auto", e-means since #45), or native e-means on the Lloyd recipe's
assignment, exact or restricted. build_s is the whole build; train_s, assign_s and encode_s its phases. Per nprobe: recall@10 of the queries and rows scanned per query (sizes of the lists probe_lists names).
Each configuration runs in its own process; results append to OUT.jsonl.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from partition_study import load_queries
import methods

BITS_PER_DIMENSION = 2
K = 10
# name -> (e-means rows per list, epochs, batch-assignment probes or 0); None: TieredQCoarseTraining.lloyd(lists), the
# automatic Lloyd recipe before #45; "auto": the index's own (automatic) training
CONFIGS = {
    "lloyd-auto": None,
    "auto": "auto",
    "emeans-s64-e2.5": (64, 2.5, 0),    # 160 row passes per list, as the fast recipe's 40 rows x 4 iterations
    "emeans-s256-e2.5": (256, 2.5, 0),  # TieredQIVFFlatIndex's winner (the sample is capped at all rows)
    "emeans-s256-e2-p4": (256, 2, 4),   # the 8.84M / 64K winner: streamed, 4 probes in training, 16 in the final pass
}


def lloyd_training(lists: int):
    """The Lloyd recipe (exact below fast_lists, the fast restricted one from there)."""
    from tieredq import TieredQCoarseTraining

    lloyd = TieredQCoarseTraining.lloyd(lists)
    return dict(iterations=lloyd.iterations, assignment_supers=lloyd.assignment_supers,
                assignment_probes=lloyd.assignment_probes, sample_rows_per_list=lloyd.sample_rows_per_list)


def emeans_training(lists: int, per_list: int, epochs: float, probes: int):
    """The automatic recipe's assignment (exact, or restricted from fast_lists on) with e-means instead of Lloyd."""
    from tieredq import TieredQCoarseTraining

    automatic = TieredQCoarseTraining.lloyd(lists)
    return dict(iterations=automatic.iterations, assignment_supers=automatic.assignment_supers,
                assignment_probes=automatic.assignment_probes, sample_rows_per_list=per_list, emeans_epochs=epochs,
                emeans_probes=probes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--lists", type=int, required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", required=True)
    args = parser.parse_args()
    from tieredq import TieredQIVFIndex

    X = np.ascontiguousarray(np.load(args.data, mmap_mode="r"), dtype=np.float32)
    queries = load_queries(args.queries)
    truth = np.load(args.truth)["ids"][:, :K]
    for name in args.only.split(","):
        config = CONFIGS[name]
        started = time.perf_counter()
        index = TieredQIVFIndex(X.shape[1], BITS_PER_DIMENSION * X.shape[1], lists=args.lists, metric="l2",
                                router=False, seed=args.seed)
        if config is None:
            index.set_coarse_training(**lloyd_training(args.lists))
        elif config != "auto":
            index.set_coarse_training(**emeans_training(args.lists, *config))
        index.build(X)
        row = {"name": name, "seed": args.seed, "lists": index.lists, "build_s": round(time.perf_counter() - started, 2),
               **{key: round(value, 2) for key, value in index.build_stats.items()}}
        sizes = np.bincount(np.asarray(index.list_assignment()), minlength=index.lists)
        row["imbalance"] = round(float(index.lists * (sizes.astype(np.float64) ** 2).sum() / len(X) ** 2), 4)
        row["empty_lists"] = int((sizes == 0).sum())
        for nprobe in map(int, args.probes.split(",")):
            _, ids = index.search(queries, K, nprobe=nprobe)
            scanned = np.mean([sizes[index.probe_lists(q, nprobe=nprobe)].sum() for q in queries])
            row[f"p{nprobe}_scanned"] = round(float(scanned), 1)
            row[f"p{nprobe}_recall"] = round(methods.recall(ids.astype(np.int64), truth), 5)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        del index


if __name__ == "__main__":
    main()
