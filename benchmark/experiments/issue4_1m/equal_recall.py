"""Issue #4 question 2 (speed at equal recall): the smallest TieredQ code budget whose recall reaches each reference's.

For every reference cell (RaBitQ, Qdrant TQ+ at 1/2/4 bits/dim) of a match_memory result, bisects TieredQ's
`code_bits` on seed-0 recall@10 (8 threads, batch 32, same build path as run.py) down to RESOLUTION code bits, then
records the found budget's recall over seeds 0-2. Reference recall is the mean over its seeds. Rows append to
OUTPUT/equal-recall-DATASET.csv (resumable); run.py then times TieredQ at these budgets. A companion query set
(measure_memory.COMPANION_QUERIES, e.g. msmarco1m-doc) takes its reference recall from the primary dataset's match.
Usage: python equal_recall.py DATASET DATA_DIR OUTPUT [--bits B ...] --matches MATCH_JSON...
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
RESOLUTION = 8
BATCH, THREADS = 32, 8
FIELDS = ["dataset", "reference_codec", "reference_bits", "reference_recall10", "code_bits", "recall10_seed0",
          "recall10_mean", "recall10_std", "probes"]


def recall_at(base, queries, truth, l2, code_bits, seed):
    import codec_native
    from measure import recall
    index = codec_native.Index(base, -(-code_bits // base.shape[1]), seed, True, l2, True, code_bits, True)
    ids = np.empty((len(queries), 10), dtype=np.uint32)
    for start in range(0, len(queries), BATCH):
        index.search(queries[start:start + BATCH], ids[start:start + BATCH], THREADS)
    return recall(truth, ids)


def smallest_budget(evaluate, target, upper):
    """Smallest code budget (to RESOLUTION) whose seed-0 recall reaches target; recall is near-monotone in budget."""
    lo, hi, probes = 0, upper, {}
    while hi - lo > RESOLUTION:
        middle = (lo + hi) // 2
        probes[middle] = evaluate(middle)
        lo, hi = (lo, middle) if probes[middle] >= target else (middle, hi)
    return hi, probes[hi] if hi in probes else evaluate(hi), len(probes)


def main(dataset, data, output, match_paths, bits_filter):
    import codec_native
    from data import prepare
    codec_native.set_threads(THREADS)
    base, queries, truth = prepare(data, dataset)
    l2 = dataset == "sift"
    path = output / f"equal-recall-{dataset}.csv"
    done = set()
    if path.exists():
        with path.open() as handle:
            done = {(r["dataset"], r["reference_codec"], r["reference_bits"]) for r in csv.DictReader(handle)}
    from measure_memory import COMPANION_QUERIES
    primary = {companion: name for name, companion in COMPANION_QUERIES.items()}.get(dataset)
    matched, recall_key = (primary, "companion_recall10") if primary else (dataset, "recall10")
    targets = {}
    for match in match_paths:
        for row in json.loads(Path(match).read_text()):
            if row["dataset"] == matched and (not bits_filter or row["reference_bits"] in bits_filter):
                targets[row["reference_codec"], row["reference_bits"]] = row["reference"][recall_key]["mean"]
    for (codec, bits), target in sorted(targets.items()):
        if (dataset, codec, str(bits)) in done:
            continue
        # A reference's recall is reached well below 8 bits/dim; 2x its own nominal budget bounds the search.
        upper = 2 * bits * base.shape[1] + RESOLUTION
        evaluate = lambda code_bits: recall_at(base, queries, truth, l2, code_bits, 0)
        code_bits, seed0, probes = smallest_budget(evaluate, target, upper)
        recalls = [seed0] + [recall_at(base, queries, truth, l2, code_bits, seed) for seed in SEEDS[1:]]
        row = dict(dataset=dataset, reference_codec=codec, reference_bits=bits, reference_recall10=target,
                   code_bits=code_bits, recall10_seed0=seed0, recall10_mean=statistics.mean(recalls),
                   recall10_std=statistics.stdev(recalls), probes=probes)
        new_file = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(row, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bits", nargs="+", type=int, default=[])
    parser.add_argument("--matches", nargs="+", required=True)
    args = parser.parse_args()
    main(args.dataset, args.data, args.output, args.matches, args.bits)
