"""Summary of partition_study.py results: recall@k at equal join time, so unbalanced partitions cannot win by
scanning more rows per probe.

    uv run summarize_partitions.py OUT_DIR/partitions.jsonl [--baseline lloyd-s64-i10] [--cost join_s]
    uv run summarize_partitions.py OUT_DIR/queries.jsonl --cost scanned

Each trainer's (cost, recall) points (cost: join seconds, or rows scanned per query for query searches) over nprobe are interpolated (recall linear in log join time) at the
baseline's join times of the same seed, then averaged over seeds. Up to MAX_EXTRAPOLATION beyond a trainer's measured
range its end segment is extended (join times vary a few percent between runs); further out the cell is blank.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MAX_EXTRAPOLATION = 0.15  # relative join time beyond the measured range


def curve(row: dict, cost: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    probes = sorted(int(key[1:].split("_")[0]) for key in row if key.endswith("_recall"))
    joins = np.array([row[f"p{p}_{cost}"] for p in probes])
    recalls = np.array([row[f"p{p}_recall"] for p in probes])
    return probes, joins, recalls


def recall_at(row: dict, cost: str, seconds: float) -> float:
    _, joins, recalls = curve(row, cost)
    order = np.argsort(joins)
    joins, recalls = np.log(joins[order]), recalls[order]
    target = np.log(seconds)
    reach = np.log1p(MAX_EXTRAPOLATION)
    if target < joins[0] - reach or target > joins[-1] + reach:
        return float("nan")
    segment = int(np.clip(np.searchsorted(joins, target) - 1, 0, len(joins) - 2))
    slope = (recalls[segment + 1] - recalls[segment]) / (joins[segment + 1] - joins[segment])
    return float(recalls[segment] + slope * (target - joins[segment]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--baseline", default="lloyd-s64-i10")
    parser.add_argument("--cost", default="join_s", choices=("join_s", "scanned"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines()]
    by_name = defaultdict(dict)
    for row in rows:
        by_name[row["name"]][row["seed"]] = row
    baseline = by_name[args.baseline]
    probes, _, _ = curve(next(iter(baseline.values())), args.cost)
    header = ["trainer", "seeds", "train_s", "objective", "imbalance", "empty"]
    header += [f"recall@cost(p{p})" for p in probes] + [f"d vs base p{p}" for p in probes]
    print(" | ".join(header))
    for name, seeds in by_name.items():
        common = sorted(set(seeds) & set(baseline))
        if not common:
            continue
        at = np.array([[recall_at(seeds[s], args.cost, baseline[s][f"p{p}_{args.cost}"]) for p in probes] for s in common])
        base = np.array([[baseline[s][f"p{p}_recall"] for p in probes] for s in common])
        mean = lambda key: np.mean([seeds[s].get(key, np.nan) for s in common])  # noqa: E731
        cells = [name, ",".join(map(str, common)), f"{mean('train_s'):.2f}", f"{mean('objective'):.6g}",
                 f"{mean('imbalance'):.3f}", f"{mean('empty_lists'):.0f}"]
        cells += [f"{v:.4f}" for v in np.mean(at, axis=0)]
        cells += [f"{v:+.4f}" for v in np.mean(at - base, axis=0)]
        print(" | ".join(cells))


if __name__ == "__main__":
    main()
