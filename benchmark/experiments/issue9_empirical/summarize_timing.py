"""Issue #9 build and search cost per arm against base (c85b0f8), from job_round2.sh time-{dataset}.tsv rows.

Per (dataset, budget, arm): median train and add seconds over rounds, best 1-thread and 8-thread batch-32 QPS,
widest tier and stored bits/vector; arms show their change in percent against base.
Usage: python summarize_timing.py RESULTS_DIR
"""
import statistics
import sys
from collections import defaultdict
from pathlib import Path

COLUMNS = ["arm", "budget", "bits_per_dim", "widest", "stored_bits", "train_s", "add_s", "t1b32", "t8b32", "model"]
DATASETS = ["glove", "sift", "openai", "gemini"]
BASE = "base"


def load(path: Path) -> dict[tuple[str, int], dict]:
    rounds: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        row = dict(zip(COLUMNS, line.split("\t")))
        rounds[row["arm"], int(row["budget"])].append(row)
    return {key: {
        "bits_per_dim": float(rows[0]["bits_per_dim"]),
        "widest": int(rows[0]["widest"]),
        "stored_bits": float(rows[0]["stored_bits"]),
        "train_s": statistics.median(float(r["train_s"]) for r in rows),
        "add_s": statistics.median(float(r["add_s"]) for r in rows),
        "t1b32": max(float(r["t1b32"]) for r in rows),
        "t8b32": max(float(r["t8b32"]) for r in rows),
        "rounds": len(rows),
    } for key, rows in rounds.items()}


def percent(value: float, reference: float) -> str:
    return f"{100 * (value / reference - 1):+5.1f}%"


def main(results: Path) -> None:
    for dataset in DATASETS:
        path = results / f"time-{dataset}.tsv"
        if not path.exists():
            continue
        cells = load(path)
        arms = [BASE] + sorted({arm for arm, _ in cells} - {BASE})
        print(f"== {dataset}  (train s, add s, 1t·b32 QPS, 8t·b32 QPS; arms as change vs base)")
        for budget in sorted({budget for _, budget in cells}):
            base = cells.get((BASE, budget))
            if not base:
                continue
            print(f"  {base['bits_per_dim']:.2f} b/d  base: train {base['train_s']:6.3f}  add {base['add_s']:6.3f}  "
                  f"q1 {base['t1b32']:6.0f}  q8 {base['t8b32']:6.0f}  w{base['widest']} {base['stored_bits']:.0f}b "
                  f"({base['rounds']} rounds)")
            for arm in arms[1:]:
                cell = cells.get((arm, budget))
                if not cell:
                    continue
                print(f"  {'':8}{arm:>13}: train {percent(cell['train_s'], base['train_s'])}  "
                      f"add {percent(cell['add_s'], base['add_s'])}  q1 {percent(cell['t1b32'], base['t1b32'])}  "
                      f"q8 {percent(cell['t8b32'], base['t8b32'])}  w{cell['widest']} {cell['stored_bits']:.0f}b")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
