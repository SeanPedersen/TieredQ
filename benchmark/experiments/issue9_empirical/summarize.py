"""Issue #9 A/B summary: recall, its change in points and stored bits per arm against base (c85b0f8).

Usage: python summarize.py RESULTS_DIR  (reads ab-{dataset}.tsv written by job_ab.sh or job_oracle.sh)
"""
import sys
from pathlib import Path

COLUMNS = ["arm", "budget", "bits_per_dim", "realised", "widest", "three", "vector_bytes", "stored_bits",
           "recall", "t1b1", "t1b32", "t8b32", "models"]
DATASETS = ["glove", "sift", "openai", "gemini"]
SKIPPED_ARMS = {"base", "gaussian"}


def load(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    for line in path.read_text().splitlines():
        row = dict(zip(COLUMNS, line.split("\t")))
        rows[row["arm"], int(row["budget"])] = row
    return rows


def describe(arm: dict, base: dict) -> str:
    if arm["models"] == base["models"]:
        return f"{'same':>20}"
    delta = 100 * (float(arm["recall"]) - float(base["recall"]))
    return f"{float(arm['recall']):.4f} {delta:+5.2f} {float(arm['stored_bits']):6.0f}"


def main(workspace: Path) -> None:
    for dataset in DATASETS:
        path = workspace / f"ab-{dataset}.tsv"
        if not path.exists():
            continue
        rows = load(path)
        arms = sorted({arm for arm, _ in rows} - SKIPPED_ARMS)
        print(f"== {dataset}")
        print(f"{'bits/dim':>8} {'stored':>7}  {'base R@10':>9} {'3b':>5}  "
              + "  ".join(f"{arm:>20}" for arm in arms))
        for budget in sorted({budget for _, budget in rows}):
            base = rows.get(("base", budget))
            if not base:
                continue
            cells = [describe(rows[arm, budget], base) if (arm, budget) in rows else f"{'-':>20}" for arm in arms]
            print(f"{float(base['bits_per_dim']):8.3f} {float(base['stored_bits']):7.1f}  {float(base['recall']):9.4f} "
                  f"{float(base['three']):5.1f}  " + "  ".join(cells))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
