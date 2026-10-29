"""Issue #9 A/B summary per arm against base (c85b0f8): R@10, its change in points, the change against base recall
interpolated at the arm's stored bits (equal memory; blank outside base's range) and stored bits/vector.

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


def base_recall_at(curve: list[tuple[float, float]], stored: float) -> float | None:
    for (low_bits, low_recall), (high_bits, high_recall) in zip(curve, curve[1:]):
        if low_bits <= stored <= high_bits:
            return low_recall + (high_recall - low_recall) * (stored - low_bits) / (high_bits - low_bits)
    return None


def describe(arm: dict, base: dict, curve: list[tuple[float, float]]) -> str:
    if arm["models"] == base["models"]:
        return f"{'same':>26}"
    recall, stored = float(arm["recall"]), float(arm["stored_bits"])
    delta = 100 * (recall - float(base["recall"]))
    matched = base_recall_at(curve, stored)
    at_memory = f"{100 * (recall - matched):+5.2f}" if matched is not None else f"{'':>5}"
    return f"{recall:.4f} {delta:+5.2f} {at_memory} {stored:6.0f}"


def main(workspace: Path) -> None:
    for dataset in DATASETS:
        path = workspace / f"ab-{dataset}.tsv"
        if not path.exists():
            continue
        rows = load(path)
        arms = sorted({arm for arm, _ in rows} - SKIPPED_ARMS)
        curve = sorted((float(row["stored_bits"]), float(row["recall"])) for (arm, _), row in rows.items() if arm == "base")
        print(f"== {dataset}")
        print(f"{'bits/dim':>8} {'stored':>7}  {'base R@10':>9} {'3b':>5}  "
              + "  ".join(f"{arm + ' Δpt Δmem':>26}" for arm in arms))
        for budget in sorted({budget for _, budget in rows}):
            base = rows.get(("base", budget))
            if not base:
                continue
            cells = [describe(rows[arm, budget], base, curve) if (arm, budget) in rows else f"{'-':>26}" for arm in arms]
            print(f"{float(base['bits_per_dim']):8.3f} {float(base['stored_bits']):7.1f}  {float(base['recall']):9.4f} "
                  f"{float(base['three']):5.1f}  " + "  ".join(cells))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
