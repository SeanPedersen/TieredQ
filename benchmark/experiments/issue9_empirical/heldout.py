"""Issue #9 held-out selection: choose an allocation setting on even query ids, report it on odd ones.

Per dataset and pooled over all datasets, ranks arms by mean change in R@10 against base on the even half, then
prints the chosen arm's even and odd change (mean, min, max over budgets). Rows are job_heldout.sh ab-{dataset}.tsv
(ab_eval with even/odd recall as the last two columns).
Usage: python heldout.py RESULTS_DIR
"""
import statistics
import sys
from pathlib import Path

DATASETS = ["glove", "sift", "openai", "gemini", "openai1m"]
BASE = "base"
BUDGET, RECALL_EVEN, RECALL_ODD = 1, -2, -1


def load(path: Path) -> dict[str, dict[int, tuple[float, float]]]:
    arms: dict[str, dict[int, tuple[float, float]]] = {}
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        arms.setdefault(fields[0], {})[int(fields[BUDGET])] = (float(fields[RECALL_EVEN]), float(fields[RECALL_ODD]))
    return arms


def deltas(arm: dict[int, tuple[float, float]], base: dict[int, tuple[float, float]], half: int) -> list[float]:
    return [100 * (arm[budget][half] - base[budget][half]) for budget in sorted(base) if budget in arm]


def describe(values: list[float]) -> str:
    return f"{statistics.mean(values):+5.2f} [{min(values):+5.2f}, {max(values):+5.2f}]"


def main(results: Path) -> None:
    pooled: dict[str, dict[int, list[float]]] = {}
    for dataset in DATASETS:
        path = results / f"ab-{dataset}.tsv"
        if not path.exists():
            continue
        arms = load(path)
        base = arms.pop(BASE)
        print(f"== {dataset}: mean Δ R@10 pt [min, max] over budgets; even = selection half, odd = held out")
        ranked = sorted(arms, key=lambda arm: statistics.mean(deltas(arms[arm], base, 0)), reverse=True)
        for arm in ranked:
            even, odd = deltas(arms[arm], base, 0), deltas(arms[arm], base, 1)
            pooled.setdefault(arm, {0: [], 1: []})
            pooled[arm][0] += even; pooled[arm][1] += odd
            marker = "*" if arm == ranked[0] else " "
            print(f" {marker}{arm:>15}  even {describe(even)}  odd {describe(odd)}")
    print("== pooled over datasets (one global setting)")
    for arm in sorted(pooled, key=lambda arm: statistics.mean(pooled[arm][0]), reverse=True):
        print(f"  {arm:>15}  even {describe(pooled[arm][0])}  odd {describe(pooled[arm][1])}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
