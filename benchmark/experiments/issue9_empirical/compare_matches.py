"""Issue #9: equal-memory TieredQ recall before and after the variance^1.5 fast default, per reference cell.

Reads match_memory.py results (matches-{qdrant,rabitq}.json) from a previous and a new match directory per dataset
group, and prints for every (reference, dataset, bits/dim, metric) target the reference R@10, TieredQ R@10 and
measured memory before and after, and the compression ratio 32*d / measured bits of the new TieredQ point.
Usage: python compare_matches.py OLD_DIR:NEW_DIR [OLD_DIR:NEW_DIR ...]
"""
import json
import statistics
import sys
from pathlib import Path

DIMENSIONS = {"sift": 128, "glove": 200, "openai": 1536, "gemini": 768, "openai1m": 1536}
REFERENCES = ["rabitq", "qdrant"]
FLOAT_BITS = 32


def load(directory: Path, reference: str) -> dict[tuple, dict]:
    path = directory / f"matches-{reference}.json"
    if not path.exists():
        return {}
    return {(r["dataset"], r["reference_bits"], r["metric"]): r for r in json.loads(path.read_text())}


def main() -> None:
    deltas: dict[str, list[float]] = {}
    for pair in sys.argv[1:]:
        old_dir, new_dir = (Path(p) for p in pair.split(":"))
        for reference in REFERENCES:
            old, new = load(old_dir, reference), load(new_dir, reference)
            print(f"\n## vs {reference} ({new_dir.name}; previous {old_dir})")
            print("| dataset | bits | metric | ref R@10 | old R@10 | new R@10 | Δ pt | old mem | new mem | target | new ratio |")
            print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for key in sorted(new):
                after, before = new[key], old.get(key)
                dataset, bits, metric = key
                memory = after["tieredq"][metric]["max"]
                ratio = FLOAT_BITS * DIMENSIONS[dataset] / memory
                old_recall = before["tieredq"]["recall10"]["mean"] if before else float("nan")
                old_memory = before["tieredq"][metric]["max"] if before else float("nan")
                delta = 100 * (after["tieredq"]["recall10"]["mean"] - old_recall)
                if before:
                    deltas.setdefault(f"{reference}:{metric}", []).append(delta)
                print(f"| {dataset} | {bits} | {metric} | {after['reference']['recall10']['mean']:.4f} | {old_recall:.4f} | "
                      f"{after['tieredq']['recall10']['mean']:.4f} | {delta:+.2f} | {old_memory:.1f} | {memory:.1f} | "
                      f"{after['target_bits_per_vector']:.1f} | {ratio:.1f}x |")
    print("\n## mean Δ R@10 (pt) at equal measured memory")
    for key, values in sorted(deltas.items()):
        print(f"{key}: {statistics.mean(values):+.2f} [{min(values):+.2f}, {max(values):+.2f}] over {len(values)} targets")


if __name__ == "__main__":
    main()
