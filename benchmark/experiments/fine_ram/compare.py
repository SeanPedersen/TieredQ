"""Compare the fine-tier RAM A/B (job_ab.sh): memory per variant and budget, result identity, and timing.

Usage: compare.py WORKSPACE > summary.md
Memory: bytes/vector of each tree's measure_memory cell; `ids` is whether both trees' reload checks returned the same
hit ids (serve.json ids_sha256). Timing: the mean of each tree's rounds (cells/: ABBA, OpenAI 4 bits/dim; cells-quick/:
one pair per changed config), candidate / baseline.
"""
import csv
import json
from pathlib import Path
import statistics
import sys

MEMORY_FIELDS = [("ram_accounted_bytes", "accounted"), ("ram_loaded_rss_bytes", "steady RSS"),
                 ("ram_load_peak_rss_bytes", "load peak"), ("ram_query_peak_rss_bytes", "query peak")]
TIMINGS = [((8, 32, 10), "qps", "QPS@10 8t·b32"), ((8, 1000, 10), "qps", "QPS@10 8t·b1000"),
           ((1, 1, 10), "p50_ms", "p50@10 1t·b1"), ((8, 1, 10), "p50_ms", "p50@10 8t·b1"),
           ((8, 32, 100), "qps", "QPS@100 8t·b32"), ((1, 1, 100), "p50_ms", "p50@100 1t·b1"),
           ((8, 1, 100), "p50_ms", "p50@100 8t·b1")]


def memory_rows(workspace, tree):
    rows = {}
    for row in csv.DictReader((workspace / f"memory-{tree}" / "memory.csv").open()):
        key = (row["dataset"], row["variant"], int(row["target_bits"]))
        cell = workspace / f"memory-{tree}" / f"cell-{row['dataset']}-tieredq{'' if row['variant'] == 'fine' else '-' + row['variant']}-{row['target_bits']}-0"
        rows[key] = row | {"ids": json.loads((cell / "serve.json").read_text())["ids_sha256"]}
    return rows


def memory_table(workspace):
    base, cand = memory_rows(workspace, "base"), memory_rows(workspace, "cand")
    header = " | ".join(f"{label} base → cand" for _, label in MEMORY_FIELDS)
    lines = [f"| dataset | variant | bits/dim | {header} | ids |", "|---" * (4 + len(MEMORY_FIELDS)) + "|"]
    for key in sorted(base):
        n = int(base[key]["base_n"])
        cells = [f"{int(base[key][field]) / n:.1f} → {int(cand[key][field]) / n:.1f}" for field, _ in MEMORY_FIELDS]
        same = "same" if base[key]["ids"] == cand[key]["ids"] else "**differ**"
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(cells) + f" | {same} |")
    return lines


def timing_table(directory):
    cells = {}
    for path in sorted(directory.glob("*.json")):
        dataset, variant, bits, tree, _ = path.stem.split("-")
        for row in json.loads(path.read_text()):
            schedule = (int(row["threads"]), int(row["batch"]), int(row["k"]))
            cells.setdefault((dataset, variant, bits, tree, schedule), []).append(row)
    measured = [timing for timing in TIMINGS if any(key[4] == timing[0] for key in cells)]
    lines = ["| dataset | variant | bits/dim | " + " | ".join(label for _, _, label in measured) + " | R@10 base / cand |",
             "|---" * (4 + len(measured)) + "|"]
    for dataset, variant, bits in sorted({key[:3] for key in cells}):
        values = []
        for schedule, field, _ in measured:
            mean = {tree: statistics.mean(float(r[field]) for r in cells[(dataset, variant, bits, tree, schedule)])
                    for tree in ("base", "cand")}
            values.append(f"{mean['base']:,.4g} → {mean['cand']:,.4g} ({mean['cand'] / mean['base']:.2f}×)")
        recall = {tree: cells[(dataset, variant, bits, tree, (8, 32, 10))][0]["recall10"] for tree in ("base", "cand")}
        lines.append(f"| {dataset} | {variant} | {bits.removeprefix('b')} | " + " | ".join(values) +
                     f" | {recall['base']:.4f} / {recall['cand']:.4f} |")
    return lines


def main():
    workspace = Path(sys.argv[1])
    print("## Memory (bytes/vector, seed 0)\n")
    print("\n".join(memory_table(workspace)))
    print("\n## Timing, OpenAI 4 bits/dim (seed 0, mean of two ABBA rounds; QPS higher is better, p50 ms lower is better)\n")
    print("\n".join(timing_table(workspace / "cells")))
    quick = workspace / "cells-quick"
    if quick.exists():
        print("\n## Timing, changed code paths (seed 0, one baseline/candidate pair, k=10)\n")
        print("\n".join(timing_table(quick)))


if __name__ == "__main__":
    main()
