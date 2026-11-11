"""Tables for the arxiv defaults study: per model, build and memory facts, the VIBE-style best QPS/GB among settings
that reach each Recall@100 target, and the full curve. "GB" is GiB of USS growth across fit(), as VIBE's harness
divides (index_size KiB / 1024^2).
Usage: summarize.py RESULTS_DIR > REPORT.md"""

import json
import sys
from pathlib import Path

TARGETS = (0.90, 0.95)
KIB_PER_GIB = 1024**2


def load(results):
    models = {}
    for path in sorted(results.glob("*/results.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if rows:
            models[path.parent.name] = rows
    return models


def gib(row):
    return row["index_size"] / KIB_PER_GIB


def best_at(rows, target):
    reaching = [r for r in rows if r["recall"] >= target]
    if not reaching:
        return "—"
    best = max(reaching, key=lambda r: r["qps"] / gib(r))
    return f"{best['qps'] / gib(best):,.0f} ({best['query']}, r {best['recall']:.3f}, {best['qps']:,.0f} QPS)"


def facts(rows):
    row, info = rows[0], rows[0].get("info", {})
    huge = info.get("memory_after_kib", {}).get("AnonHugePages", 0) - info.get("memory_before_kib", {}).get("AnonHugePages", 0)
    padding = info.get("padding_rows")
    return (f"{info.get('lists', '')} | {row['thp']} | {row['build_time']:.0f} s | {gib(row):.3f} | "
            f"{info.get('storage_bytes', 0) / 2**30:.3f} | "
            f"{f'{padding:,} ({padding / info['live_rows']:.1%})' if padding is not None else ''} | {huge / 1024:.0f}")


def main():
    models = load(Path(sys.argv[1]))
    print("| model | lists | THP | build | USS GiB | storage GiB | padding rows | AnonHuge MiB |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for name, rows in models.items():
        print(f"| {name} | {facts(rows)} |")
    print("\n| model | " + " | ".join(f"best QPS/GB at r ≥ {t}" for t in TARGETS) + " |")
    print("|---|" + "---|" * len(TARGETS))
    for name, rows in models.items():
        print(f"| {name} | " + " | ".join(best_at(rows, t) for t in TARGETS) + " |")
    print("\nCurves (query setting: recall @ QPS):\n")
    for name, rows in models.items():
        print(f"- **{name}**: " + ", ".join(f"{r['query']}: {r['recall']:.4f} @ {r['qps']:,.0f}" for r in rows))


if __name__ == "__main__":
    main()
