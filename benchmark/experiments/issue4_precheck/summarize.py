"""Collect the #4 precheck cells into results.csv and one markdown summary per dataset.

Usage: summarize.py OUT BUNDLE
OUT is the job's output directory (cells/, matches-*.json, memory/memory.csv); BUNDLE receives results.csv,
matches.csv, memory.csv and summary-<dataset>.md. Recall is the mean ± sample std over seeds 0-2 (R@10 from k=10
searches, R@100 from k=100 searches); timing, build time and bytes come from the seed-0 cell of the timing session.
"""
import csv
import json
from pathlib import Path
import shutil
import statistics
import sys

from run import FIELDS

DATASET_NAMES = {"openai": "OpenAI-1536 100K", "gemini": "Gemini-768 99K", "openai1m": "OpenAI-1536 1M"}
CODEC_NAMES = {("tieredq", "fast-native"): "TieredQ default (fine tiers)", ("tieredq", "fast-native-nofine"): "TieredQ off",
               ("tieredq", "exact-native"): "TieredQ exact", ("rabitq", None): "RaBitQ", ("qdrant", None): "Qdrant TQ+"}
CODEC_ORDER = list(CODEC_NAMES)
REFERENCE_NAMES = {"rabitq": "RaBitQ", "qdrant": "Qdrant TQ+"}
BATCH_SCHEDULE, BULK_SCHEDULE, SINGLE_SCHEDULE, PARALLEL_SINGLE_SCHEDULE = (8, 32), (8, 1000), (1, 1), (8, 1)
SEEDS = (0, 1, 2)
NA = "n/a"


def load_rows(out):
    rows = []
    for path in sorted((out / "cells").glob("*.json")):
        for row in json.loads(path.read_text()):
            rows.append(row | {"cell": path.stem, "requested_code_bits": cell_code_bits(path.stem)})
    return rows


def cell_code_bits(stem):
    parts = [part for part in stem.split("-") if part.startswith("c") and part[1:].isdigit()]
    return int(parts[0][1:]) if parts else 0


def codec_key(row):
    return (row["codec"], row["config"] if row["codec"] == "tieredq" else None)


def mean_std(values):
    values = [value for value in values if value == value]
    if not values:
        return NA
    if len(values) < len(SEEDS):
        return f"{statistics.mean(values):.4f} (n={len(values)})"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


class Cell:
    """All seeds' rows of one configuration (codec, config, nominal bits, requested code_bits)."""

    def __init__(self, rows):
        self.rows = rows
        self.timed = [row for row in rows if int(row["seed"]) == 0]

    def recall(self, depth):
        values = []
        for seed in SEEDS:
            seed_rows = [row for row in self.rows if int(row["seed"]) == seed and int(row["k"]) == depth]
            if seed_rows:
                values.append(float(seed_rows[0][f"recall{depth}"]))
        return mean_std(values)

    def timing(self, schedule, k, field):
        threads, batch = schedule
        for row in self.timed:
            if (int(row["threads"]), int(row["batch"]), int(row["k"])) == (threads, batch, k):
                return float(row[field])
        return None

    def first(self, field):
        source = self.timed or self.rows
        return float(source[0][field])


def group(rows):
    cells = {}
    for row in rows:
        key = (row["dataset"], codec_key(row), int(row["target_bits"]), int(row["requested_code_bits"]))
        cells.setdefault(key, []).append(row)
    return {key: Cell(value) for key, value in cells.items()}


def number(value, digits=0):
    return NA if value is None else f"{value:,.{digits}f}"


def metrics(cell):
    """The summary columns after the row labels: bytes/vector, R@10, R@100, batch QPS@10, single p50@10, build s."""
    return (f"{cell.first('effective_bytes_per_vector'):.1f} | {cell.recall(10)} | {cell.recall(100)} | "
            f"{number(cell.timing(BATCH_SCHEDULE, 10, 'qps'))} | {number(cell.timing(SINGLE_SCHEDULE, 10, 'p50_ms'), 2)} | "
            f"{cell.first('build_s'):.1f} |")


def timing_row(label, bits, cell, code_bits=""):
    values = []
    for k in (10, 100):
        values += [number(cell.timing(BATCH_SCHEDULE, k, "qps")), number(cell.timing(BULK_SCHEDULE, k, "qps")),
                   number(cell.timing(SINGLE_SCHEDULE, k, "p50_ms"), 2), number(cell.timing(SINGLE_SCHEDULE, k, "p95_ms"), 2),
                   number(cell.timing(PARALLEL_SINGLE_SCHEDULE, k, "p50_ms"), 2),
                   number(cell.timing(PARALLEL_SINGLE_SCHEDULE, k, "p95_ms"), 2)]
    return (f"| {label} | {bits} | {code_bits} | " + " | ".join(values) +
            f" | {cell.first('effective_bits_per_dim') * cell.first('d'):,.0f} | {number(cell.first('encode_vps'))} |")


METRIC_HEADER = "bytes/vector | R@10 | R@100 | batch QPS@10 | single p50@10 ms | build s |"
SUMMARY_HEADER = f"| codec | bits/dim | {METRIC_HEADER}\n" + "|---" * 8 + "|"
DISK_HEADER = f"| reference | bits/dim | codec | code bits | {METRIC_HEADER}\n" + "|---" * 10 + "|"
TIMING_HEADER = ("| codec | bits/dim | code bits | QPS@10 8t·b32 | QPS@10 8t·b1000 | p50@10 1t·b1 | p95@10 1t·b1 | "
                 "p50@10 8t·b1 | p95@10 8t·b1 | QPS@100 8t·b32 | QPS@100 8t·b1000 | p50@100 1t·b1 | p95@100 1t·b1 | "
                 "p50@100 8t·b1 | p95@100 8t·b1 | bits/vector | encode vectors/s |\n" + "|---" * 17 + "|")


def memory_order(row):
    config = {"fine": "fast-native", "nofine": "fast-native-nofine", "exact": "exact-native"}.get(row["variant"])
    return int(row["target_bits"]), CODEC_ORDER.index((row["codec"], config))


def dataset_summary(dataset, cells, matches, memory):
    nominal = {key: cell for key, cell in cells.items() if key[0] == dataset and key[3] == 0}
    budgets = sorted({key[2] for key in nominal})
    lines = [f"# #4 precheck: {DATASET_NAMES.get(dataset, dataset)}", "",
             "Recall: mean ± sample std over seeds 0-2. Timing, bytes and build: seed 0, one session alone on CPUs 0-7. "
             "batch QPS@10: 8 threads, batch 32. single p50@10: 1 thread, batch 1. bytes/vector: run.py effective "
             "storage (codes + per-vector metadata + padding + model, amortized). TQ+ searches k=10 only (R@100 n/a) "
             "and has no 8-bit mode.", "", "## Nominal budgets", "", SUMMARY_HEADER]
    timing = ["## Timing detail (seed 0)", "", TIMING_HEADER]
    for bits in budgets:
        for codec in CODEC_ORDER:
            cell = nominal.get((dataset, codec, bits, 0))
            if cell is None:
                lines.append(f"| {CODEC_NAMES[codec]} | {bits} | " + " | ".join([NA] * 6) + " |")
                continue
            lines.append(f"| {CODEC_NAMES[codec]} | {bits} | " + metrics(cell))
            timing.append(timing_row(CODEC_NAMES[codec], bits, cell))
    lines += ["", "## Equal disk (TieredQ budget matched to the reference's bytes/vector)", "",
              DISK_HEADER]
    for match in sorted(matches, key=lambda m: (m["reference_bits"], m["reference"], m["variant"])):
        if match["dataset"] != dataset:
            continue
        reference = nominal.get((dataset, (match["reference"], None), match["reference_bits"], 0))
        config = {"fine": "fast-native", "nofine": "fast-native-nofine", "exact": "exact-native"}[match["variant"]]
        cell = cells.get((dataset, ("tieredq", config), match["reference_bits"], match["code_bits"]))
        if cell is None:
            continue
        label = f"| {REFERENCE_NAMES[match['reference']]} | {match['reference_bits']} |"
        if match["variant"] == "fine" and reference is not None:
            lines.append(f"{label} (reference) | | " + metrics(reference))
        lines.append(f"{label} {CODEC_NAMES[('tieredq', config)]} | {match['code_bits']} | " + metrics(cell))
        timing.append(timing_row(f"{CODEC_NAMES[('tieredq', config)]} @ {REFERENCE_NAMES[match['reference']]} disk",
                                 match["reference_bits"], cell, match["code_bits"]))
    lines += ["", *timing]
    if memory:
        lines += ["", "## Memory (seed 0, nominal budgets; measure_memory quick cells)", "",
                  "| codec | bits/dim | disk bytes/vector | steady RAM after reload, bytes/vector | peak RAM bytes/vector | build peak bytes/vector |",
                  "|---|---|---|---|---|---|"]
        for row in sorted(memory, key=memory_order):
            if row["dataset"] != dataset:
                continue
            label = {"fine": CODEC_NAMES[("tieredq", "fast-native")], "nofine": CODEC_NAMES[("tieredq", "fast-native-nofine")],
                     "exact": CODEC_NAMES[("tieredq", "exact-native")]}.get(row["variant"], REFERENCE_NAMES.get(row["codec"]))
            per_vector = lambda field: f"{float(row[field]) / 8:.1f}"
            lines.append(f"| {label} | {row['target_bits']} | {per_vector('disk_bits_per_vector')} | "
                         f"{per_vector('ram_loaded_bits_per_vector')} | {per_vector('ram_peak_bits_per_vector')} | "
                         f"{per_vector('build_peak_bits_per_vector')} |")
    return "\n".join(lines) + "\n"


def main():
    out, bundle = map(Path, sys.argv[1:])
    bundle.mkdir(parents=True, exist_ok=True)
    rows = load_rows(out)
    with (bundle / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cell", *FIELDS, "requested_code_bits"])
        writer.writeheader()
        writer.writerows(rows)
    matches = [match for path in sorted(out.glob("matches-*.json")) for match in json.loads(path.read_text())]
    if matches:
        with (bundle / "matches.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(matches[0]))
            writer.writeheader()
            writer.writerows(matches)
    for path in sorted(out.glob("calibration-*.csv")):
        shutil.copy(path, bundle / path.name)
    memory_path = out / "memory" / "memory.csv"
    memory = list(csv.DictReader(memory_path.open())) if memory_path.exists() else []
    if memory:
        shutil.copy(memory_path, bundle / "memory.csv")
    cells = group(rows)
    for dataset in sorted({row["dataset"] for row in rows}):
        (bundle / f"summary-{dataset}.md").write_text(dataset_summary(dataset, cells, matches, memory))


if __name__ == "__main__":
    main()
