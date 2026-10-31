"""Issue #4 report tables from committed result bundles: equal-memory recall, minimum bits for R@10 >= 0.95, speed.

Current TieredQ (072ea48 library) cells: SIFT from issue9-ship match-sift (native L2), GloVe from match-a, OpenAI and
Gemini (100k) from match-b, OpenAI-1M from the issue4-1m bundle (one match directory per bits/dim). Reference (RaBitQ, Qdrant TQ+) cells are the ones
those matches reused; reference timing comes from their own timing runs (issue4 final provenance), and at 1M from
the bundle's timing cells. Equal-memory speed rows are at the RAM-matched TieredQ budget; equal-recall rows at the
smallest TieredQ budget reaching the reference's mean R@10 (bundle equal-recall/, from equal_recall.py).
Usage: python report_tables.py ISSUE4_1M_BUNDLE > tables.md
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"
SHIP = RESULTS / "issue9-ship-20261030"
FINAL = RESULTS / "issue4-final-20261029"
REFERENCES = {"rabitq": "RaBitQ", "qdrant": "Qdrant TQ+"}
NAMES = {"sift": "SIFT1M (L2)", "glove": "GloVe-200 100k", "openai": "OpenAI-1536 100k", "gemini": "Gemini-768 99k",
         "openai1m": "OpenAI-1536 1M"}
DIMENSIONS = {"sift": 128, "glove": 200, "openai": 1536, "gemini": 768, "openai1m": 1536}
BITS = (1, 2, 4)
TARGET_RECALL = 0.95
FLOAT_BITS = 32
SCHEDULES = {"qps_t8_b32": (8, 32, "qps"), "qps_t1_b32": (1, 32, "qps"), "p50_t1_b1": (1, 1, "p50_ms")}


def match_directories(bundle):
    """dataset -> directories holding matches-{reference}.json and cells.csv."""
    return {"sift": [SHIP / "match-sift"], "glove": [SHIP / "match-a"], "openai": [SHIP / "match-b"],
            "gemini": [SHIP / "match-b"], "openai1m": sorted(bundle.glob("match-b*/"))}


def load_matches(directories):
    matches = {}
    for dataset, folders in directories.items():
        for directory, ref in ((folder, ref) for folder in folders for ref in REFERENCES):
            path = directory / f"matches-{ref}.json"
            for row in json.loads(path.read_text()) if path.exists() else []:
                if row["dataset"] == dataset and row["reference_bits"] in BITS:
                    matches[dataset, ref, row["reference_bits"], row["metric"]] = row
    return matches


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def timing_rows(bundle):
    """Seed-0 timing rows keyed (dataset, codec, bits or TieredQ code bits, threads, batch)."""
    rows = {}
    def add(row, budget):
        rows[row["dataset"], row["codec"], int(budget), int(row["threads"]), int(row["batch"])] = row
    for row in read_csv(SHIP / "output-final-matched-timing/matched-timing.csv"):
        add(row, row["code_bits"])
    for source in sorted(set(json.loads((FINAL / "reference-timing-provenance.json").read_text()).values())):
        path = FINAL / source if (FINAL / source).exists() else RESULTS / source
        for row in read_csv(path):
            if row["seed"] == "0" and row["codec"] != "tieredq":
                add(row, row["target_bits"])
    for path in (bundle / "timing").glob("cell-*.json"):
        for row in json.loads(path.read_text()):
            budget = path.stem.rsplit("-c", 1)[1] if row["codec"] == "tieredq" else row["target_bits"]
            add(row, budget)
    return rows


def speed(rows, dataset, codec, budget):
    if (dataset, codec, budget, 8, 32) not in rows:
        return None
    values = {name: float(rows[dataset, codec, budget, threads, batch][field])
              for name, (threads, batch, field) in SCHEDULES.items()}
    row = rows[dataset, codec, budget, 8, 32]
    return values | {"build_s": float(row["build_s"]), "bits": 8 * float(row["effective_bytes_per_vector"])}


def recall_table(matches, metrics):
    lines = ["| Dataset | Reference | bits/dim | Compression | Reference R@10 | "
             + " | ".join(f"TieredQ R@10 ({m}) | Δ pt ({m})" for m in metrics) + " |",
             "|---|---|---:|---:|---:|" + "---:|---:|" * len(metrics)]
    for dataset in NAMES:
        for ref, name in REFERENCES.items():
            for bits in BITS:
                first = matches.get((dataset, ref, bits, metrics[0]))
                if first is None:
                    continue
                reference = first["reference"]["recall10"]["mean"]
                compression = FLOAT_BITS * DIMENSIONS[dataset] / first["target_bits_per_vector"]
                cells = []
                for metric in metrics:
                    match = matches[dataset, ref, bits, metric]
                    if match["tieredq"] is None:
                        cells += ["none fits", "–"]
                        continue
                    recall = match["tieredq"]["recall10"]["mean"]
                    cells += [f"{recall:.4f}", f"{100 * (recall - reference):+.2f}"]
                lines.append(f"| {NAMES[dataset]} | {name} | {bits} | {compression:.1f}× | {reference:.4f} | "
                             + " | ".join(cells) + " |")
    return "\n".join(lines)


def minimum_bits(directories):
    """Smallest measured disk bits/vector reaching mean R@10 >= TARGET_RECALL per codec (TieredQ: probed budgets)."""
    lines = ["| Dataset | Codec | Min disk bits/vector at R@10 ≥ 0.95 | Compression | R@10 there | Best below 0.95 |",
             "|---|---|---:|---:|---:|---|"]
    for dataset, folders in directories.items():
        cells = {}
        for path in (folder / "cells.csv" for folder in folders if (folder / "cells.csv").exists()):
            for row in read_csv(path):
                budget = row["requested_code_bits"] if row["codec"] == "tieredq" else row["target_bits"]
                if row["dataset"] == dataset:
                    cells[row["codec"], int(budget), row["seed"]] = row
        budgets = defaultdict(list)
        for (codec, budget, _), row in cells.items():
            budgets[codec, budget].append(row)
        for codec in ("tieredq", *REFERENCES):
            points = sorted((max(float(r["disk_bits_per_vector"]) for r in rows),
                             statistics.mean(float(r["recall10"]) for r in rows), budget)
                            for (name, budget), rows in budgets.items() if name == codec)
            if not points:
                continue
            reached = [p for p in points if p[1] >= TARGET_RECALL]
            below = [p for p in points if p[1] < TARGET_RECALL and (not reached or p[0] < reached[0][0])]
            label = "TieredQ" if codec == "tieredq" else REFERENCES[codec]
            unit = "code bits" if codec == "tieredq" else "bits/dim"
            best_below = f"{below[-1][1]:.4f} at {below[-1][0]:,.0f} ({below[-1][2]} {unit})" if below else "–"
            if not reached:
                lines.append(f"| {NAMES[dataset]} | {label} | not reached (max {max(p[1] for p in points):.4f}) "
                             f"| – | – | {best_below} |")
                continue
            disk, recall, budget = reached[0]
            lines.append(f"| {NAMES[dataset]} | {label} | {disk:,.0f} ({budget} {unit}) | "
                         f"{FLOAT_BITS * DIMENSIONS[dataset] / disk:.1f}× | {recall:.4f} | {best_below} |")
    return "\n".join(lines)


def speed_table(matches, rows):
    lines = ["| Dataset | Reference | bits/dim | TieredQ code bits | QPS 8t·b32 TQ / ref | QPS 1t·b32 TQ / ref "
             "| p50 1t·b1 ms TQ / ref | Build s TQ / ref |", "|---|---|---:|---:|---|---|---|---|"]
    for dataset in NAMES:
        for ref, name in REFERENCES.items():
            for bits in BITS:
                match = matches.get((dataset, ref, bits, "ram"))
                if match is None or not match["code_bits"]:
                    continue
                tieredq, reference = speed(rows, dataset, "tieredq", match["code_bits"]), speed(rows, dataset, ref, bits)
                if tieredq is None or reference is None:
                    lines.append(f"| {NAMES[dataset]} | {name} | {bits} | {match['code_bits']} | not timed | | | |")
                    continue
                cell = lambda key, digits: ratio_cell(tieredq, reference, key, digits)
                lines.append(f"| {NAMES[dataset]} | {name} | {bits} | {match['code_bits']} | {cell('qps_t8_b32', 0)} | "
                             f"{cell('qps_t1_b32', 0)} | {cell('p50_t1_b1', 2)} | {cell('build_s', 1)} |")
    return "\n".join(lines)


def ratio_cell(tieredq, reference, key, digits):
    return (f"{tieredq[key]:,.{digits}f} / {reference[key]:,.{digits}f} ({tieredq[key] / reference[key]:.2f}×)")


def equal_recall_table(bundle, matches, rows):
    """TieredQ at the smallest budget reaching each reference's R@10: storage and speed against the reference."""
    lines = ["| Dataset | Reference | bits/dim | Reference R@10 | TieredQ code bits | TieredQ R@10 | Stored bits TQ / ref "
             "| QPS 8t·b32 TQ / ref | QPS 1t·b32 TQ / ref | p50 1t·b1 ms TQ / ref |", "|---|---|---:|---:|---:|---:|---|---|---|---|"]
    found = {(r["dataset"], r["reference_codec"], int(r["reference_bits"])): r
             for path in sorted(bundle.glob("equal-recall/**/equal-recall-*.csv")) for r in read_csv(path)}
    for dataset in NAMES:
        for ref, name in REFERENCES.items():
            for bits in BITS:
                point = found.get((dataset, ref, bits))
                if point is None:
                    continue
                code_bits = int(point["code_bits"])
                tieredq, reference = speed(rows, dataset, "tieredq", code_bits), speed(rows, dataset, ref, bits)
                head = (f"| {NAMES[dataset]} | {name} | {bits} | {float(point['reference_recall10']):.4f} | {code_bits} | "
                        f"{float(point['recall10_mean']):.4f} | ")
                if tieredq is None or reference is None:
                    lines.append(head + "not timed | | | |")
                    continue
                lines.append(head + f"{ratio_cell(tieredq, reference, 'bits', 0)} | {ratio_cell(tieredq, reference, 'qps_t8_b32', 0)} | "
                             f"{ratio_cell(tieredq, reference, 'qps_t1_b32', 0)} | {ratio_cell(tieredq, reference, 'p50_t1_b1', 2)} |")
    return "\n".join(lines)


def main(bundle):
    directories = match_directories(bundle)
    matches = load_matches(directories)
    print("## Recall@10 at equal measured disk and steady RAM\n")
    print(recall_table(matches, ["disk", "ram"]))
    print("\n## Recall@10 at equal peak RAM\n")
    print(recall_table(matches, ["peak"]))
    print("\n## Minimum measured disk bits/vector for R@10 ≥ 0.95\n")
    print(minimum_bits(directories))
    print("\n## Speed and build time at the RAM-matched budget (seed 0)\n")
    rows = timing_rows(bundle)
    print(speed_table(matches, rows))
    print("\n## Speed at equal recall (seed 0 timing; TieredQ R@10 is the mean over seeds 0-2)\n")
    print(equal_recall_table(bundle, matches, rows))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
