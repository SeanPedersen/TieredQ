"""Summarize an interleaved A/B JSONL (timing_probe.py rows) against a reference tree (default 'source').

Per index × schedule × tree: median QPS / p50 / p95 over rounds, ratio of medians to source, the range of
same-round paired QPS ratios, and source's own max/min QPS spread as the run-to-run noise reference.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path



def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.startswith("{")]
    runs = defaultdict(list)
    for row in rows:
        runs[(row["index"], row["label"])].append(row)
    return runs


def table(runs, reference_label):
    indexes = sorted({index for index, _ in runs})
    labels = sorted({label for _, label in runs}, key=lambda label: (label != reference_label, label))
    lines = ["| Index | Schedule | Tree | Rounds | QPS (median) | QPS ratio | Paired ratio range | p50 ratio | p95 ratio "
             "| Reference spread | IDs vs reference |", "|---|---|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for index in indexes:
        reference = runs[(index, reference_label)]
        reference_ids = {row["ids_sha256"] for row in reference}
        schedules = sorted({key for row in reference for key in row if key.startswith("t")})
        for schedule in schedules:
            base = [row[schedule] for row in reference if schedule in row]
            base_qps = statistics.median(m["qps"] for m in base)
            spread = max(m["qps"] for m in base) / min(m["qps"] for m in base)
            for label in labels:
                measured = [row[schedule] for row in runs.get((index, label), []) if schedule in row]
                if not measured:
                    continue
                paired = [m["qps"] / b["qps"] for m, b in zip(measured, base)]
                ratio = lambda key: statistics.median(m[key] for m in measured) / statistics.median(b[key] for b in base)
                same = {row["ids_sha256"] for row in runs[(index, label)]} == reference_ids
                lines.append(f"| {index} | {schedule} | {label} | {len(measured)} | {statistics.median(m['qps'] for m in measured):,.0f} "
                             f"| {statistics.median(m['qps'] for m in measured) / base_qps:.3f} | {min(paired):.3f}–{max(paired):.3f} "
                             f"| {ratio('p50_ms'):.3f} | {ratio('p95_ms'):.3f} | {spread:.3f} | {'identical' if same else 'differs'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(table(load(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "source"), end="")
