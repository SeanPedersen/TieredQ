"""Summarize the mid-width routing A/B: bytes, mean recall over seeds and seed-0 QPS per point and variant."""
import csv
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
VARIANTS = ("baseline", "new")
RUNS = {"b32": "", "b1": "-single"}


def rows(variant, run):
    for path in sorted((HERE / f"{variant}{RUNS[run]}").glob("*/results.csv")):
        yield from csv.DictReader(path.open())


def summarize():
    points = {}
    for variant in VARIANTS:
        for run in RUNS:
            for row in rows(variant, run):
                key = (row["dataset"], int(row["target_bits"]), row["codec"])
                entry = points.setdefault(key, {}).setdefault(variant, {})
                entry["bytes"] = float(row["effective_bytes_per_vector"])
                if run == "b32":
                    entry.setdefault("recall", []).append(float(row["recall10"]))
                if row["seed"] == "0":
                    entry[f"qps_{run}_t{row['threads']}"] = float(row["qps"])
    result = []
    for (dataset, bits, codec), variants in sorted(points.items()):
        for entry in variants.values():
            entry["recall"] = statistics.fmean(entry["recall"])
        result.append(dict(dataset=dataset, bits=bits, codec=codec, **variants))
    return result


def main():
    result = summarize()
    (HERE / "summary.json").write_text(json.dumps(result, indent=1))
    columns = ("bytes", "recall", "qps_b32_t8", "qps_b1_t1", "qps_b1_t8")
    writer = csv.writer(sys.stdout)
    writer.writerow(["dataset", "bits", "codec", *(f"{v}_{c}" for c in columns for v in VARIANTS)])
    for entry in result:
        writer.writerow([entry["dataset"], entry["bits"], entry["codec"],
                         *(f"{entry[v].get(c, float('nan')):.4g}" for c in columns for v in VARIANTS)])


if __name__ == "__main__":
    main()
