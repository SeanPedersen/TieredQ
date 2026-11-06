"""Summarize completed repeated sweeps at equal recall, optionally constraining index bytes."""
import argparse
import csv
import json
from pathlib import Path

TARGETS = (0.80, 0.90, 0.95, 0.97, 0.98, 0.99)
FIELDS = ("codec", "bits", "lists", "nprobe", "index_bits", "recall", "qps", "build_s")


def read_rows(directory):
    rows = []
    for path in sorted(directory.glob("*.tsv")):
        for values in csv.reader(path.open(), delimiter="\t"):
            raw = dict(zip(FIELDS, values, strict=True))
            rows.append({key: value if key == "codec" else float(value)
                         for key, value in raw.items()})
    return rows


def fastest(rows, target, budget=float("inf")):
    eligible = [row for row in rows if row["recall"] >= target and row["index_bits"] <= budget]
    return max(eligible, key=lambda row: row["qps"], default=None)


def smallest(rows, target):
    eligible = [row for row in rows if row["recall"] >= target]
    return min(eligible, key=lambda row: (row["index_bits"], -row["qps"]), default=None)


def describe(row):
    if row is None:
        return "unreached"
    return f'{row["codec"]} {row["bits"]:g}b np{row["nprobe"]:g}'


def comparison(rows):
    zvec = [row for row in rows if row["codec"] == "zvec"]
    tieredq = [row for row in rows if row["codec"] != "zvec"]
    print("| R@10 target | zvec config | QPS | B/vector | TieredQ config | QPS | B/vector | Speed |")
    print("|---|---|---:|---:|---|---:|---:|---:|")
    for target in TARGETS:
        z, t = fastest(zvec, target), fastest(tieredq, target)
        if z is None or t is None:
            print(f"| {target:.2f} | {describe(z)} | | | {describe(t)} | | | |")
            continue
        print(f'| {target:.2f} | {describe(z)} | {z["qps"]:,.0f} | {z["index_bits"]/8:.1f} '
              f'| {describe(t)} | {t["qps"]:,.0f} | {t["index_bits"]/8:.1f} | {t["qps"]/z["qps"]:.2f}× |')
    print("\nTieredQ constrained to at most the fastest qualifying zvec configuration's index bytes:\n")
    print("| R@10 target | TieredQ config | QPS | B/vector | Speed vs zvec |")
    print("|---|---|---:|---:|---:|")
    for target in TARGETS:
        z = fastest(zvec, target)
        t = fastest(tieredq, target, z["index_bits"]) if z else None
        if t is None:
            print(f"| {target:.2f} | unreached | | | |")
            continue
        print(f'| {target:.2f} | {describe(t)} | {t["qps"]:,.0f} | {t["index_bits"]/8:.1f} '
              f'| {t["qps"]/z["qps"]:.2f}× |')
    print("\nSmallest qualifying index per engine (ties broken by QPS):\n")
    print("| R@10 target | zvec B/vector | QPS | TieredQ config | B/vector | QPS | Byte reduction |")
    print("|---|---:|---:|---|---:|---:|---:|")
    for target in TARGETS:
        z, t = smallest(zvec, target), smallest(tieredq, target)
        if z is None or t is None:
            print(f"| {target:.2f} | | | {describe(t)} | | | |")
            continue
        reduction = 1 - t["index_bits"] / z["index_bits"]
        print(f'| {target:.2f} | {z["index_bits"]/8:.1f} | {z["qps"]:,.0f} '
              f'| {describe(t)} | {t["index_bits"]/8:.1f} | {t["qps"]:,.0f} | {reduction:.1%} |')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.directory)
    samples = [json.loads(line) for path in args.directory.glob("*.jsonl")
               for line in path.read_text().splitlines()]
    print(f"{len(rows)} configurations; {sum(len(row['passes']) for row in samples)} timed passes.\n")
    comparison(rows)


if __name__ == "__main__":
    main()
