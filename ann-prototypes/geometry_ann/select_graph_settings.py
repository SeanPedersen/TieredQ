"""Freeze calibration-selected settings and paired geometry ablations before reading test queries."""
import argparse
import csv
import json
from pathlib import Path

BOUNDS = ("none", "radius", "pca-max", "pca", "shared")
TARGETS = (0.95, 0.99)
FIELDS = ("routing", "bounds", "weight", "metadata_cap", "vector_cap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.calibration.open() as source:
        rows = list(csv.DictReader(source))
    selected, decisions = {}, []

    def add(row: dict[str, str]) -> None:
        values = tuple(row[field] for field in FIELDS)
        selected[values] = dict(zip(FIELDS, values, strict=True))

    def best(route: str, bounds: str, target: float) -> dict[str, str] | None:
        candidates = [row for row in rows if row["routing"] == route and row["bounds"] == bounds
                      and float(row["recall10"]) >= target]
        return min(candidates, key=lambda row: float(row["p50_us"])) if candidates else None

    for target in TARGETS:
        baseline = best("graph", "none", target)
        if baseline:
            for mode in BOUNDS:
                for weight in ((0,) if mode == "none" else (0, 0.5, 1)):
                    paired = dict(baseline, bounds=mode, weight=str(weight))
                    add(paired)
        for mode in BOUNDS:
            row = best("graph", mode, target)
            decisions.append(dict(target=target, routing="graph", bounds=mode, setting=row))
            if row:
                add(row)
        for route, mode in (("flat", "none"), ("hierarchy", "radius")):
            row = best(route, mode, target)
            decisions.append(dict(target=target, routing=route, bounds=mode, setting=row))
            if row:
                add(row)
    with args.output.open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected.values())
    with args.output.with_suffix(".json").open("x") as output:
        json.dump(dict(calibration=str(args.calibration), decisions=decisions), output, indent=2)
        output.write("\n")
    print(f"Frozen {len(selected)} settings; calibration targets and paired comparisons recorded")


if __name__ == "__main__":
    main()
