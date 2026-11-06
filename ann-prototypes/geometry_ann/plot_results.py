# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Plot measured recall at strict vector caps from the Instagram pilot CSV artifacts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    count = json.loads((args.results / "manifest.json").read_text())["base_shape"][0]
    curves = (
        ("V0-variance-caps.csv", "flat", "Fixed ~1K leaves, direct ranking"),
        ("V0-4096-caps.csv", "flat", "Fixed ~4K leaves, direct ranking"),
        ("V1-4096-caps.csv", "hierarchy", "Adaptive 4K, coarse bounds"),
        ("V1-8192-caps.csv", "hierarchy", "Adaptive 8K, coarse bounds"),
    )
    figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for filename, routing, label in curves:
        with (args.results / filename).open() as source:
            rows = [row for row in csv.DictReader(source)
                    if row["routing"] == routing and row["ordering"] == "centroid"]
        rows.sort(key=lambda row: int(row["vector_cap"]))
        axis.plot([100 * int(row["vector_cap"]) / count for row in rows],
                  [float(row["recall10"]) for row in rows], marker="o", label=label)
    for target in (0.95, 0.99):
        axis.axhline(target, color="gray", linestyle="--", linewidth=0.8)
        axis.text(1.03, target + 0.002, f"{target:.2f} target", color="dimgray", fontsize=9)
    axis.set(xlabel="Hard vector-scan cap (% of base)", ylabel="Recall@10",
             title="Instagram: recall under strict scan budgets", ylim=(0.75, 1.005))
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right", fontsize=9)
    figure.savefig(args.results / "recall-vs-scan-cap.svg")
    figure.savefig(args.results / "recall-vs-scan-cap.png", dpi=160)


if __name__ == "__main__":
    main()
