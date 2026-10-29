"""Seed a match_memory output with already measured reference cells (RaBitQ / TQ+ at 1/2/4 bits)."""
import csv, sys
from pathlib import Path
output, datasets, sources = Path(sys.argv[1]), sys.argv[2].split(","), sys.argv[3:]
rows = {}
for source in sources:
    for row in csv.DictReader(open(source)):
        if row["codec"] != "tieredq" and row["dataset"] in datasets and row["target_bits"] in ("1", "2", "4"):
            rows[(row["dataset"], row["codec"], row["target_bits"], row["seed"])] = row | {"requested_code_bits": "0"}
output.mkdir(parents=True, exist_ok=True)
fields = list(next(iter(rows.values())).keys())
with (output / "cells.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows.values())
print(output, len(rows), "reference cells")
