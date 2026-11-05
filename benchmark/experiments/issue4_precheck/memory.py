"""One measure_memory cell (quick: untimed serving passes) for the #4 precheck; appends its row to OUTPUT/memory.csv.

Usage: memory.py OUTPUT DATA DATASET CODEC VARIANT BITS SEED   (VARIANT is ignored for the references)
"""
import csv
import os
from pathlib import Path
import sys

from measure_memory import FIELDS, run_cell


def main():
    output, data, dataset, codec, variant, bits, seed = sys.argv[1:]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = dict(dataset=dataset, codec=codec, bits=int(bits), seed=int(seed), data=data, quick=True)
    if codec == "tieredq":
        spec["variant"] = variant
    row = run_cell(spec, output, dict(os.environ)) | dict(variant=variant if codec == "tieredq" else "")
    path = output / "memory.csv"
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*FIELDS, "variant"])
        if new_file:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
