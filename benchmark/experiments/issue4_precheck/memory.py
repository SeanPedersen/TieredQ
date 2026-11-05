"""One measure_memory cell (quick: untimed serving passes) for the #4 precheck; appends its row to OUTPUT/memory.csv.

Usage: memory.py OUTPUT DATA DATASET CODEC VARIANT BITS SEED [CODE_BITS]   (VARIANT is ignored for the references;
CODE_BITS overrides a TieredQ budget)
"""
import csv
import os
from pathlib import Path
import sys

from measure_memory import FIELDS, run_cell


def main():
    output, data, dataset, codec, variant, bits, seed, *code_bits = sys.argv[1:]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = dict(dataset=dataset, codec=codec, bits=int(bits), seed=int(seed), data=data, quick=True)
    if codec == "tieredq":
        spec["variant"] = variant
        if code_bits:
            spec["code_bits"] = int(code_bits[0])
    row = run_cell(spec, output, dict(os.environ)) | dict(variant=variant if codec == "tieredq" else "",
                                                           requested_code_bits=spec.get("code_bits", ""))
    path = output / "memory.csv"
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*FIELDS, "variant", "requested_code_bits"])
        if new_file:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
