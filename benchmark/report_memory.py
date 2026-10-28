"""Render memory.csv from measure_memory.py as a Markdown table of disk/RAM bits per vector."""
import csv
from pathlib import Path
import sys

COLUMNS = [("Dataset", "dataset"), ("Codec", "codec"), ("Bits/dim", "target_bits"),
           ("Disk", "disk_bits_per_vector"), ("RAM accounted", "ram_accounted_bits_per_vector"),
           ("RAM loaded RSS", "ram_loaded_bits_per_vector"), ("RAM load peak", "ram_load_peak_bits_per_vector"),
           ("RAM query peak", "ram_query_peak_bits_per_vector"), ("RAM peak RSS", "ram_peak_bits_per_vector"),
           ("Build peak RSS", "build_peak_bits_per_vector"), ("Recall@10 (seed 0)", "recall10")]


def cell(key, value):
    if key in {"dataset", "codec", "target_bits"}:
        return value
    return f"{float(value):.4f}" if key == "recall10" else f"{float(value):,.1f}"


def table(rows):
    lines = ["| " + " | ".join(title for title, _ in COLUMNS) + " |",
             "|" + "|".join("---" if index < 2 else "---:" for index in range(len(COLUMNS))) + "|"]
    lines += ["| " + " | ".join(cell(key, row[key]) for _, key in COLUMNS) + " |" for row in rows]
    return "\n".join(lines)


def main(output):
    with (output / "memory.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    (output / "memory-table.md").write_text("All storage columns are bits per vector.\n\n" + table(rows) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
