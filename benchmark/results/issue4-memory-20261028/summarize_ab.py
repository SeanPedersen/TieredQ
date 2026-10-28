"""Summarize ab/ab.csv: ID parity, memory deltas and after/before timing ratios per cell."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
SCHEDULES = ["t1-b1", "t1-b32", "t8-b1", "t8-b32"]
BITS_PER_BYTE = 8


def main():
    rows = list(csv.DictReader((ROOT / "ab" / "ab.csv").open()))
    base_n = {"sift": 1_000_000, "glove": 100_000, "openai": 100_000, "gemini": 99_000}
    cells, lines = [], ["| Dataset | Bits | IDs equal | Load peak before → after (bits/vector) | "
                        "Query peak before → after | t1-b1 QPS | t8-b1 QPS | t1-b32 QPS | t8-b32 QPS | t1-b1 p50 |",
                        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in dict.fromkeys((row["dataset"], row["bits"]) for row in rows):
        group = [row for row in rows if (row["dataset"], row["bits"]) == key]
        side = lambda variant, field: mean(float(row[field]) for row in group if row["variant"] == variant)
        per_vector = lambda variant, field: side(variant, field) * BITS_PER_BYTE / base_n[key[0]]
        ratio = lambda field: side("after", field) / side("before", field)
        cell = dict(dataset=key[0], bits=int(key[1]),
                    ids_equal=len({row["ids_sha256"] for row in group}) == 1,
                    accounted_equal=len({row["ram_accounted_bytes"] for row in group}) == 1,
                    load_peak_bits=[per_vector(v, "ram_load_peak_rss_bytes") for v in ("before", "after")],
                    query_peak_bits=[per_vector(v, "ram_query_peak_rss_bytes") for v in ("before", "after")],
                    qps_ratio={s: ratio(f"{s}_qps") for s in SCHEDULES},
                    p50_ratio={s: ratio(f"{s}_p50_ms") for s in SCHEDULES})
        cells.append(cell)
        lines.append(f"| {cell['dataset']} | {cell['bits']} | {cell['ids_equal'] and cell['accounted_equal']} | "
                     f"{cell['load_peak_bits'][0]:,.1f} → {cell['load_peak_bits'][1]:,.1f} | "
                     f"{cell['query_peak_bits'][0]:,.1f} → {cell['query_peak_bits'][1]:,.1f} | "
                     + " | ".join(f"{cell['qps_ratio'][s]:.3f}×" for s in ("t1-b1", "t8-b1", "t1-b32", "t8-b32"))
                     + f" | {cell['p50_ratio']['t1-b1']:.3f}× |")
    qps = [value for cell in cells for value in cell["qps_ratio"].values()]
    summary = dict(cells=cells, all_ids_equal=all(c["ids_equal"] and c["accounted_equal"] for c in cells),
                   qps_ratio_min=min(qps), qps_ratio_max=max(qps), qps_ratio_mean=mean(qps))
    (ROOT / "ab" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "ab" / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print({k: v for k, v in summary.items() if k != "cells"})


if __name__ == "__main__":
    main()
