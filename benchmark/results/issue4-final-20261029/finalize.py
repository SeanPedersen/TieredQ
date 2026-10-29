"""TieredQ vs Qdrant TQ+ and RaBitQ at equal measured memory: audit inputs, write matched tables and plots.

For each reference point (codec x dataset x 1/2/4 bits/dim) and each memory metric (disk, steady RAM, peak RAM),
TieredQ on the final library is matched to the reference's measured bits/vector (final-match-{a,b}/matches-*.json):
recall is the mean over seeds 0-2, speed is seed 0 at the matched budget (output-final-matched-timing) against the
reference's own seed-0 timing. Output: matched.md, matched.csv, recall-memory-{qdrant,rabitq}.png.

Reference timing: Qdrant TQ+ from output-qdrant/ (fresh); RaBitQ from issue4-range-refresh-20261027 (same inputs,
adapter and timing harness). Compression ratio = FP32 bits per vector (32 * d) / target bits per vector.
"""
import csv
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT.parents[1]
sys.path.insert(0, str(BENCHMARK))
from plotting import plot_matched

RESULTS = ROOT.parent
REFRESH = RESULTS / "issue4-range-refresh-20261027"
DATASETS = ["sift", "glove", "openai", "gemini"]
REFERENCES = ["qdrant", "rabitq"]
BITS = [1, 2, 4]
METRICS = ["disk", "ram", "peak"]
SCHEDULES = [("1", "1"), ("1", "32"), ("8", "1"), ("8", "32")]
FP32_BITS = 32
DIMENSIONS = {"sift": 128, "glove": 200, "openai": 1536, "gemini": 768}
NAMES = {"tieredq": "TieredQ", "rabitq": "RaBitQ", "qdrant": "Qdrant TQ+"}
MATCH_DIRECTORIES = ("final-match-a", "final-match-b")
MATCHED_TIMING = ROOT / "output-final-matched-timing/matched-timing.csv"


def read(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_reference_timing():
    """Seed-0 timing rows of both references at 1/2/4 bits, keyed by source for provenance."""
    rows = [(row, "output-qdrant/fresh-results.csv") for row in read(ROOT / "output-qdrant/fresh-results.csv")]
    rows += [(row, "issue4-range-refresh-20261027/output/results.csv")
             for row in read(REFRESH / "output/results.csv") if row["codec"] == "rabitq"]
    timed = [(row, source) for row, source in rows if row["seed"] == "0" and int(row["target_bits"]) in BITS]
    keys = [(r["codec"], r["dataset"], int(r["target_bits"]), r["threads"], r["batch"]) for r, _ in timed]
    expected = {(c, d, b, t, n) for c, d, b in itertools.product(REFERENCES, DATASETS, BITS) for t, n in SCHEDULES}
    assert len(keys) == len(set(keys)) and set(keys) == expected, "reference timing grid incomplete"
    for row, _ in timed:
        assert all(math.isfinite(float(row[f])) for f in ("qps", "p50_ms", "p95_ms")), row
    return timed


def audit_inputs():
    reference = json.loads((REFRESH / "output/inputs.json").read_text())
    inputs = json.loads((ROOT / "output-qdrant/inputs.json").read_text())
    for dataset in DATASETS:
        assert inputs[dataset]["sha256"] == reference[dataset]["sha256"], dataset
    return {dataset: reference[dataset]["sha256"] for dataset in DATASETS}


def load_matches():
    results = []
    for directory in MATCH_DIRECTORIES:
        for reference in REFERENCES:
            results += [r for r in json.loads((ROOT / directory / f"matches-{reference}.json").read_text())
                        if r["reference_bits"] in BITS]
    keys = [(r["reference_codec"], r["dataset"], r["reference_bits"], r["metric"]) for r in results]
    assert len(keys) == len(set(keys)) and set(keys) == set(itertools.product(REFERENCES, DATASETS, BITS, METRICS)), \
        "matched grid incomplete"
    return results


def schedule_metrics(rows):
    by = {(r["threads"], r["batch"]): r for r in rows}
    return dict(qps_t1_b32=float(by[("1", "32")]["qps"]), qps_t8_b32=float(by[("8", "32")]["qps"]),
                p50_ms_t1_b1=float(by[("1", "1")]["p50_ms"]))


def group_metrics(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return {k: schedule_metrics(v) for k, v in groups.items()}


def audit_matched_recall(timing):
    """Timed runs rebuild each matched budget; their seed-0 recall must equal the match cells' seed-0 recall."""
    # Timing is keyed by the requested budget; a cell records the realised one when an odd bit stays unallocated.
    cells = {(r["dataset"], r["requested_code_bits"] or r["code_bits"]): float(r["recall10"])
             for directory in MATCH_DIRECTORIES for r in read(ROOT / directory / "cells.csv")
             if r["codec"] == "tieredq" and r["seed"] == "0"}
    for row in timing:
        assert abs(float(row["recall10"]) - cells[(row["dataset"], row["code_bits"])]) < 1e-12, row


def matched_rows(matches, references, tieredq_speed):
    rows = []
    for m in matches:
        reference = references[(m["reference_codec"], m["dataset"], m["reference_bits"])]
        row = dict(reference_codec=m["reference_codec"], dataset=m["dataset"], reference_bits=m["reference_bits"],
                   metric=m["metric"], target_bits_per_vector=m["target_bits_per_vector"],
                   compression=FP32_BITS * DIMENSIONS[m["dataset"]] / m["target_bits_per_vector"],
                   reference_recall10=m["reference"]["recall10"]["mean"], reference_recall10_std=m["reference"]["recall10"]["std"],
                   code_bits=m["code_bits"], win=m["win"]) | {f"reference_{k}": v for k, v in reference.items()}
        if m["tieredq"]:
            speed = tieredq_speed[(m["dataset"], m["code_bits"])]
            row |= dict(tieredq_recall10=m["tieredq"]["recall10"]["mean"], tieredq_recall10_std=m["tieredq"]["recall10"]["std"],
                        tieredq_memory_max=m["tieredq"][m["metric"]]["max"]) | {f"tieredq_{k}": v for k, v in speed.items()}
        rows.append(row)
    return rows


def matched_markdown(rows):
    parts = []
    for codec in REFERENCES:
        name = NAMES[codec]
        parts += [f"### TieredQ vs {name} at equal measured memory", "",
                  f"Target = {name}'s measured bits/vector (min over seeds); TieredQ's max over seeds is at or below it. "
                  "Ratio = FP32 / target. Recall is the mean over seeds 0-2; speed is seed 0 at the matched budget.", "",
                  f"| Dataset | {name} bits/dim | Matched on | Target bits/vec | Ratio | TieredQ R@10 | {name} R@10 | Δ pts "
                  f"| TieredQ QPS 8t·b32 | {name} QPS 8t·b32 | TieredQ QPS 1t·b32 | {name} QPS 1t·b32 | TieredQ p50 1t·b1 | {name} p50 1t·b1 |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in (r for r in rows if r["reference_codec"] == codec):
            head = (f"| {r['dataset']} | {r['reference_bits']} | {r['metric']} | {r['target_bits_per_vector']:,.1f} | "
                    f"{r['compression']:.1f}× | ")
            reference = f"{r['reference_recall10']:.4f}"
            if "tieredq_recall10" not in r:
                parts.append(head + f"none fits | {reference} | – | – | {r['reference_qps_t8_b32']:,.0f} | – | "
                             f"{r['reference_qps_t1_b32']:,.0f} | – | {r['reference_p50_ms_t1_b1']:.2f} |")
                continue
            delta = 100 * (r["tieredq_recall10"] - r["reference_recall10"])
            tieredq = f"**{r['tieredq_recall10']:.4f}**" if delta > 0 else f"{r['tieredq_recall10']:.4f}"
            parts.append(head + f"{tieredq} | {reference} | {delta:+.1f} | {r['tieredq_qps_t8_b32']:,.0f} | "
                         f"{r['reference_qps_t8_b32']:,.0f} | {r['tieredq_qps_t1_b32']:,.0f} | {r['reference_qps_t1_b32']:,.0f} | "
                         f"{r['tieredq_p50_ms_t1_b1']:.2f} | {r['reference_p50_ms_t1_b1']:.2f} |")
        parts.append("")
    return "\n".join(parts)


def main():
    inputs = audit_inputs()
    reference_timing = load_reference_timing()
    tieredq_timing = read(MATCHED_TIMING)
    audit_matched_recall(tieredq_timing)
    references = group_metrics([r for r, _ in reference_timing], lambda r: (r["codec"], r["dataset"], int(r["target_bits"])))
    tieredq_speed = group_metrics(tieredq_timing, lambda r: (r["dataset"], int(r["code_bits"])))
    matched = matched_rows(load_matches(), references, tieredq_speed)
    write_csv(ROOT / "matched.csv", matched, sorted({k for r in matched for k in r}))
    (ROOT / "matched.md").write_text(matched_markdown(matched))
    provenance = {"/".join((r["codec"], r["dataset"], r["target_bits"], r["threads"], r["batch"])): source
                  for r, source in reference_timing}
    (ROOT / "reference-timing-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (ROOT / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    for reference in REFERENCES:
        plot_matched(ROOT, matched, reference)
    for reference in REFERENCES:
        subset = [r for r in matched if r["reference_codec"] == reference]
        print(f"vs {NAMES[reference]}: {sum(r['win'] for r in subset)}/{len(subset)} TieredQ wins")


if __name__ == "__main__":
    main()
