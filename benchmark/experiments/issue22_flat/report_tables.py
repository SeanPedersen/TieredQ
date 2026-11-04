"""Issue #22 phase 1 flat tables on MS MARCO-1M: equal memory with real and doc queries side by side, then equal recall.

Reads a bundle laid out by job_flat.sh: match-b{1,2,4}/ (match_memory.py --quick on msmarco1m; every cell carries the
doc-query recall as companion_recall10), equal-recall/b*/equal-recall-{msmarco1m,msmarco1m-doc}.csv and timing/cell-*.json
(run.py, seed 0). Equal-recall speed is shown against the reference cell that set the recall level and against the
reference's fastest bits/dim whose mean recall on that query set reaches the level.
Usage: python report_tables.py BUNDLE > tables.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "issue4_1m"))
from report_tables import (BITS, DIMENSIONS, FLOAT_BITS, NAMES, REFERENCES, load_matches, ratio_cell, read_csv,
                           speed, speed_table, timing_rows)

DATASET = "msmarco1m"
QUERY_SETS = {"msmarco1m": ("real", "recall10"), "msmarco1m-doc": ("doc", "companion_recall10")}
FASTEST = {"qps_t8_b32": max, "qps_t1_b32": max, "p50_t1_b1": min}


def recall_table(matches, metrics):
    lines = ["| Reference | bits/dim | Compression | Matched on | Ref R@10 real | TieredQ R@10 real | Δ pt real "
             "| Ref R@10 doc | TieredQ R@10 doc | Δ pt doc |", "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for ref, name in REFERENCES.items():
        for bits in BITS:
            for metric in metrics:
                match = matches.get((DATASET, ref, bits, metric))
                if match is None:
                    continue
                compression = FLOAT_BITS * DIMENSIONS[DATASET] / match["target_bits_per_vector"]
                cells = []
                for _, key in QUERY_SETS.values():
                    reference = match["reference"][key]["mean"]
                    if match["tieredq"] is None:
                        cells += [f"{reference:.4f}", "none fits", "–"]
                        continue
                    recall = match["tieredq"][key]["mean"]
                    cells += [f"{reference:.4f}", f"{recall:.4f}", f"{100 * (recall - reference):+.2f}"]
                lines.append(f"| {name} | {bits} | {compression:.1f}× | {metric} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fastest(rows, dataset, ref, level, matches, key, pick):
    """Value and bits/dim of the reference's fastest timed cell whose mean recall on this query set reaches level."""
    candidates = []
    for bits in BITS:
        match = matches.get((DATASET, ref, bits, "ram"))
        timed = speed(rows, dataset, ref, bits)
        if match and timed and match["reference"][QUERY_SETS[dataset][1]]["mean"] >= level:
            candidates.append((timed[key], bits))
    return pick(candidates) if candidates else None


def reaches(point):
    return float(point["recall10_mean"]) >= float(point["reference_recall10"])


def equal_recall_table(bundle, matches, rows):
    lines = ["| Queries | Reference | bits/dim | Ref R@10 | TieredQ code bits | TieredQ R@10 | Stored bits TQ / ref "
             "| QPS 8t·b32 TQ / ref | QPS 1t·b32 TQ / ref | p50 1t·b1 ms TQ / ref "
             "| vs fastest ref: QPS 8t·b32 | QPS 1t·b32 | p50 1t·b1 |",
             "|---|---|---:|---:|---:|---:|---|---|---|---|---|---|---|"]
    found = {}
    for row in (r for path in sorted(bundle.glob("equal-recall/b*/equal-recall-*.csv")) for r in read_csv(path)):
        key = row["dataset"], row["reference_codec"], int(row["reference_bits"])
        # A follow-up row stepped up to reach the reference on the seed mean supersedes a seed-0 bisection that missed it.
        if key not in found or reaches(row) and not reaches(found[key]):
            found[key] = row
    for dataset, (label, _) in QUERY_SETS.items():
        for ref, name in REFERENCES.items():
            for bits in BITS:
                point = found.get((dataset, ref, bits))
                if point is None:
                    continue
                code_bits, level = int(point["code_bits"]), float(point["reference_recall10"])
                tieredq, reference = speed(rows, dataset, "tieredq", code_bits), speed(rows, dataset, ref, bits)
                head = (f"| {label} | {name} | {bits} | {level:.4f} | {code_bits} | {float(point['recall10_mean']):.4f} | ")
                if tieredq is None or reference is None:
                    lines.append(head + "not timed | | | | | | |")
                    continue
                best = []
                for key, pick in FASTEST.items():
                    value, best_bits = fastest(rows, dataset, ref, level, matches, key, pick)
                    best.append(f"{tieredq[key] / value:.2f}× ({best_bits} bit)")
                lines.append(head + " | ".join(ratio_cell(tieredq, reference, key, digits) for key, digits in
                                               (("bits", 0), ("qps_t8_b32", 0), ("qps_t1_b32", 0), ("p50_t1_b1", 2)))
                             + " | " + " | ".join(best) + " |")
    return "\n".join(lines)


def main(bundle):
    matches = load_matches({DATASET: sorted(bundle.glob("match-b*/"))})
    rows = timing_rows(bundle)
    print("## Recall@10 at equal measured disk and steady RAM (one match; both query sets scored on each index)\n")
    print(recall_table(matches, ["disk", "ram"]))
    print("\n## Appendix: recall@10 at equal peak RAM\n")
    print(recall_table(matches, ["peak"]))
    print("\n## Speed and build time at the RAM-matched budget (real queries, seed 0)\n")
    print(speed_table(matches, rows).replace(f"| {NAMES[DATASET]} ", "| real "))
    print("\n## Speed at equal recall (seed 0 timing; R@10 is the mean over seeds 0-2; p50: lower is better)\n")
    print(equal_recall_table(bundle, matches, rows))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
