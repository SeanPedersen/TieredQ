"""Measured operating points and budget-constrained cross-codec comparisons."""
import numpy as np


def operating_points(rows):
    datasets = sorted({r["dataset"] for r in rows})
    summary = []
    # Recall means across seeds; timing comes only from seed zero, by thread count.
    for dataset in datasets:
        for threads in sorted({int(row["threads"]) for row in rows}):
            for codec in ("tieredq", "rabitq"):
                operating = []
                for bits in sorted({int(r["target_bits"]) for r in rows}):
                    group = [r for r in rows if r["dataset"]==dataset and r["codec"]==codec
                             and int(r["threads"])==threads and int(r["batch"])==32 and int(r["target_bits"])==bits]
                    if not group:
                        continue
                    timed = next((r for r in group if int(r["seed"])==0), None)
                    if timed is None:
                        continue
                    single = next((r for r in rows if r["dataset"]==dataset and r["codec"]==codec
                                  and int(r["threads"])==threads and int(r["batch"])==1
                                  and int(r["target_bits"])==bits and int(r["seed"])==0), None)
                    recalls = [float(r["recall10"]) for r in group]
                    operating.append({"bits":bits, "recall_mean":float(np.mean(recalls)),
                                      "recall_std":float(np.std(recalls)), "bytes_per_vector":max(float(r["effective_bytes_per_vector"]) for r in group),
                                      "qps_seed0":float(timed["qps"]), "build_s_seed0":float(timed["build_s"]),
                                      "p50_ms_seed0":float(single["p50_ms"]) if single else None,
                                      "p95_ms_seed0":float(single["p95_ms"]) if single else None,
                                      "encode_vps_seed0":float(timed["encode_vps"])})
                reached = [r for r in operating if r["recall_mean"]>=.95]
                summary.append({"dataset":dataset,"threads":threads,"codec":codec,"points":operating,
                                "minimum_measured_at_095":min(reached,key=lambda r:r["bytes_per_vector"]) if reached else None})
    return summary


def compare(points):
    comparisons = []
    for anchor in points:
        if anchor["codec"] != "rabitq":
            continue
        tieredq = next(p for p in points if p["dataset"] == anchor["dataset"] and p["threads"] == anchor["threads"] and p["codec"] == "tieredq")
        for point in anchor["points"]:
            affordable = [p for p in tieredq["points"] if p["bytes_per_vector"] <= point["bytes_per_vector"]]
            comparable = [p for p in tieredq["points"] if p["recall_mean"] >= point["recall_mean"]]
            comparisons.append({"dataset": anchor["dataset"], "threads": anchor["threads"], "rabitq_anchor": point,
                                "tieredq_best_recall_within_budget": max(affordable, key=lambda p:p["recall_mean"]) if affordable else None,
                                "tieredq_fastest_at_least_anchor_recall": max(comparable, key=lambda p:p["qps_seed0"]) if comparable else None})
    return comparisons
