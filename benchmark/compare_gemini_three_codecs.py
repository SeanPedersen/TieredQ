"""Combine the measured Gemini/DBpedia bundles into one audited three-codec comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

TOP_K = 10
ANCHOR_BITS = (1, 2, 4)
SCHEDULE = {(threads, batch) for threads in (1, 8) for batch in (1, 32)}
CODEC_ORDER = ("rabitq", "qdrant-turboquant-plus", "tieredq-fast")
CODEC_LABELS = {"rabitq": "RaBitQ", "qdrant-turboquant-plus": "Qdrant TQ+", "tieredq-fast": "TieredQ fast"}
CODEC_COLORS = {"rabitq": "#d95f02", "qdrant-turboquant-plus": "#7570b3", "tieredq-fast": "#1b9e77"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def read_rows(bundle):
    with (bundle / "results.csv").open() as handle:
        return list(csv.DictReader(handle))


def verify_bundle(bundle):
    manifest = json.loads((bundle / "provenance/gemini-manifest.json").read_text())
    truth = np.load(bundle / "truth.npy")
    corpus = np.load(bundle / "provenance/gemini-corpus-row-ids.npy")
    queries = np.load(bundle / "provenance/gemini-query-row-ids.npy")
    assert array_digest(truth) == manifest["array_sha256"]["truth"]
    assert array_digest(corpus) == manifest["array_sha256"]["corpus_ids"]
    assert array_digest(queries) == manifest["array_sha256"]["query_ids"]
    assert manifest["original_shape"] == [100_000, 768]
    assert truth.shape == (1000, TOP_K)
    assert len(corpus) == 99_000 and len(queries) == 1_000
    assert not np.intersect1d(corpus, queries).size
    assert np.array_equal(np.sort(np.concatenate([corpus, queries])), np.arange(100_000))
    return manifest, truth, corpus, queries


def verify_rows(bundle, rows, truth):
    for row in rows:
        codec = row["codec"]
        bits = int(row["anchor_bits"])
        threads = int(row["threads"])
        batch = int(row["batch"])
        label = f"{codec}-{bits}"
        assert (threads, batch) in SCHEDULE
        assert int(row["base_n"]) == 99_000 and int(row["queries"]) == 1_000
        ids = np.load(bundle / f"ids-{label}-{threads}-{batch}.npy")
        reference = np.load(bundle / f"ids-reference-{label}.npy")
        assert np.array_equal(ids, reference)
        assert ids.shape == (1000, TOP_K) and np.max(ids) < 99000
        assert np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0)
        measured = float(np.mean([len(set(a) & set(b)) / TOP_K for a, b in zip(truth, ids, strict=True)]))
        assert abs(measured - float(row["recall10"])) < 1e-14
        times = json.loads((bundle / f"timings-{label}-{threads}-{batch}.json").read_text())
        assert len(times) == 3 * ((1000 + batch - 1) // batch)
        qps = 1000 * 3 / sum(entry["seconds"] for entry in times)
        latencies = np.repeat([entry["seconds"] * 1000 for entry in times], [entry["count"] for entry in times])
        assert np.isclose(qps, float(row["qps"]), rtol=1e-12)
        assert np.isclose(np.percentile(latencies, 50), float(row["p50_ms"]), rtol=1e-12)
        assert np.isclose(np.percentile(latencies, 95), float(row["p95_ms"]), rtol=1e-12)


def figures(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [row for row in rows if row["threads"] == "8" and row["batch"] == "32"]
    for metric, title, ylabel, name in (
        ("recall10", "Recall@10", "Recall@10", "recall-storage.png"),
        ("qps", "8 threads, batch 32", "Queries/second", "qps-storage.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for codec in CODEC_ORDER:
            points = sorted([row for row in primary if row["codec"] == codec],
                            key=lambda row: int(row["anchor_bits"]))
            ax.plot([float(row["bytes_per_vector"]) for row in points],
                    [float(row[metric]) for row in points], "o-",
                    color=CODEC_COLORS[codec], label=CODEC_LABELS[codec])
            for row in points:
                ax.annotate(f"A{row['anchor_bits']}", (float(row["bytes_per_vector"]), float(row[metric])),
                            xytext=(5, 6), textcoords="offset points", color=CODEC_COLORS[codec], fontsize=9)
        ax.set(xlabel="Measured serving bytes/vector", ylabel=ylabel,
               title=f"Gemini/DBpedia · single-seed (0) · {title}")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / name, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for codec in CODEC_ORDER:
        points = sorted([row for row in primary if row["codec"] == codec],
                        key=lambda row: int(row["anchor_bits"]))
        ax.plot([float(row["recall10"]) for row in points],
                [float(row["qps"]) for row in points], "o-",
                color=CODEC_COLORS[codec], label=CODEC_LABELS[codec])
        for row in points:
            ax.annotate(f"A{row['anchor_bits']}", (float(row["recall10"]), float(row["qps"])),
                        xytext=(5, 6), textcoords="offset points", color=CODEC_COLORS[codec], fontsize=9)
    ax.set(xlabel="Recall@10", ylabel="Queries/second · 8 threads, batch 32",
           title="Gemini/DBpedia · single-seed (0) · throughput versus recall")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "qps-recall.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, threads in zip(axes, (1, 8), strict=True):
        for codec in CODEC_ORDER:
            points = sorted([row for row in rows
                             if row["codec"] == codec and int(row["threads"]) == threads and row["batch"] == "1"],
                            key=lambda row: int(row["anchor_bits"]))
            for percentile, style in (("p50_ms", "o-"), ("p95_ms", "s--")):
                ax.plot([float(row["bytes_per_vector"]) for row in points],
                        [float(row[percentile]) for row in points], style,
                        alpha=1.0 if percentile == "p50_ms" else 0.55,
                        color=CODEC_COLORS[codec], label=f"{CODEC_LABELS[codec]} {percentile[:3]}")
        ax.set(xlabel="Measured serving bytes/vector", ylabel="Batch-1 response latency (ms)",
               title=f"{threads} thread(s)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Gemini/DBpedia · single-seed (0)")
    fig.tight_layout()
    fig.savefig(out / "latency-storage.png", dpi=160)
    plt.close(fig)


def readme(out, rows):
    primary = [row for row in rows if row["threads"] == "8" and row["batch"] == "32"]
    by_anchor = {}
    for row in primary:
        by_anchor.setdefault(int(row["anchor_bits"]), {})[row["codec"]] = row
    lines = [
        "# Gemini/DBpedia: TieredQ fast vs Qdrant TurboQuant+ vs RaBitQ",
        "",
        "Single-seed experiment (seed 0), Hartmut i9-7980XE, CPUs 0–7. All 36 scheduling rows passed the audit. No statistical-confidence claims. Queries are held-out document vectors, not independently embedded search queries.",
        "",
        "99,000 corpus rows and 1,000 disjoint held-out queries from the same 100,000 × 768D Gemini/DBpedia dataset. All three codecs use identical FP32-normalized arrays and exhaustive FP32 cosine top-10 ground truth. Dataset revision: `e14a0f193988e9efb7ccc57b4c615418e6540f62`.",
        "",
        "## Primary results (8 threads, batch 32)",
        "",
        "| Pair | Codec | Requested code bits | Bytes/vector | vs Qdrant anchor | Recall@10 | QPS | 1t p50/p95 ms | 8t p50/p95 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bits in ANCHOR_BITS:
        anchor = float(by_anchor[bits]["qdrant-turboquant-plus"]["bytes_per_vector"])
        for codec in CODEC_ORDER:
            row = by_anchor[bits][codec]
            single = {int(item["threads"]): item for item in rows
                      if item["codec"] == codec and item["anchor_bits"] == row["anchor_bits"] and item["batch"] == "1"}
            delta = 100 * (float(row["bytes_per_vector"]) - anchor) / anchor
            requested = row["requested_code_bits"] or str(bits * 768)
            lines.append(
                f"| A{bits} | {CODEC_LABELS[codec]} | {requested} | "
                f"{float(row['bytes_per_vector']):.3f} | {delta:+.3f}% | {float(row['recall10']):.4f} | "
                f"{float(row['qps']):,.1f} | {float(single[1]['p50_ms']):.3f}/{float(single[1]['p95_ms']):.3f} | "
                f"{float(single[8]['p50_ms']):.3f}/{float(single[8]['p95_ms']):.3f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- At the nominal 1/2/4-bit anchors, RaBitQ uses **3.0–5.9% more serving bytes** than the Qdrant anchor because its flat format stores native per-block factors and the rotation/center state. Its recall is above Qdrant TQ+ at 1 and 2 bits, and slightly below it at 4 bits (0.9508 versus 0.9521) while using 3.0% more storage. It is not a strictly equal-storage point.",
        "- TieredQ is the only codec at or below each Qdrant anchor (0.86%, 0.37%, 0.77% below) and has the highest recall at every anchor. Its 4-bit row uses learned 5/6-bit tiers under the same total code budget.",
        "- RaBitQ is much faster than Qdrant TQ+ at every anchor in batch 32. It is slower than TieredQ at 1 and 2 bits and faster at 4 bits, where TieredQ's learned high-bit path is more expensive.",
        "- These are compressed-only exhaustive searches with native top-k. Neither an ANN container nor original-vector reranking is used.",
        "",
        "## Measurement boundaries",
        "",
        "Each codec's storage is the measured serving bytes: `max(saved bytes, reloaded serving allocation)` for TieredQ, `max(saved codes/model, resident quantizer allocation)` for Qdrant, and the flat saved/RaBitQ allocation for RaBitQ. Input FP32 arrays, transient query/top-k/score scratch and language runtimes are excluded; this is not process RSS.",
        "",
        "RaBitQ and Qdrant timings include query preprocessing, full scan and top-k. TieredQ uses its native set-up and scan/selection scheduling. Build times are not compared here: the bundles have different construction boundaries (Qdrant includes calibration and persistence, TieredQ includes training and add/pack, RaBitQ includes rotation/mean and encode/pack).",
        "",
        "The Qdrant/TieredQ rows were measured in the issue #15 run; the RaBitQ rows were measured in a subsequent run on the same host and CPU set. The runs were not interleaved, so QPS and latency are subject to run-to-run host variation. No confidence intervals are claimed.",
        "",
        "Three single-query warmups precede each scheduling point; all 1,000 queries run three timing repeats and every repeat is checked against the serial reference. Raw timings and predictions for all codecs are retained in their source bundles. This combined directory contains the merged rows, independent audit and figures.",
        "",
        "## Source artifacts",
        "",
        "- Qdrant TQ+ / TieredQ: `../gemini-qdrant-turboquant-single-seed-20261026/`",
        "- RaBitQ: `../gemini-rabitq-single-seed-20261026/`",
        "",
        "Both bundles carry the same pinned dataset manifest and identical base/query/truth hashes. The RaBitQ bundle records a separate run and binary but uses the same prepared arrays.",
        "",
        "![Recall](recall-storage.png)",
        "![Throughput](qps-storage.png)",
        "![Throughput versus recall](qps-recall.png)",
        "![Batch-1 latency](latency-storage.png)",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--three-codec", type=Path,
                        default=Path(__file__).parent / "results/gemini-qdrant-turboquant-single-seed-20261026")
    parser.add_argument("--rabitq", type=Path,
                        default=Path(__file__).parent / "results/gemini-rabitq-single-seed-20261026")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifests = {"qdrant-tieredq": verify_bundle(args.three_codec), "rabitq": verify_bundle(args.rabitq)}
    assert manifests["qdrant-tieredq"][0]["array_sha256"] == manifests["rabitq"][0]["array_sha256"]
    truth = manifests["qdrant-tieredq"][1]
    assert np.array_equal(truth, manifests["rabitq"][1])

    rows = []
    for name, bundle, expected in (
        ("qdrant-tieredq", args.three_codec, {"qdrant-turboquant-plus", "tieredq-fast"}),
        ("rabitq", args.rabitq, {"rabitq"}),
    ):
        bundle_rows = read_rows(bundle)
        assert {row["codec"] for row in bundle_rows} == expected
        verify_rows(bundle, bundle_rows, truth)
        for row in bundle_rows:
            rows.append(dict(row) | {"source_bundle": name})
    assert len(rows) == 36
    assert {(row["codec"], int(row["anchor_bits"]), int(row["threads"]), int(row["batch"])) for row in rows} == {
        (codec, bits, threads, batch)
        for codec in CODEC_ORDER for bits in ANCHOR_BITS for threads, batch in SCHEDULE
    }
    for codec in CODEC_ORDER:
        recalls = {int(row["anchor_bits"]): {row["recall10"] for row in rows
                                             if row["codec"] == codec and int(row["anchor_bits"]) == bits}
                   for bits in ANCHOR_BITS}
        assert all(len(values) == 1 for values in recalls.values())
    fields = list(rows[0])
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (int(row["anchor_bits"]), CODEC_ORDER.index(row["codec"]),
                                                       int(row["threads"]), int(row["batch"]))))
    primary = [row for row in rows if row["threads"] == "8" and row["batch"] == "32"]
    with (out / "comparison-primary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(primary, key=lambda row: (int(row["anchor_bits"]), CODEC_ORDER.index(row["codec"]))))

    audit = dict(
        status="PASS", scheduling_rows=len(rows), primary_rows=len(primary), codecs=list(CODEC_ORDER),
        dataset_identity=True, split_disjoint=True, truth_hashes_equal=True,
        raw_predictions_reproduce_recall=True, raw_timings_reproduce_qps_and_latency=True,
        recall_variation_across_scheduling=False, storage_definition="max(saved_bytes, resident_serving_bytes)",
        note="Single seed and one run per codec; no statistical-confidence claims.",
        source_bundles={
            "qdrant-tieredq": {"path": str(args.three_codec.resolve()),
                               "results_sha256": digest(args.three_codec / "results.csv"),
                               "manifest_sha256": digest(args.three_codec / "provenance/gemini-manifest.json")},
            "rabitq": {"path": str(args.rabitq.resolve()),
                       "results_sha256": digest(args.rabitq / "results.csv"),
                       "manifest_sha256": digest(args.rabitq / "provenance/gemini-manifest.json")},
        },
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    figures(out, rows)
    readme(out, rows)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
