"""Measure the repository flat RaBitQ adapter on the pinned Gemini/DBpedia dataset.

The output bundle mirrors the TurboQuant experiment's scheduling matrix and raw
artifacts so all three codecs can be compared on the same normalized corpus and
held-out document queries.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "build-adapters"))
from measure import recall  # noqa: E402
import codec_native  # noqa: E402

TOP_K = 10
ANCHOR_BITS = (1, 2, 4)
SCHEDULE = [(threads, batch) for threads in (1, 8) for batch in (1, 32)]
WARMUPS = 3
REPEATS = 3
VALIDATION_QUERIES = 65
BASE_N = 99_000
QUERY_N = 1_000
DIMENSIONS = 768


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_array(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


class RaBitQ:
    codec = "rabitq"

    def __init__(self, base, bits, path, seed):
        self.path = Path(path)
        self.dim = base.shape[1]
        codec_native.set_threads(8)
        tick = time.perf_counter()
        self.index = codec_native.Index(base, bits, seed, False, False)
        self.build_s = time.perf_counter() - tick
        self.encode_seconds = self.index.encode_seconds
        tick = time.perf_counter()
        measured = self.index.storage(str(self.path))
        self.storage_write_s = time.perf_counter() - tick
        self.disk_bytes = self.path.stat().st_size
        self.serving_bytes = max(measured, self.disk_bytes)

    def details(self):
        return dict(
            disk_bytes=self.disk_bytes,
            resident_bytes=self.serving_bytes,
            serving_bytes=self.serving_bytes,
            bytes_per_vector=self.serving_bytes / BASE_N,
            file_sha256=sha_file(self.path),
            storage_write_s=self.storage_write_s,
        )

    def search(self, queries, output, threads):
        self.index.search(queries, output, threads)

    def close(self):
        self.index = None


def predictions(index, queries, threads, batch):
    ids = np.empty((len(queries), TOP_K), dtype=np.uint32)
    for start in range(0, len(queries), batch):
        index.search(queries[start:start + batch], ids[start:start + batch], threads)
    return ids


def validate(index, queries, label):
    probes = np.ascontiguousarray(queries[:VALIDATION_QUERIES])
    expected = predictions(index, probes, 1, 1)
    assert np.all(np.diff(np.sort(expected, axis=1), axis=1) > 0), (label, "duplicate ids")
    for threads, batch in SCHEDULE:
        assert np.array_equal(expected, predictions(index, probes, threads, batch)), (label, threads, batch)
    return dict(label=label, query_count=len(probes), serial_batch_thread_ids_equal=True,
                reload_ids_equal=None,
                reload_note="The existing flat RaBitQ adapter exposes storage bytes but no load path; predictions are not reload-tested.")


def measure(index, queries, truth, threads, batch, reference, label, out):
    ids = np.empty((len(queries), TOP_K), dtype=np.uint32)
    for i in range(WARMUPS):
        index.search(queries[i:i + 1], ids[:1], threads)
    durations = []
    for repeat in range(REPEATS):
        for start in range(0, len(queries), batch):
            stop = min(start + batch, len(queries))
            tick = time.perf_counter()
            index.search(queries[start:stop], ids[start:stop], threads)
            durations.append(dict(repeat=repeat, start=start, count=stop - start,
                                  seconds=time.perf_counter() - tick))
        assert np.array_equal(ids, reference), (label, threads, batch, repeat, "scheduling variation")
    latencies = np.repeat([entry["seconds"] * 1000 for entry in durations],
                          [entry["count"] for entry in durations])
    save(out / f"timings-{label}-{threads}-{batch}.json", durations)
    np.save(out / f"ids-{label}-{threads}-{batch}.npy", ids)
    return dict(recall10=recall(truth, ids),
                qps=len(queries) * REPEATS / sum(entry["seconds"] for entry in durations),
                p50_ms=float(np.percentile(latencies, 50)),
                p95_ms=float(np.percentile(latencies, 95)))


def load_dataset(data):
    with np.load(data / "gemini-prepared.npz") as prepared:
        base, queries, truth = (prepared[key] for key in ("base", "queries", "truth"))
    manifest = json.loads((data / "gemini-manifest.json").read_text())
    for name, array in (("base", base), ("queries", queries), ("truth", truth)):
        assert sha_array(array) == manifest["array_sha256"][name], name
    corpus_ids = np.load(data / "gemini-corpus-row-ids.npy")
    query_ids = np.load(data / "gemini-query-row-ids.npy")
    assert base.shape == (BASE_N, DIMENSIONS) and queries.shape == (QUERY_N, DIMENSIONS)
    assert truth.shape == (QUERY_N, TOP_K)
    assert not np.intersect1d(corpus_ids, query_ids).size
    assert np.array_equal(np.sort(np.concatenate([corpus_ids, query_ids])), np.arange(100_000))
    return base, queries, truth, manifest, corpus_ids, query_ids


def write_provenance(out, data, manifest, corpus_ids, query_ids, truth, arguments):
    provenance = out / "provenance"
    provenance.mkdir()
    np.save(provenance / "gemini-corpus-row-ids.npy", corpus_ids)
    np.save(provenance / "gemini-query-row-ids.npy", query_ids)
    np.save(out / "truth.npy", truth)
    (provenance / "gemini-manifest.json").write_text((data / "gemini-manifest.json").read_text())
    overlay = HERE / ".deps" / "rabitq-native"
    (provenance / "rabitq-patches.json").write_text((overlay / "patches.json").read_text())
    build = HERE / "build-adapters"
    (provenance / "compile_commands.json").write_text((build / "compile_commands.json").read_text())
    binary = next(build.glob("codec_native*.so"))
    (out / "binaries").mkdir()
    (out / "binaries" / binary.name).write_bytes(binary.read_bytes())
    save(provenance / "binary-sha256.json", {binary.name: sha_file(binary)})
    repo = HERE.parent
    save(provenance / "source-sha256.json", {
        str(path.relative_to(repo)): sha_file(path)
        for folder in (HERE / "codec_adapters", repo / "src", repo / "include")
        for path in folder.rglob("*") if path.is_file()
    } | {"benchmark/run_rabitq_gemini.py": sha_file(Path(__file__)),
         "measure.py": sha_file(HERE / "measure.py"),
         "setup_adapters.py": sha_file(HERE / "setup_adapters.py"),
         "adapters_build.py": sha_file(HERE / "adapters_build.py"),
         "build_policy.py": sha_file(HERE / "build_policy.py")})
    save(provenance / "host.json", {
        "hostname": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "cpu": command(["lscpu"]), "ram": command(["free", "-b"]),
        "compiler": command(["c++", "--version"]), "affinity": sorted(os.sched_getaffinity(0)),
        "seed_environment": os.environ.get("BENCH_SEED"),
        "rabitq_revision": command(["git", "-C", str(HERE / ".deps/RaBitQ-Library"), "rev-parse", "HEAD"]),
        "arguments": arguments,
    })
    (provenance / "processes.txt").write_text(command(["ps", "-eo", "pid,pcpu,pmem,comm", "--sort=-pcpu"]))


def audit(out, queries, truth):
    rows = list(csv.DictReader((out / "results.csv").open()))
    expected = {("rabitq", bits, threads, batch)
                for bits in ANCHOR_BITS for threads, batch in SCHEDULE}
    assert {(row["codec"], int(row["anchor_bits"]), int(row["threads"]), int(row["batch"]))
            for row in rows} == expected
    assert len(rows) == 12
    manifest = json.loads((out / "provenance/gemini-manifest.json").read_text())
    assert manifest["source"]["revision"] == "e14a0f193988e9efb7ccc57b4c615418e6540f62"
    assert manifest["original_shape"] == [100_000, DIMENSIONS]
    assert manifest["corpus_n"] == BASE_N and manifest["query_n"] == QUERY_N and manifest["finite"]
    corpus = np.load(out / "provenance/gemini-corpus-row-ids.npy")
    held_out = np.load(out / "provenance/gemini-query-row-ids.npy")
    assert len(corpus) == BASE_N and len(held_out) == QUERY_N
    assert not np.intersect1d(corpus, held_out).size
    assert np.array_equal(np.sort(np.concatenate([corpus, held_out])), np.arange(100_000))
    assert sha_array(corpus) == manifest["array_sha256"]["corpus_ids"]
    assert sha_array(held_out) == manifest["array_sha256"]["query_ids"]
    archived_truth = np.load(out / "truth.npy")
    assert sha_array(archived_truth) == manifest["array_sha256"]["truth"]
    validations = json.loads((out / "validation.json").read_text())
    assert len(validations) == 3
    assert all(v["serial_batch_thread_ids_equal"] for v in validations)
    anchors = {entry["anchor_bits"]: entry for entry in json.loads((out / "anchors.json").read_text())}
    per_bits = {}
    for row in rows:
        bits = int(row["anchor_bits"])
        threads = int(row["threads"])
        batch = int(row["batch"])
        label = f"rabitq-{bits}"
        for key in ("qps", "p50_ms", "p95_ms", "build_s", "encode_vps", "bytes_per_vector"):
            assert np.isfinite(float(row[key])) and float(row[key]) > 0, (label, key)
        assert int(row["base_n"]) == BASE_N and int(row["queries"]) == QUERY_N and int(row["seed"]) == 0
        assert float(row["p95_ms"]) >= float(row["p50_ms"])
        ids = np.load(out / f"ids-{label}-{threads}-{batch}.npy")
        reference = np.load(out / f"ids-reference-{label}.npy")
        assert np.array_equal(ids, reference), (label, "scheduling variation")
        assert ids.shape == (QUERY_N, TOP_K) and np.max(ids) < BASE_N
        assert np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0)
        measured_recall = float(np.mean([len(set(a) & set(b)) / TOP_K
                                        for a, b in zip(truth, ids, strict=True)]))
        assert abs(measured_recall - float(row["recall10"])) < 1e-14
        times = json.loads((out / f"timings-{label}-{threads}-{batch}.json").read_text())
        assert len(times) == REPEATS * ((QUERY_N + batch - 1) // batch)
        assert all(np.isfinite(entry["seconds"]) and entry["seconds"] > 0 for entry in times)
        for repeat in range(REPEATS):
            calls = [entry for entry in times if entry["repeat"] == repeat]
            assert [entry["start"] for entry in calls] == list(range(0, QUERY_N, batch))
            assert sum(entry["count"] for entry in calls) == QUERY_N
        qps = QUERY_N * REPEATS / sum(entry["seconds"] for entry in times)
        latencies = np.repeat([entry["seconds"] * 1000 for entry in times],
                              [entry["count"] for entry in times])
        assert np.isclose(qps, float(row["qps"]), rtol=1e-12)
        assert np.isclose(np.percentile(latencies, 50), float(row["p50_ms"]), rtol=1e-12)
        assert np.isclose(np.percentile(latencies, 95), float(row["p95_ms"]), rtol=1e-12)
        assert float(row["bytes_per_vector"]) == anchors[bits]["storage"]["serving_bytes"] / BASE_N
        per_bits.setdefault(bits, []).append(measured_recall)
    assert all(len(set(values)) == 1 for values in per_bits.values())
    commands = json.loads((out / "provenance/compile_commands.json").read_text())
    for item in commands:
        text = item.get("command", " ".join(item.get("arguments", [])))
        assert "-O3" in text and "-march=native" in text and "-ffp-contract=off" in text
        assert "-Os" not in text
    for name, expected_hash in json.loads((out / "provenance/binary-sha256.json").read_text()).items():
        assert sha_file(out / "binaries" / name) == expected_hash
    result = dict(status="PASS", scheduling_rows=12, unique_codec_anchor_pairs=3,
                  finite_timings=True, dataset_identity=True, disjoint_split=True,
                  native_validation=validations, serial_batch_thread_prediction_equality=True,
                  recall_variation_across_scheduling=False, reload_validation=False,
                  note="One seed and one run; no statistical-confidence claims. Reload is not exposed by the existing flat RaBitQ adapter.")
    save(out / "audit.json", result)
    return result


def figures(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary = [row for row in rows if int(row["threads"]) == 8 and int(row["batch"]) == 32]
    for metric, title, ylabel, name in (("recall10", "Recall@10", "Recall@10", "recall-storage.png"),
                                        ("qps", "8 threads, batch 32", "Queries/second", "qps-storage.png")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        points = sorted(primary, key=lambda row: int(row["anchor_bits"]))
        ax.plot([float(row["bytes_per_vector"]) for row in points],
                [float(row[metric]) for row in points], "o-", color="#d95f02", label="RaBitQ")
        for row in points:
            ax.annotate(f"A{row['anchor_bits']}", (float(row["bytes_per_vector"]), float(row[metric])),
                        xytext=(5, 6), textcoords="offset points")
        ax.set(xlabel="Measured serving bytes/vector", ylabel=ylabel,
               title=f"Gemini/DBpedia RaBitQ · single-seed (0) · {title}")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, threads in zip(axes, (1, 8), strict=True):
        points = sorted([row for row in rows if int(row["threads"]) == threads and int(row["batch"]) == 1],
                        key=lambda row: int(row["anchor_bits"]))
        for percentile, style in (("p50_ms", "o-"), ("p95_ms", "s--")):
            ax.plot([float(row["bytes_per_vector"]) for row in points],
                    [float(row[percentile]) for row in points], style, label=percentile[:3])
        ax.set(xlabel="Measured serving bytes/vector", ylabel="Batch-1 response latency (ms)",
               title=f"{threads} thread(s)")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Gemini/DBpedia RaBitQ · single-seed (0)")
    fig.tight_layout()
    fig.savefig(out / "latency-storage.png", dpi=160)
    plt.close(fig)


def readme(out, rows):
    primary = [row for row in rows if int(row["threads"]) == 8 and int(row["batch"]) == 32]
    lines = [
        "# Gemini/DBpedia: repository RaBitQ flat adapter",
        "",
        "Single-seed experiment (seed 0), Hartmut i9-7980XE, CPUs 0–7. All 12 scheduling rows passed the audit. No statistical-confidence claims. Queries are held-out document vectors, not independently embedded search queries.",
        "",
        "99,000 corpus rows and 1,000 disjoint held-out queries from the 100,000 × 768D Google Gemini/DBpedia dataset. Inputs are FP32-normalized and are the exact arrays used by the Qdrant TurboQuant/TieredQ bundle. Ground truth is exhaustive FP32 cosine top-10. Dataset revision: `e14a0f193988e9efb7ccc57b4c615418e6540f62`.",
        "",
        "| Pair | Codec | Code bits | Bytes/vector | Recall@10 | Build s | Encode vectors/s | QPS (8t/32) | 1t p50/p95 ms | 8t p50/p95 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        single = {int(item["threads"]): item for item in rows
                  if item["anchor_bits"] == row["anchor_bits"] and int(item["batch"]) == 1}
        lines.append(
            f"| A{row['anchor_bits']} | rabitq | {row['requested_code_bits']} | {float(row['bytes_per_vector']):.3f} | "
            f"{float(row['recall10']):.4f} | {float(row['build_s']):.3f} | {float(row['encode_vps']):,.0f} | "
            f"{float(row['qps']):,.1f} | {float(single[1]['p50_ms']):.3f}/{float(single[1]['p95_ms']):.3f} | "
            f"{float(single[8]['p50_ms']):.3f}/{float(single[8]['p95_ms']):.3f} |")
    meta = json.loads((out / "meta.json").read_text())
    lines += [
        "",
        "RaBitQ uses the repository's official pinned RaBitQ-Library adapter: flat row-position storage, no IVF container, FHT/Kac rotation, native HACC split-batch kernels with extra-bit bound pruning, and the seeded rotation overlay. One-bit rows store no extra-bit payload; 2/4-bit rows store 1/3 extra bits per dimension. Build time covers rotation, global mean and native encode/pack; saving the flat representation is measured separately.",
        "",
        f"Anchor construction wall time: {meta['anchor_wall_s']:.2f} s. Search matrix and artifact capture: {meta['search_matrix_wall_s']:.2f} s. Storage is max(saved flat bytes, adapter serving bytes); the existing adapter does not expose a load path, so reload identity is not tested. Search timings include rotation, full scan, extra-bit scoring and top-k.",
        "",
        "Three single-query warmups precede each scheduling point; all 1,000 queries run three timing repeats. Raw call times and predictions are retained. Batch-32 latency is whole-batch response latency, including the final partial batch. Scheduling recall is identical per anchor.",
        "",
        "`provenance/` contains the pinned dataset manifest, split IDs, truth hash, RaBitQ patch hashes, compiler commands, source and binary hashes and host state. `binaries/` contains the measured extension.",
        "",
        "![Recall](recall-storage.png)",
        "![Throughput](qps-storage.png)",
        "![Batch-1 latency](latency-storage.png)",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    args.data = args.data.resolve()
    os.environ["BENCH_SEED"] = str(args.seed)
    assert sorted(os.sched_getaffinity(0)) == list(range(8)), sorted(os.sched_getaffinity(0))

    with threadpool_limits(limits=1):
        base, queries, truth, manifest, corpus_ids, query_ids = load_dataset(args.data)
        write_provenance(out, args.data, manifest, corpus_ids, query_ids, truth, sys.argv[1:])
        metadata = dict(status="building-anchors", seed=args.seed, affinity=sorted(os.sched_getaffinity(0)),
                        dataset="gemini", base_n=BASE_N, queries=QUERY_N, dimensions=DIMENSIONS,
                        anchor_bits=list(ANCHOR_BITS), schedule=SCHEDULE,
                        selection_rule="Integer bits per dimension at the Qdrant/TieredQ nominal 1/2/4-bit anchors.")
        save(out / "meta.json", metadata)

        tick = time.perf_counter()
        anchors = []
        validations = []
        indexes = []
        for bits in ANCHOR_BITS:
            index = RaBitQ(base, bits, out / "models" / f"rabitq-{bits}.bin", args.seed)
            validations.append(validate(index, queries, f"rabitq-{bits}"))
            anchors.append(dict(anchor_bits=bits, requested_code_bits=bits * DIMENSIONS,
                                build_s=index.build_s, encode_vps=BASE_N / index.encode_seconds,
                                storage=index.details()))
            indexes.append(index)
            save(out / "anchors.json", anchors)
            save(out / "validation.json", validations)
            print("ANCHOR", anchors[-1], flush=True)
        metadata["anchor_wall_s"] = time.perf_counter() - tick
        metadata["status"] = "timing"
        save(out / "meta.json", metadata)

        rows = []
        search_start = time.perf_counter()
        for index, anchor in zip(indexes, anchors, strict=True):
            bits = anchor["anchor_bits"]
            label = f"rabitq-{bits}"
            reference = predictions(index, queries, 1, 1)
            np.save(out / f"ids-reference-{label}.npy", reference)
            for threads, batch in SCHEDULE:
                row = dict(codec=index.codec, anchor_bits=bits,
                           requested_code_bits=bits * DIMENSIONS, learned_code_bits=bits * DIMENSIONS,
                           base_n=BASE_N, queries=QUERY_N, seed=args.seed,
                           bytes_per_vector=anchor["storage"]["bytes_per_vector"],
                           disk_bytes=anchor["storage"]["disk_bytes"],
                           resident_bytes=anchor["storage"]["resident_bytes"],
                           anchor_bytes_per_vector=anchor["storage"]["bytes_per_vector"],
                           percent_below_anchor=0.0,
                           build_s=anchor["build_s"], encode_vps=anchor["encode_vps"],
                           threads=threads, batch=batch)
                row.update(measure(index, queries, truth, threads, batch, reference, label, out))
                rows.append(row)
                with (out / "results.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                print("MEASURED", row, flush=True)
            index.close()
        metadata.update(status="complete", search_matrix_wall_s=time.perf_counter() - search_start,
                        rows=len(rows))
        save(out / "meta.json", metadata)

    audit(out, queries, truth)
    figures(out, rows)
    readme(out, rows)


if __name__ == "__main__":
    main()
