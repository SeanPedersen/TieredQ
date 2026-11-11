"""Disk and peak-RAM bits/vector per dataset × codec × budget, each phase in a fresh process.

Disk is the self-contained saved index (plus the SIFT TieredQ L2 lift). RAM is reported three ways:
component accounting of the reloaded index, steady RSS growth after reload, and peak RSS growth
(VmHWM) during reload and, separately, during the full timed query workload; the overall peak is
their maximum. Recall is a reload check against the
8-thread batch-32 benchmark rows; per-schedule recall is kept in each cell's serve.json. Build peak is RSS growth from the loaded
FP32 corpus through training, encoding and packing. RSS deltas start from a reset high-water mark.
A cell spec may carry an exact TieredQ `code_bits` budget in place of bits × dimensions (see match_memory.py), and a
run.py TieredQ `variant` (default fine).
With `quick`, the query workload is one untimed pass per schedule: same allocations and peak, no timing.
A dataset with a companion query set on the same base (MS MARCO's doc queries) also scores that set at the primary
schedule after all memory reads, so one match yields both recalls (`companion_recall10`).
"""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATASETS = ["sift", "glove", "openai", "gemini", "openai1m", "msmarco1m", "msmarco1m-doc"]
# Second query set scored on the same base and index (#22: real vs doc queries); storage does not depend on it.
COMPANION_QUERIES = {"msmarco1m": "msmarco1m-doc"}
CODECS = ["tieredq", "rabitq"]
# Qdrant TQ+ needs the Rust shim (codec_adapters/qdrant_turboquant_plus/native), so it is opt-in via --codecs.
OPTIONAL_CODECS = ["qdrant"]
BIT_BUDGETS = [1, 2, 4]
SUPPORTED_BITS = range(1, 9)
# Same scheduling grid as the timed matrix; the workload sets the serving high-water mark.
SERVING_SCHEDULES = [(1, 1), (1, 32), (8, 1), (8, 32)]
PRIMARY_THREADS, PRIMARY_BATCH = 8, 32
PRIMARY_SCHEDULE = f"t{PRIMARY_THREADS}-b{PRIMARY_BATCH}"
BUILD_THREADS = 8
RESET_ATTEMPTS = 5
RESET_PAUSE_S = 0.02
FIELDS = ("dataset,d,base_n,codec,target_bits,code_bits,seed,disk_bytes,disk_bits_per_vector,"
          "ram_accounted_bytes,ram_accounted_bits_per_vector,ram_loaded_rss_bytes,ram_loaded_bits_per_vector,"
          "ram_load_peak_rss_bytes,ram_load_peak_bits_per_vector,ram_query_peak_rss_bytes,ram_query_peak_bits_per_vector,"
          "ram_peak_rss_bytes,ram_peak_bits_per_vector,build_peak_rss_bytes,build_peak_bits_per_vector,"
          "build_s,recall10,companion_recall10").split(",")


def memory_status():
    fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    kib = lambda key: int(fields[key].split()[0]) * 1024
    return kib("VmRSS"), kib("VmHWM")


def reset_peak():
    """Reset VmHWM to the current RSS (Linux clear_refs 5) and return that baseline.

    The kernel sets VmHWM to RSS at the reset, so the post-reset VmHWM is the baseline. Pages released
    between the reset and the status read (e.g. reclaim on a busy host) only lower RSS, so a few short
    retries look for a quiet read before falling back to the post-reset VmHWM.
    """
    gc.collect()
    for _ in range(RESET_ATTEMPTS):
        Path("/proc/self/clear_refs").write_text("5")
        rss, peak = memory_status()
        if peak == rss:
            return rss
        time.sleep(RESET_PAUSE_S)
    return peak


def uses_lift(spec):
    """TieredQ serves SIFT with its native L2 metric (the lift is part of the model); no adapter lift remains."""
    return False


def index_path(spec, cell):
    """TQ+ saves a directory (metadata + codes); the native codecs save one file."""
    return cell / ("index" if spec["codec"] == "qdrant" else "index.bin")


def build_index(spec, vectors):
    import codec_native
    if spec["codec"] == "qdrant":
        from qdrant_codec import QdrantIndex
        return QdrantIndex.build(vectors, spec["bits"], spec["dataset"] == "sift")
    if spec["codec"] == "tieredq":
        from run import tieredq_index
        return tieredq_index(vectors, spec["bits"], spec["seed"], spec["dataset"] == "sift", spec.get("code_bits", 0),
                             spec.get("variant", "fine"))
    return codec_native.Index(vectors, spec["bits"], spec["seed"], False, spec["dataset"] == "sift")


def load_index(spec, cell):
    import codec_native
    if spec["codec"] == "qdrant":
        from qdrant_codec import QdrantIndex
        return QdrantIndex.load(index_path(spec, cell))
    return codec_native.Index.load(str(index_path(spec, cell)), spec["codec"] == "tieredq")


def preload_codec(spec):
    """Load codec libraries before the RSS baseline so their code pages are not counted as index memory."""
    import codec_native
    if spec["codec"] == "qdrant":
        from qdrant_codec import QdrantIndex
        QdrantIndex._lib()
    return codec_native


def index_sha256(path):
    from qdrant_codec import directory_sha256
    return directory_sha256(path) if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()


def build_worker(spec, cell):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from data import prepare
    from l2_transform import L2Transform
    with threadpool_limits(limits=1):
        base, _, _ = prepare(Path(spec["data"]), spec["dataset"])
        preload_codec(spec).set_threads(BUILD_THREADS)
        baseline = reset_peak()
        start = time.perf_counter()
        transform = L2Transform.fit(base) if uses_lift(spec) else None
        vectors = transform.base_vectors(base) if transform else base
        index = build_index(spec, vectors)
        build_s = time.perf_counter() - start
        build_peak = memory_status()[1] - baseline
        code_bits = (sum(tier["dimensions"] * tier["bits"] for tier in index.diagnostics()["tiers"])
                     if spec["codec"] == "tieredq" else None)
        index_bytes = index.save(str(index_path(spec, cell)))
        transform_bytes = 0
        if transform:
            np.savez(cell / "transform.npz", scale=transform.scale, base_map=transform.base_map,
                     query_map=transform.query_map)
            transform_bytes = (cell / "transform.npz").stat().st_size
        return dict(d=base.shape[1], base_n=len(base), code_bits=code_bits, build_s=build_s,
                    build_peak_rss_bytes=build_peak,
                    disk_bytes=index_bytes + transform_bytes,
                    index_sha256=index_sha256(index_path(spec, cell)))


def single_pass(index, queries, truth, threads, batch, recall):
    import numpy as np
    ids = np.empty((len(queries), 10), dtype=np.uint32)
    for start in range(0, len(queries), batch):
        index.search(queries[start:start + batch], ids[start:start + batch], threads)
    return {"recall10": recall(truth, ids)}


def serve_worker(spec, cell):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from data import prepare
    from l2_transform import L2Transform
    from measure import L2Search, measure, recall
    with threadpool_limits(limits=1):
        base, queries, truth = prepare(Path(spec["data"]), spec["dataset"])
        del base
        companion = spec["dataset"] in COMPANION_QUERIES
        if companion:
            base, companion_queries, companion_truth = prepare(Path(spec["data"]), COMPANION_QUERIES[spec["dataset"]])
            del base
        preload_codec(spec)
        baseline = reset_peak()
        index = load_index(spec, cell)
        accounted = index.resident_bytes()
        if uses_lift(spec):
            with np.load(cell / "transform.npz") as saved:
                transform = L2Transform(float(saved["scale"]), saved["base_map"], saved["query_map"])
            accounted += transform.storage_bytes
            index = L2Search(index, transform.scale, transform)
        loaded_rss, load_hwm = memory_status()
        # Second reset separates the reload transient from steady query-time usage.
        reset_peak()
        metrics = {f"t{threads}-b{batch}": (single_pass(index, queries, truth, threads, batch, recall) if spec.get("quick")
                                              else measure(index, queries, truth, threads, batch))
                   for threads, batch in SERVING_SCHEDULES}
        load_peak, query_peak = load_hwm - baseline, memory_status()[1] - baseline
        ids = np.empty((len(queries), 10), dtype=np.uint32)
        index.search(queries, ids, BUILD_THREADS)
        companion_recall = (single_pass(index, companion_queries, companion_truth, PRIMARY_THREADS, PRIMARY_BATCH,
                                        recall)["recall10"]
                            if companion else None)
        return dict(ram_accounted_bytes=accounted, ram_loaded_rss_bytes=loaded_rss - baseline,
                    ram_load_peak_rss_bytes=load_peak, ram_query_peak_rss_bytes=query_peak,
                    ram_peak_rss_bytes=max(load_peak, query_peak), recall10=metrics[PRIMARY_SCHEDULE]["recall10"],
                    ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(), companion_recall10=companion_recall,
                    # Diagnostic only: memory runs are not the headline timing runs.
                    schedule_metrics=metrics)


def to_row(spec, built, served):
    row = dict(dataset=spec["dataset"], d=built["d"], base_n=built["base_n"], codec=spec["codec"],
               target_bits=spec["bits"], code_bits=built["code_bits"], seed=spec["seed"], build_s=built["build_s"], recall10=served["recall10"],
               companion_recall10=served["companion_recall10"])
    measured = built | served
    for key in ("disk", "ram_accounted", "ram_loaded_rss", "ram_load_peak_rss", "ram_query_peak_rss",
                "ram_peak_rss", "build_peak_rss"):
        row[f"{key}_bytes"] = measured[f"{key}_bytes"]
        row[f"{key.removesuffix('_rss')}_bits_per_vector"] = measured[f"{key}_bytes"] * 8 / built["base_n"]
    return row


def cell_name(spec):
    budget = f"c{spec['code_bits']}" if spec.get("code_bits") else spec["bits"]
    variant = f"-{spec['variant']}" if spec.get("variant", "fine") != "fine" else ""
    return f"cell-{spec['dataset']}-{spec['codec']}{variant}-{budget}-{spec['seed']}"


def run_cell(spec, output, env):
    cell = output / cell_name(spec)
    cell.mkdir(exist_ok=True)
    results = {}
    for phase in ("build", "serve"):
        subprocess.run([sys.executable, str(HERE / "measure_memory.py"), "--worker", phase,
                        "--spec", json.dumps(spec), "--cell", str(cell)],
                       env=env | {"BENCH_SEED": str(spec["seed"])}, check=True)
        results[phase] = json.loads((cell / f"{phase}.json").read_text())
    for artifact in ("index.bin", "transform.npz"):
        (cell / artifact).unlink(missing_ok=True)
    shutil.rmtree(cell / "index", ignore_errors=True)
    row = to_row(spec, results["build"], results["serve"])
    print(f"{cell_name(spec)}: disk {row['disk_bits_per_vector']:.1f}, "
          f"peak RAM {row['ram_peak_bits_per_vector']:.1f} bits/vector", flush=True)
    return row


def metadata(compiler_commands, arguments):
    command = lambda args: subprocess.check_output(args, text=True).strip()
    return {"cpu": command(["lscpu"]), "ram": command(["free", "-b"]), "os": platform.platform(),
            "compiler": command(["c++", "--version"]), "compiler_commands": compiler_commands,
            "tieredq_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                             text=True).stdout.strip() or None,
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "arguments": arguments, "python": sys.version,
            "serving_schedules": SERVING_SCHEDULES, "build_threads": BUILD_THREADS,
            "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for folder in [REPO / "src", REPO / "include", HERE / "codec_adapters"]
                              for p in folder.rglob("*") if p.is_file()}}


def prepare_native():
    """Build the native module; return the worker environment and run metadata."""
    from adapters_build import build_adapters
    native = build_adapters()
    env = os.environ | {"PYTHONPATH": os.pathsep.join([str(native.directory), str(HERE)]),
                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": str(BUILD_THREADS)}
    return env, metadata(native.compiler_commands, sys.argv[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS+["glove1m"], default=DATASETS)
    parser.add_argument("--codecs", nargs="+", choices=CODECS + OPTIONAL_CODECS, default=CODECS)
    parser.add_argument("--bits", nargs="+", type=int, choices=SUPPORTED_BITS, default=BIT_BUDGETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE / "output-memory")
    parser.add_argument("--worker", choices=["build", "serve"])
    parser.add_argument("--spec")
    parser.add_argument("--cell", type=Path)
    parser.add_argument("--quick", action="store_true", help="memory-only query workload, no timing")
    args = parser.parse_args()
    if args.worker:
        worker = build_worker if args.worker == "build" else serve_worker
        result = worker(json.loads(args.spec), args.cell)
        (args.cell / f"{args.worker}.json").write_text(json.dumps(result, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    env, meta = prepare_native()
    start = time.perf_counter()
    with (args.output / "memory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for dataset in args.datasets:
            for codec in args.codecs:
                for bits in args.bits:
                    for seed in args.seeds:
                        spec = dict(dataset=dataset, codec=codec, bits=bits, seed=seed, data=str(args.data.resolve()),
                                    quick=args.quick)
                        writer.writerow(run_cell(spec, args.output, env))
                        handle.flush()
    meta["wall_s"] = time.perf_counter() - start
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
