"""One-command reproducible matrix; subprocesses isolate randomized codec state."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# Bits per original dimension accepted by both native codecs.
BIT_BUDGETS = range(1, 9)
FIELDS = "dataset,d,base_n,queries,metric,codec,config,target_bits,effective_bits_per_dim,effective_bytes_per_vector,build_s,encode_vps,recall10,qps,p50_ms,p95_ms,threads,batch,rerank,seed".split(",")


def command(args):
    return subprocess.check_output(args, text=True).strip()


def git_metadata(*args):
    result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def worker(args):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from data import prepare, codec_vectors
    from measure import measure
    import codec_native
    from l2_transform import L2Transform
    with threadpool_limits(limits=1):
        base, queries, truth = prepare(args.data, args.datasets[0], args.max_base, args.queries)
        d = base.shape[1]
        lift = args.datasets[0] == "sift" and args.codec == "tieredq"
        codec_native.set_threads(8)
        start = time.perf_counter()
        transform = L2Transform.fit(base) if lift and not args.legacy_l2_lift else None
        vectors, probes = (transform.base_vectors(base), transform.queries(queries)) if transform else codec_vectors(base, queries, lift)
        index = codec_native.Index(vectors, args.bits[0], args.seeds[0], args.codec == "tieredq", args.datasets[0] == "sift")
        build_s = time.perf_counter() - start
        with tempfile.TemporaryDirectory(dir=args.output) as temporary:
            total_bytes = index.storage(str(Path(temporary) / "index.bin"))
        if transform:
            total_bytes += transform.storage_bytes
        common = dict(dataset=args.datasets[0], d=d, base_n=len(base), queries=len(queries),
                      metric="l2" if args.datasets[0] == "sift" else "cosine", codec=args.codec,
                      config=("fast-native-query-metric-lift" if transform else "fast-native") if args.codec == "tieredq" else ("flat-fht-full-score-hacc" if args.full_score else "flat-fht-native-bounds-hacc"),
                      target_bits=args.bits[0], effective_bits_per_dim=total_bytes*8/len(base)/d,
                      effective_bytes_per_vector=total_bytes/len(base), build_s=build_s,
                      encode_vps=len(base)/index.encode_seconds, rerank=False, seed=args.seeds[0])
        if args.datasets[0] == "sift" and args.codec == "tieredq":
            from measure import L2Search
            scale = transform.scale if transform else float(np.max(np.linalg.norm(base, axis=1)))
            index = L2Search(index, scale, transform)
            probes = queries
        recall_only = None
        if args.seeds[0] != 0:
            from measure import recall
            ids = np.empty((len(probes), 10), dtype=np.uint32)
            for offset in range(0, len(probes), 32):
                index.search(probes[offset:offset+32], ids[offset:offset+32], 8)
            recall_only = dict(recall10=recall(truth, ids), qps=float("nan"),
                               p50_ms=float("nan"), p95_ms=float("nan"))
        rows = []
        for threads in args.threads:
            for batch in args.batches:
                metrics = measure(index, probes, truth, threads, batch) if recall_only is None else recall_only
                rows.append(common | metrics | dict(threads=threads, batch=batch))
                print(f"{args.datasets[0]} {args.codec} b={args.bits[0]} seed={args.seeds[0]} t={threads} batch={batch}: {metrics}", flush=True)
        args.cell.write_text(json.dumps(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["sift", "glove", "openai"], default=["sift", "glove", "openai"])
    parser.add_argument("--bits", nargs="+", type=int, choices=BIT_BUDGETS, default=[1,2,4,8], metavar="{1..8}")
    parser.add_argument("--threads", nargs="+", type=int, choices=[1,8], default=[1,8])
    parser.add_argument("--batches", nargs="+", type=int, choices=[1,32], default=[1,32])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--max-base", type=int)
    parser.add_argument("--data", type=Path, default=HERE/"data")
    parser.add_argument("--output", type=Path, default=HERE/"output")
    parser.add_argument("--full-score", action="store_true", help="Disable native RaBitQ extra-bit bound pruning")
    parser.add_argument("--legacy-l2-lift", action="store_true", help="Use the original unconditioned TieredQ L2 lift")
    parser.add_argument("--codec", choices=["tieredq", "rabitq"])
    parser.add_argument("--cell", type=Path)
    args = parser.parse_args()
    if args.max_base is not None and args.max_base < 10:
        parser.error("--max-base must be at least 10")
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    if not args.codec and 0 not in args.seeds:
        parser.error("include seed 0 for timing and reports")
    args.data, args.output = args.data.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.codec:
        worker(args)
        return
    from setup_native import REVISION
    from native_build import build_native
    native = build_native(full_score=args.full_score)
    build, overlay, compiler_commands = native.directory, native.overlay, native.compiler_commands
    env = os.environ | {"PYTHONPATH": str(build), "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "8"}
    # Preparing and caching truth is intentionally outside measured matrix wall time.
    from data import prepare
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=8):
        for dataset in args.datasets:
            prepare(args.data, dataset, args.max_base, args.queries)
    metadata = {"validation_only": args.max_base is not None or args.queries < 1000, "cpu": command(["lscpu"]), "ram": command(["free", "-b"]), "os": platform.platform(),
                "compiler": command(["c++", "--version"]), "compiler_commands": compiler_commands, "flags": "-O3 -march=native -ffp-contract=off",
                "tieredq_commit": (REPO/"SOURCE_REVISION").read_text().strip() if (REPO/"SOURCE_REVISION").exists() else git_metadata("rev-parse", "HEAD"),
                "tieredq_checkout_commit": git_metadata("rev-parse", "HEAD"),
                "tieredq_dirty": git_metadata("status", "--porcelain"),
                "rabitq_commit": REVISION,
                "rabitq_search": {"container":"direct-flat", "implicit_row_ids":True, "high_accuracy":True, "extra_bit_bounds":not args.full_score}, "rabitq_patches": json.loads((overlay/"patches.json").read_text()),
                "arguments": sys.argv[1:], "python": sys.version,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "harness_sha256": {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [*HERE.glob("*.py"), HERE/"run.sh", HERE/"validate.sh", HERE/"uv.lock", HERE/"datasets.json"]},
                "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for folder in [REPO/"src", REPO/"include", HERE/"native"] for p in folder.rglob("*") if p.is_file()}}
    metadata["status"] = "running"
    (args.output/"meta.json").write_text(json.dumps(metadata, indent=2))
    start = time.perf_counter()
    with (args.output/"results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for dataset in args.datasets:
            for codec in ("tieredq", "rabitq"):
                for bits in args.bits:
                    for seed in args.seeds:
                        cell = args.output/"cell.json"
                        subprocess.run([sys.executable, str(HERE/"run.py"), "--codec", codec,
                                        "--datasets", dataset, "--bits", str(bits), "--seeds", str(seed),
                                        "--threads", *map(str,args.threads), "--queries", str(args.queries),
                                        "--batches", *map(str,args.batches),
                                        "--data", str(args.data), "--output", str(args.output), "--cell", str(cell),
                                        *(["--full-score"] if args.full_score else []),
                                        *(["--legacy-l2-lift"] if args.legacy_l2_lift else []),
                                        *(["--max-base", str(args.max_base)] if args.max_base else [])],
                                       env=env | {"BENCH_SEED": str(seed)}, check=True)
                        writer.writerows(json.loads(cell.read_text()))
                        output.flush()
    metadata["matrix_wall_s"] = time.perf_counter() - start
    metadata["status"] = "complete"
    (args.output/"meta.json").write_text(json.dumps(metadata, indent=2))
    subprocess.run([sys.executable, str(HERE/"report.py"), str(args.output)], check=True)
    diagnostic_start = time.perf_counter()
    subprocess.run([sys.executable, str(HERE/"profile.py"), str(args.output), "--data", str(args.data),
                    *(["--legacy-l2-lift"] if args.legacy_l2_lift else [])], check=True)
    metadata["profiling_wall_s"] = time.perf_counter() - diagnostic_start
    (args.output/"meta.json").write_text(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
