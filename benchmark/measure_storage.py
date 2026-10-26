"""Refresh reported serving bytes without running recall, latency or throughput benchmarks."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
from threadpoolctl import threadpool_limits

from data import BASE_COUNTS, prepare
from l2_transform import L2Transform

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REPORTED_BUDGETS = {"sift": (2, 4, 8), "glove": (4, 5, 6, 8), "openai": (2, 3, 4, 8)}
DEFAULT_SEEDS = (0, 1, 2)
QUERY_COUNT = 1000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    points = sorted({(row["dataset"], row["bits"]) for row in rows})
    return [dict(dataset=dataset, bits=bits,
                 seeds=[row["seed"] for row in rows if (row["dataset"], row["bits"]) == (dataset, bits)],
                 effective_bytes_per_vector=max(row["effective_bytes_per_vector"] for row in rows
                                                if (row["dataset"], row["bits"]) == (dataset, bits)))
            for dataset, bits in points]


def measure_point(native, vectors, dataset, bits, seed, adapter_bytes, output_directory):
    index = native.Index(vectors, bits, seed, True, dataset == "sift")
    with tempfile.TemporaryDirectory(dir=output_directory) as temporary:
        path = Path(temporary) / "index.tq"
        serving_bytes = index.storage(str(path))
        persisted_bytes = path.stat().st_size
    return dict(dataset=dataset, bits=bits, seed=seed, base_n=len(vectors),
                adapter_bytes=adapter_bytes, persisted_index_bytes=persisted_bytes,
                effective_index_bytes=serving_bytes,
                effective_bytes_per_vector=(serving_bytes + adapter_bytes) / len(vectors),
                diagnostics=index.diagnostics())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=HERE / "build-native")
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=REPORTED_BUDGETS, default=list(REPORTED_BUDGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-base", type=int, help="Small validation runs only")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output file")
    if args.threads < 1 or any(seed < 0 for seed in args.seeds):
        parser.error("threads must be positive and seeds nonnegative")
    if args.max_base is not None and args.max_base < 10:
        parser.error("--max-base must be at least 10")
    sys.path.insert(0, str(args.native.resolve()))
    import codec_native

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", validation_only=args.max_base is not None,
                  protocol="storage-only; max(saved index bytes, reloaded cache + serving codec bytes) + adapter",
                  seeds=args.seeds, threads=args.threads, platform=platform.platform(),
                  native_sha256=digest(Path(codec_native.__file__)),
                  source_sha256={str(path.relative_to(REPO)): digest(path)
                                 for folder in (REPO / "src", REPO / "include", HERE / "native")
                                 for path in sorted(folder.rglob("*")) if path.is_file()},
                  harness_sha256={name: digest(HERE / name)
                                  for name in ("measure_storage.py", "data.py", "l2_transform.py", "datasets.json")},
                  compiler_commands=json.loads((args.native / "compile_commands.json").read_text()),
                  data_manifests={}, rows=[])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    with threadpool_limits(limits=1):
        for dataset in args.datasets:
            count = min(args.max_base or BASE_COUNTS[dataset], BASE_COUNTS[dataset])
            cache = args.data / f"{dataset}-{count}-{QUERY_COUNT}-v2.npz"
            if not cache.exists():
                raise FileNotFoundError(f"Storage refresh requires an existing dataset cache: {cache}")
            manifest = cache.with_suffix(".json")
            report["data_manifests"][dataset] = json.loads(manifest.read_text())
            base, _, _ = prepare(args.data, dataset, args.max_base, QUERY_COUNT)
            transform = L2Transform.fit(base) if dataset == "sift" else None
            vectors = transform.base_vectors(base) if transform else base
            adapter_bytes = transform.storage_bytes if transform else 0
            for bits in REPORTED_BUDGETS[dataset]:
                for seed in args.seeds:
                    print(f"Measuring storage: {dataset} {bits}-bit seed {seed}", flush=True)
                    codec_native.set_threads(args.threads)
                    row = measure_point(codec_native, vectors, dataset, bits, seed, adapter_bytes, args.output.parent)
                    report["rows"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(f"  {row['effective_bytes_per_vector']:.6f} bytes/vector", flush=True)
    report.update(status="complete", summary=summarize(report["rows"]))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
