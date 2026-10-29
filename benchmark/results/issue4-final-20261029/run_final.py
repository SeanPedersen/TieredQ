"""Fresh timing matrix for one codec: 4 datasets x 1/2/4 bits x seeds 0-2 (seed 0 timed, 1-2 recall only).

FINAL_CODEC picks tieredq (default) or qdrant (TQ+); FINAL_SOURCE and FINAL_OUTPUT pick the source
tree and output directory. RaBitQ rows are reused from issue4-range-refresh-20261027; see finalize.py.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CODEC = os.environ.get("FINAL_CODEC", "tieredq")
SOURCE = ROOT / os.environ.get("FINAL_SOURCE", "source")
OUTPUT = ROOT / os.environ.get("FINAL_OUTPUT", "output")
QDRANT = ROOT / "qdrant"
DATA = Path(os.environ["FINAL_DATA"])
DATASETS = ["sift", "glove", "openai", "gemini"]
BITS = [1, 2, 4]
SEEDS = [0, 1, 2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def execute(command, **kwargs):
    print("RUN", *map(str, command), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def input_hashes():
    import numpy as np
    from data import prepare
    from threadpoolctl import threadpool_limits
    hashes = {}
    with threadpool_limits(limits=8):
        for name in DATASETS:
            base, queries, truth = prepare(DATA, name)
            hashes[name] = dict(base_shape=list(base.shape), query_shape=list(queries.shape),
                                sha256={k: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                                        for k, a in [("base", base), ("queries", queries), ("truth", truth)]})
            print("Verified input", name, hashes[name]["base_shape"], flush=True)
    return hashes


def main():
    OUTPUT.mkdir(exist_ok=True)
    sys.path.insert(0, str(SOURCE / "benchmark"))
    from native_build import build_native
    from run import FIELDS
    meta = dict(status="running", codec=CODEC, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                tieredq_commit=(SOURCE / "SOURCE_REVISION").read_text().strip(),
                flags="-O3 -march=native -ffp-contract=off", affinity=sorted(os.sched_getaffinity(0)))
    write_json(OUTPUT / "meta.json", meta)
    native = build_native()
    meta["compiler_commands"] = native.compiler_commands
    meta["source_sha256"] = {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for folder in [SOURCE / "src", SOURCE / "include", SOURCE / "benchmark/native"]
                             for p in folder.rglob("*") if p.is_file()}
    if CODEC == "qdrant":
        library = SOURCE / "benchmark/turboquant/native/target/release/libqdrant_bench.so"
        meta["qdrant_shim_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        meta["qdrant_commit"] = subprocess.check_output(["git", "-C", QDRANT, "rev-parse", "HEAD"], text=True).strip()
        meta["qdrant_patch"] = subprocess.check_output(["git", "-C", QDRANT, "diff"], text=True)
    write_json(OUTPUT / "inputs.json", input_hashes())
    execute(["cmake", "-S", SOURCE, "-B", ROOT / f"tests-build-{SOURCE.name}", "-DTIEREDQ_EXAMPLES=OFF",
             "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_FLAGS=-march=native"])
    execute(["cmake", "--build", ROOT / f"tests-build-{SOURCE.name}", "-j", "8"])
    execute(["ctest", "--test-dir", ROOT / f"tests-build-{SOURCE.name}", "--output-on-failure"])
    env = os.environ | {"PYTHONPATH": str(native.directory), "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "8"}
    started = time.perf_counter()
    cells = [(d, b, s) for d in DATASETS for b in BITS for s in SEEDS]
    with (OUTPUT / "fresh-results.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for number, (dataset, bits, seed) in enumerate(cells, 1):
            cell = OUTPUT / f"cell-{dataset}-{CODEC}-{bits}-{seed}.json"
            execute([sys.executable, SOURCE / "benchmark/run.py", "--codec", CODEC, "--datasets", dataset,
                     "--bits", bits, "--seeds", seed, "--data", DATA, "--output", OUTPUT, "--cell", cell],
                    env=env | {"BENCH_SEED": str(seed)}, cwd=SOURCE / "benchmark")
            writer.writerows(json.loads(cell.read_text()))
            out.flush()
            print(f"PROGRESS {number}/{len(cells)}", flush=True)
    meta.update(status="matrix_complete", matrix_wall_s=time.perf_counter() - started)
    write_json(OUTPUT / "meta.json", meta)
    print("MATRIX COMPLETE", flush=True)


if __name__ == "__main__":
    main()
