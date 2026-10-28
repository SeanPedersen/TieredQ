"""A/B the streamed TieredQ loader: same saved index files, old vs new library, ABAB order.

Each TieredQ cell is built once with the new tree, then served in fresh processes by both
library builds. Compares top-10 ID hashes, accounted bytes, load/query peak RSS and QPS.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TREES = {"before": ROOT / "source-before", "after": ROOT / "source"}
DATA = ROOT / "data"
OUTPUT = ROOT / "ab"
DATASETS = ["sift", "glove", "openai", "gemini"]
BITS = [1, 2, 4]
ORDER = ["before", "after", "before", "after"]
SCRIPT = TREES["after"] / "benchmark" / "measure_memory.py"


def environment(tree):
    benchmark = TREES["after"] / "benchmark"
    return os.environ | {"PYTHONPATH": os.pathsep.join([str(tree / "benchmark" / "build-native"), str(benchmark)]),
                         "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "8",
                         "BENCH_SEED": "0"}


def worker(phase, spec, cell, tree):
    subprocess.run([sys.executable, str(SCRIPT), "--worker", phase, "--spec", json.dumps(spec), "--cell", str(cell)],
                   env=environment(tree), check=True)
    return json.loads((cell / f"{phase}.json").read_text())


def main():
    for tree in TREES.values():
        subprocess.run([sys.executable, "-c", "from native_build import build_native; build_native()"],
                       cwd=tree / "benchmark", check=True)
    OUTPUT.mkdir(exist_ok=True)
    rows = []
    for dataset in DATASETS:
        for bits in BITS:
            spec = dict(dataset=dataset, codec="tieredq", bits=bits, seed=0, data=str(DATA))
            cell = OUTPUT / f"cell-{dataset}-{bits}"
            cell.mkdir(exist_ok=True)
            worker("build", spec, cell, TREES["after"])
            for repeat, variant in enumerate(ORDER):
                served = worker("serve", spec, cell, TREES[variant])
                row = dict(dataset=dataset, bits=bits, variant=variant, repeat=repeat,
                           **{key: served[key] for key in ("ram_accounted_bytes", "ram_loaded_rss_bytes",
                                                           "ram_load_peak_rss_bytes", "ram_query_peak_rss_bytes",
                                                           "recall10", "ids_sha256")})
                for schedule, metrics in served["schedule_metrics"].items():
                    row |= {f"{schedule}_qps": metrics["qps"], f"{schedule}_p50_ms": metrics["p50_ms"],
                            f"{schedule}_p95_ms": metrics["p95_ms"]}
                rows.append(row)
                print(dataset, bits, variant, repeat, "load peak", row["ram_load_peak_rss_bytes"],
                      "t8-b32 qps", row["t8-b32_qps"], flush=True)
            for artifact in ("index.bin", "transform.npz"):
                (cell / artifact).unlink(missing_ok=True)
            with (OUTPUT / "ab.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
