"""Reproduce the range-bound refresh with cached inputs and isolated codec workers."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / os.environ.get("REFRESH_SOURCE", "source")
OUTPUT = ROOT / os.environ.get("REFRESH_OUTPUT", "output")
DATA = Path("/home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze/benchmark/data")
GEMINI = Path("/home/sean/experiments/TieredQ-gemini-turboquant-20261026-vDAm7R/TieredQ/benchmark/data")
DATASETS = ["sift", "glove", "openai", "gemini"]
BITS = [1, 2, 4]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def execute(command, **kwargs):
    print("RUN", *map(str, command), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def cell(spec):
    sys.path.insert(0, str(SOURCE / "benchmark"))
    import data
    import numpy as np
    original = data.prepare
    def cached_prepare(root, name, max_base=None, query_count=1000):
        if name != "gemini":
            return original(root, name, max_base, query_count)
        with np.load(GEMINI / "gemini-prepared.npz") as arrays:
            return arrays["base"], arrays["queries"], arrays["truth"]
    data.prepare = cached_prepare
    import measure
    original_measure = measure.measure
    def captured_measure(index, probes, truth, threads, batch):
        metrics = original_measure(index, probes, truth, threads, batch)
        ids = np.empty((len(probes), 10), dtype=np.uint32)
        for offset in range(0, len(probes), batch):
            index.search(probes[offset:offset+batch], ids[offset:offset+batch], threads)
        label = Path(spec["cell"]).stem
        np.save(OUTPUT / f"{label}-t{threads}-b{batch}-ids.npy", ids)
        native_index = getattr(index, "index", index)
        if spec["codec"] == "tieredq":
            write_json(OUTPUT / f"{label}-diagnostics.json", native_index.diagnostics())
        return metrics
    measure.measure = captured_measure
    from run import worker
    args = dict(data=DATA, max_base=None, queries=1000, legacy_l2_lift=False,
                full_score=False, output=OUTPUT, threads=[1,8], batches=[1,32])
    args.update(spec)
    args["cell"] = Path(args["cell"])
    worker(SimpleNamespace(**args))


def main():
    if len(sys.argv) > 1:
        cell(json.loads(sys.argv[1]))
        return
    OUTPUT.mkdir(exist_ok=True)
    sys.path.insert(0, str(SOURCE / "benchmark"))
    import numpy as np
    from data import prepare
    from threadpoolctl import threadpool_limits
    from native_build import build_native
    from run import FIELDS
    meta = dict(status="running", started=time.time(), reuse_rabitq=True,
                flags="-O3 -march=native -ffp-contract=off", affinity=sorted(os.sched_getaffinity(0)))
    write_json(OUTPUT / "meta.json", meta)
    native = build_native()
    meta["compiler_commands"] = native.compiler_commands
    meta["source_sha256"] = {str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in [SOURCE/"src", SOURCE/"include", SOURCE/"benchmark/native"]
        for p in folder.rglob("*") if p.is_file()}
    inputs = {}
    with threadpool_limits(limits=8):
        for name in DATASETS:
            if name == "gemini":
                with np.load(GEMINI / "gemini-prepared.npz") as a:
                    base, queries, truth = a["base"], a["queries"], a["truth"]
                manifest = json.loads((GEMINI / "gemini-manifest.json").read_text())
                for key, array in [("base",base),("queries",queries),("truth",truth)]:
                    assert hashlib.sha256(array.tobytes()).hexdigest() == manifest["array_sha256"][key]
                assert (len(base), base.shape[1],len(queries)) == (99000,768,1000)
                for name_part in ["gemini-manifest.json","gemini-corpus-row-ids.npy","gemini-query-row-ids.npy"]:
                    import shutil
                    shutil.copy2(GEMINI/name_part, OUTPUT/name_part)
            else:
                base, queries, truth = prepare(DATA, name)
            inputs[name] = dict(base_shape=list(base.shape), query_shape=list(queries.shape),
                sha256={k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in [("base",base),("queries",queries),("truth",truth)]})
            print("Verified input",name,inputs[name],flush=True)
    write_json(OUTPUT/"inputs.json",inputs)
    execute(["cmake","-S",SOURCE,"-B",ROOT/"tests-build","-DTIEREDQ_EXAMPLES=OFF","-DCMAKE_BUILD_TYPE=Release","-DCMAKE_CXX_FLAGS=-march=native"])
    execute(["cmake","--build",ROOT/"tests-build","-j","8"])
    execute(["ctest","--test-dir",ROOT/"tests-build","--output-on-failure"])
    env = os.environ | {"PYTHONPATH":str(native.directory),"OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OMP_NUM_THREADS":"8"}
    # A small search for every dataset precedes full timed searches.
    for name in DATASETS:
        execute([sys.executable, "-c", "import codec_native, numpy as np; x=np.random.default_rng(0).normal(size=(1024,32)).astype('f'); i=codec_native.Index(x,2,0,True,False); o=np.empty((2,10),dtype='uint32'); i.search(x[:2],o,8); print('native smoke passed')"],env=env)
    started = time.perf_counter()
    with (OUTPUT/"fresh-results.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=FIELDS);writer.writeheader()
        cells=[(d,"tieredq",b,s) for d in DATASETS for b in BITS for s in [0,1,2]]
        cells += [("gemini","rabitq",b,s) for b in BITS for s in [1,2]]
        for number,(dataset,codec,bits,seed) in enumerate(cells,1):
            destination=OUTPUT/f"cell-{dataset}-{codec}-{bits}-{seed}.json"
            spec=dict(datasets=[dataset],codec=codec,bits=[bits],seeds=[seed],cell=str(destination))
            execute([sys.executable,__file__,json.dumps(spec)],env=env|{"BENCH_SEED":str(seed)})
            writer.writerows(json.loads(destination.read_text()));out.flush()
            print(f"PROGRESS {number}/{len(cells)}",flush=True)
    meta.update(status="matrix_complete",matrix_wall_s=time.perf_counter()-started)
    write_json(OUTPUT/"meta.json",meta)
    print("MATRIX COMPLETE",flush=True)

if __name__ == "__main__":
    main()
