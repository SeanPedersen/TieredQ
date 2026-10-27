"""Measure the seed-0 control with only the automatic checkpoint dispatch bypassed."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from run_refresh import ROOT, DATASETS, BITS, execute, write_json

BASELINE = ROOT/"source-no-range"
OUTPUT = ROOT/"ablation"
OUTPUT.mkdir(exist_ok=True)
path = BASELINE/"src/tieredq_index.cpp"
text=path.read_text()
old="if (rows==1 && backend==tieredq_detail::FastScanBackend::avx512 &&"
new="if (false && rows==1 && backend==tieredq_detail::FastScanBackend::avx512 &&"
assert text.count(old)==1
path.write_text(text.replace(old,new))
write_json(OUTPUT/"variant.json",dict(file="src/tieredq_index.cpp",before=old,after=new))
python=ROOT/"source/benchmark/.venv/bin/python"
build=BASELINE/"benchmark/build-native"
execute(["cmake","-S",BASELINE/"benchmark/native","-B",build,
    "-DRABITQ_SOURCE="+str(ROOT/"source/benchmark/.deps/rabitq-native"),
    "-DPython_EXECUTABLE="+str(python),"-DCMAKE_BUILD_TYPE=Release"])
execute(["cmake","--build",build,"-j","8"])
sys.path.insert(0,str(ROOT/"source/benchmark"))
from run import FIELDS
from build_policy import audit_compile_commands
write_json(OUTPUT/"compiler_commands.json",audit_compile_commands(json.loads((build/"compile_commands.json").read_text())))
env=os.environ|dict(PYTHONPATH=str(build),OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="8",
    REFRESH_SOURCE="source-no-range",REFRESH_OUTPUT="ablation",BENCH_SEED="0")
started=time.perf_counter()
with (OUTPUT/"results.csv").open("w",newline="") as handle:
    writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader()
    for number,(dataset,bits) in enumerate([(d,b) for d in DATASETS for b in BITS],1):
        destination=OUTPUT/f"cell-{dataset}-tieredq-{bits}-0.json"
        spec=dict(datasets=[dataset],codec="tieredq",bits=[bits],seeds=[0],cell=str(destination))
        execute([python,ROOT/"run_refresh.py",json.dumps(spec)],env=env)
        writer.writerows(json.loads(destination.read_text()));handle.flush()
        print(f"ABLATION {number}/12",flush=True)
parity=[]
for dataset in DATASETS:
    for bits in BITS:
        for threads in [1,8]:
            for batch in [1,32]:
                name=f"cell-{dataset}-tieredq-{bits}-0-t{threads}-b{batch}-ids.npy"
                a=np.load(ROOT/"output"/name);b=np.load(OUTPUT/name)
                identical=bool(np.array_equal(a,b))
                parity.append(dict(dataset=dataset,bits=bits,threads=threads,batch=batch,identical_ids=identical,
                    differing_queries=int(np.count_nonzero(np.any(a!=b,axis=1)))))
write_json(OUTPUT/"parity.json",parity)
assert all(p["identical_ids"] for p in parity), "bounds-on/off ranking mismatch"
write_json(OUTPUT/"meta.json",dict(status="complete",wall_s=time.perf_counter()-started,paired_cells=len(parity)))
print("ABLATION COMPLETE; ALL 48 ID COMPARISONS IDENTICAL",flush=True)

execute([python, ROOT/"finalize.py"])
