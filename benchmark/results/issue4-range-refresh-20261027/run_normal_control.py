"""Small matched-query control of TieredQ normal and fast representations on full corpora."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"normal-control"
SOURCE=ROOT/"source"
QUERY_COUNT=100
THREADS=8
BITS=[1,2,4]
DATASETS=["sift","glove","openai","gemini"]

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2)+"\n")

def run_cell(dataset,bits):
    import numpy as np
    from threadpoolctl import threadpool_limits
    sys.path.insert(0,str(SOURCE/"benchmark"))
    from data import prepare
    from gemini_data import digest
    from l2_transform import L2Transform
    from measure import measure,L2Search
    from run_refresh import DATA,GEMINI
    import codec_native
    with threadpool_limits(limits=1):
        if dataset=="gemini":
            with np.load(GEMINI/"gemini-prepared.npz") as arrays:
                base,queries,truth=arrays["base"],arrays["queries"],arrays["truth"]
        else:
            base,queries,truth=prepare(DATA,dataset)
        queries=np.ascontiguousarray(queries[:QUERY_COUNT]);truth=truth[:QUERY_COUNT]
        transform=L2Transform.fit(base) if dataset=="sift" else None
        vectors=transform.base_vectors(base) if transform else base
        rows=[];diags={};predictions={}
        for fast in [True,False]:
            mode="fast" if fast else "normal"
            codec_native.set_threads(THREADS)
            started=time.perf_counter()
            index=codec_native.Index(vectors,bits,0,True,dataset=="sift",fast)
            build_s=time.perf_counter()-started
            diags[mode]=index.diagnostics()
            with tempfile.TemporaryDirectory(dir=OUT) as temporary:
                size=index.storage(str(Path(temporary)/"index.bin"))
            size+=transform.storage_bytes if transform else 0
            search=L2Search(index,transform.scale,transform) if transform else index
            for batch in [1,32]:
                metrics=measure(search,queries,truth,THREADS,batch)
                ids=np.empty((len(queries),10),dtype=np.uint32)
                for offset in range(0,len(queries),batch):
                    search.search(queries[offset:offset+batch],ids[offset:offset+batch],THREADS)
                predictions[(mode,batch)]=ids
                np.save(OUT/f"{dataset}-{bits}-{mode}-b{batch}-ids.npy",ids)
                row=dict(dataset=dataset,bits=bits,mode=mode,seed=0,threads=THREADS,batch=batch,
                    base_n=len(base),queries=len(queries),effective_bytes_per_vector=size/len(base),
                    build_s=build_s,encode_vps=len(base)/index.encode_seconds,**metrics)
                rows.append(row);print(row,flush=True)
            del search,index
        assert diags["fast"]["tiers"]==diags["normal"]["tiers"]
        write_json(OUT/f"{dataset}-{bits}-diagnostics.json",diags)
        write_json(OUT/f"{dataset}-{bits}-parity.json",[
            dict(batch=b,identical_ids=bool(np.array_equal(predictions[("fast",b)],predictions[("normal",b)])),
                differing_queries=int(np.count_nonzero(np.any(predictions[("fast",b)]!=predictions[("normal",b)],axis=1))))
            for b in [1,32]])
        write_json(OUT/f"{dataset}-{bits}-rows.json",rows)
        write_json(OUT/f"{dataset}-{bits}-inputs.json",dict(base_sha256=digest(base),queries_sha256=digest(queries),truth_sha256=digest(truth)))


def main():
    if len(sys.argv)>1:
        run_cell(sys.argv[1],int(sys.argv[2]));return
    OUT.mkdir(exist_ok=True)
    while not (ROOT/"ablation-exit-code").exists():
        time.sleep(10)
    assert (ROOT/"ablation-exit-code").read_text().strip()=="0", "prior run failed"
    started=time.perf_counter()
    env=os.environ|dict(PYTHONPATH=str(SOURCE/"benchmark/build-native"),OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="8",BENCH_SEED="0")
    allrows=[]
    for dataset in DATASETS:
        for bits in BITS:
            subprocess.run([sys.executable,"-u",__file__,dataset,str(bits)],env=env,check=True)
            allrows.extend(json.loads((OUT/f"{dataset}-{bits}-rows.json").read_text()))
            print(f"NORMAL CONTROL {len(allrows)//4}/12",flush=True)
    assert len(allrows)==48
    with (OUT/"results.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    pairs=[]
    for dataset in DATASETS:
        for bits in BITS:
            for batch in [1,32]:
                group=[r for r in allrows if (r["dataset"],r["bits"],r["batch"])==(dataset,bits,batch)]
                fast=next(r for r in group if r["mode"]=="fast");normal=next(r for r in group if r["mode"]=="normal")
                pairs.append(dict(dataset=dataset,bits=bits,batch=batch,fast_recall=fast["recall10"],normal_recall=normal["recall10"],
                    fast_bytes=fast["effective_bytes_per_vector"],normal_bytes=normal["effective_bytes_per_vector"],
                    fast_qps=fast["qps"],normal_qps=normal["qps"],fast_over_normal_qps=fast["qps"]/normal["qps"]))
    write_json(OUT/"comparison.json",pairs)
    write_json(OUT/"meta.json",dict(status="complete",wall_s=time.perf_counter()-started,query_count=QUERY_COUNT,
        threads=THREADS,seeds=[0],timing_repeats=3,warmups=3,full_corpora=True,source_revision=(SOURCE/"SOURCE_REVISION").read_text().strip(),
        note="Matched first 100 held-out queries; independent deterministic builds with matching tier allocations. Build time excludes shared SIFT transform preparation."))
    lines=["# Quick TieredQ normal-mode control","",
        "Full corpora, first 100 held-out queries, seed 0, 8 threads, batches 1/32, budgets 1/2/4. Both modes are freshly timed on the identical query subset, with three warmups and three repeats. This small control does not replace the main 1,000-query benchmark or its three-seed recall results.","",
        "Both builds use the same seed/training inputs and matching learned tiers. Normal mode uses the sixth native Index constructor argument `fast=False`; fast mode uses `fast=True`. Storage includes native serving state plus the SIFT metric-preserving adapter. Build times exclude the SIFT transform shared by both modes in this control.","",
        "| Dataset | Bits | Batch | Fast recall | Normal recall | Fast bytes | Normal bytes | Fast QPS | Normal QPS | Fast/normal QPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in pairs:
        lines.append(f"| {p['dataset']} | {p['bits']} | {p['batch']} | {p['fast_recall']:.4f} | {p['normal_recall']:.4f} | {p['fast_bytes']:.2f} | {p['normal_bytes']:.2f} | {p['fast_qps']:,.0f} | {p['normal_qps']:,.0f} | {p['fast_over_normal_qps']:.2f}× |")
    lines += ["","See results.csv for p50/p95 latency and build/encode metrics. Per-cell diagnostics, input hashes and top-10 IDs are retained; normal versus fast rankings are reported without requiring equality because their scoring representations can differ."]
    (OUT/"README.md").write_text("\n".join(lines)+"\n")
    print("NORMAL CONTROL COMPLETE",flush=True)

if __name__=="__main__":
    main()
