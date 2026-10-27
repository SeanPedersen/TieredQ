"""Audit and combine fresh TieredQ measurements with explicitly sourced RaBitQ baselines."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np
from run_refresh import ROOT, DATASETS, BITS, write_json
sys.path.insert(0,str(ROOT/"source/benchmark"))
from run import FIELDS
from report import report
import plotting
plotting.DATASET_NAMES["gemini"]="DBpedia Gemini-768"
OUT=ROOT/"output"
REUSE=ROOT/"reuse"

def read(path):
    return list(csv.DictReader(path.open()))

def key(row):
    return tuple(row[k] for k in ["dataset","codec","target_bits","seed","threads","batch"])

fresh=read(OUT/"fresh-results.csv")
rows=list(fresh)
origins={key(r):"fresh-results.csv" for r in rows}
for row in read(REUSE/"issue4-results.csv"):
    if row["codec"]=="rabitq" and row["target_bits"] in ["1","2","4"]:
        rows.append(row);origins[key(row)]="issue4-current-single-seed-20261026/results.csv"
for row in read(REUSE/"curve-results.csv"):
    if row["codec"]!="rabitq" or row["target_bits"] not in ["1","2","4"] or row["seed"] not in ["1","2"]:
        continue
    for threads,batch in itertools.product([1,8],[1,32]):
        copied=row|dict(threads=str(threads),batch=str(batch),qps="nan",p50_ms="nan",p95_ms="nan")
        rows.append(copied);origins[key(copied)]="issue12-recall/pack-opt/curve/results.csv (recall only)"
for row in read(REUSE/"gemini-results.csv"):
    converted=dict(dataset="gemini",d="768",base_n=row["base_n"],queries=row["queries"],metric="cosine",codec="rabitq",
        config="flat-fht-native-bounds-hacc",target_bits=row["anchor_bits"],
        effective_bits_per_dim=str(float(row["bytes_per_vector"])*8/768),effective_bytes_per_vector=row["bytes_per_vector"],
        build_s=row["build_s"],encode_vps=row["encode_vps"],recall10=row["recall10"],qps=row["qps"],
        p50_ms=row["p50_ms"],p95_ms=row["p95_ms"],threads=row["threads"],batch=row["batch"],rerank="False",seed=row["seed"])
    rows.append(converted);origins[key(converted)]="gemini-rabitq-single-seed-20261026/results.csv"
expected=set(itertools.product(DATASETS,["tieredq","rabitq"],["1","2","4"],["0","1","2"],["1","8"],["1","32"]))
assert len(rows)==len(expected)==288
assert {key(r) for r in rows}==expected
for row in rows:
    assert 0<=float(row["recall10"])<=1
    assert float(row["effective_bytes_per_vector"])>0
    for field in ["qps","p50_ms","p95_ms"]:
        assert math.isfinite(float(row[field])) if row["seed"]=="0" else math.isnan(float(row[field]))
inputs=json.loads((OUT/"inputs.json").read_text())
previous=json.loads((REUSE/"data-verification.json").read_text())
for dataset in ["sift","glove","openai"]:
    assert inputs[dataset]["sha256"]==previous[dataset]["array_sha256"]
gemini=json.loads((REUSE/"gemini-manifest.json").read_text())
for field in ["base","queries","truth"]:
    assert inputs["gemini"]["sha256"][field]==gemini["array_sha256"][field]
oldmeta=json.loads((REUSE/"issue4-meta.json").read_text())
curve_meta=json.loads((REUSE/"curve-meta.json").read_text())
from setup_native import REVISION
assert REVISION == oldmeta["rabitq_commit"]
assert json.loads((ROOT/"source/benchmark/.deps/rabitq-native/patches.json").read_text()) == oldmeta["rabitq_patches"]
for field in ["rabitq_commit","rabitq_patches","rabitq_search"]:
    assert oldmeta[field]==curve_meta[field]
for path in ["benchmark/native/module.cpp","benchmark/native/rabitq_flat.h"]:
    assert hashlib.sha256((ROOT/"source"/path).read_bytes()).hexdigest()==oldmeta["source_sha256"][path]
assert hashlib.sha256((ROOT/"source/benchmark/measure.py").read_bytes()).hexdigest()==oldmeta["harness_sha256"]["measure.py"]
with (OUT/"results.csv").open("w",newline="") as handle:
    writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
write_json(OUT/"row-provenance.json",[{"key":list(k),"source":v} for k,v in origins.items()])
meta=json.loads((OUT/"meta.json").read_text())
meta.update(status="complete",tieredq_commit=(ROOT/"source/SOURCE_REVISION").read_text().strip(),
    cpu=(ROOT/"host-cpu.txt").read_text(),compiler=(ROOT/"compiler.txt").read_text(),rabitq_commit=oldmeta["rabitq_commit"],
    reused_timing_date="2026-10-26",fresh_timing_date="2026-10-27",source_note="Native files match saved adapter hashes; build is fresh.")
write_json(OUT/"meta.json",meta)
report(OUT)
parity=json.loads((ROOT/"ablation/parity.json").read_text())
assert len(parity)==48 and all(p["identical_ids"] for p in parity)
control={key(r):r for r in read(ROOT/"ablation/results.csv")}
comparisons=[]
for row in fresh:
    if row["codec"]!="tieredq" or row["seed"]!="0":continue
    other=control[key(row)]
    assert row["recall10"]==other["recall10"]
    assert row["effective_bytes_per_vector"]==other["effective_bytes_per_vector"]
    diagnostics=json.loads((OUT/f"cell-{row['dataset']}-tieredq-{row['target_bits']}-0-diagnostics.json").read_text())
    tiers=diagnostics["tiers"]
    eligible=len(tiers)>4 and max(t["bits"] for t in tiers)>4
    comparisons.append(dict(dataset=row["dataset"],bits=int(row["target_bits"]),threads=int(row["threads"]),batch=int(row["batch"]),
        eligible_layout=eligible,checkpoint_active=eligible and row["batch"]=="1",tier_count=len(tiers),
        bounds_qps=float(row["qps"]),control_qps=float(other["qps"]),qps_ratio=float(row["qps"])/float(other["qps"])))
write_json(OUT/"ablation-summary.json",comparisons)
write_json(OUT/"audit.json",dict(status="passed",rows=288,timed_rows=96,recall_seeds=[0,1,2],
    input_hashes_match=True,reused_adapter_and_measurement_hashes_match=True,ablation_identical_ids=48))
lines=["# Issue #4: fast-mode range-bound refresh", "", 
    "Four datasets; average code budgets 1/2/4 bits/dimension; 1/8 threads; batches 1/32; 1,000 queries; top-10. Recall uses seeds 0/1/2; timing uses seed 0 with three warmups and three repeats. Tables show seed-0 storage and timing, with population standard deviation across the three recall seeds. No reranking.","",
    "Fresh TieredQ and Gemini RaBitQ recall seeds 1/2 were run on Hartmut i9-7980XE, affinity CPUs 0–7. Saved RaBitQ seed-0 timings are from 2026-10-26; the first three datasets also reuse their saved recall seeds 1/2. Dataset array/truth hashes, native adapter and measurement hashes were checked. Between-run timing variation remains; the baseline reuse is explicit in row-provenance.json.","",
    "Gemini is 99,000 base / 1,000 disjoint held-out document queries, normalized cosine, 768D. Other datasets: SIFT 1M × 128 (L2), GloVe 100K × 200, OpenAI 100K × 1536 (normalized cosine).","",
    "## Primary serving point: 8 threads, batch 32","",
    "| Dataset | Codec | Bits | Recall@10 mean ± std | Bytes/vector | QPS |","|---|---|---:|---:|---:|---:|"]
for dataset in DATASETS:
    for codec in ["tieredq","rabitq"]:
        for bits in BITS:
            group=[r for r in rows if r["dataset"]==dataset and r["codec"]==codec and r["target_bits"]==str(bits) and r["threads"]=="8" and r["batch"]=="32"]
            timed=next(r for r in group if r["seed"]=="0")
            recalls=[float(r["recall10"]) for r in group]
            lines.append(f"| {dataset} | {codec} | {bits} | {np.mean(recalls):.4f} ± {np.std(recalls):.4f} | {float(timed['effective_bytes_per_vector']):.3f} | {float(timed['qps']):,.0f} |")
lines += ["","## Range-bound control","",
    "The control build changes only the automatic checkpoint dispatch condition in src/tieredq_index.cpp. All 48 paired scheduling configurations produced identical top-10 IDs, recall and storage. Native range-checkpoint tests additionally cover surviving-score bit parity. The checkpoint dispatch is AVX-512, single-query only and requires a supported code-lookup layout with more than four tiers; batch-32 points are controls for timing variation, not bound speedup measurements.","",
    "| Dataset | Bits | Threads | Batch | Checkpoint active | Bounds/control QPS |","|---|---:|---:|---:|---|---:|"]
for c in comparisons:
    lines.append(f"| {c['dataset']} | {c['bits']} | {c['threads']} | {c['batch']} | {c['checkpoint_active']} | {c['qps_ratio']:.2f}× |")
lines += ["","The measured grid cannot establish an optimum between budgets. Target recall 0.95 is reported only where a measured point reaches it; effective storage, rather than nominal bits alone, determines within-budget comparisons. See results.csv, summary.json, audit.json and ablation-summary.json.","",
    "![Recall and storage](recall-storage.png)","![Throughput](qps-recall.png)","![Latency](latency-recall.png)","",
    "Reproduction scripts and source.tar.gz are in the parent directory; remote-workspace.txt records the isolated remote job directory. Native tests and adapter validation logs are preserved in job.log and ablation.log. Compiler commands, source hashes, source revision and affinity are recorded in meta.json."]
(OUT/"README.md").write_text("\n".join(lines)+"\n")
print("FINAL AUDIT PASSED: 288 rows, 96 timed cells, 48 bounds/control ID parity checks")
