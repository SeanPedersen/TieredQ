"""Calibrate a serving-aware candidate selector using base-only distortion and measured costs."""
import argparse
import json
import random
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tiny_budgets'))
from common import append_json,load_cache,output_arrays,read_transform,write_json

CALIBRATION_ROWS=2048
COST_QUERIES=32
COST_REPEATS=3
parser=argparse.ArgumentParser(description=__doc__)
for name in ('data','native','previous','results'):
    parser.add_argument('--'+name,type=Path,required=True)
parser.add_argument('--output-name',default='costs.jsonl')
args=parser.parse_args()
sys.path.insert(0,str(args.native.resolve()))
import tiny_native as native
native.threads(8)
with threadpool_limits(limits=1,user_api='blas'):
    original,_,_=load_cache(args.data)
    transform=read_transform(args.previous/'adapter.npz')
    train=np.random.default_rng(0).choice(len(original),32768,replace=False)
    choices=np.setdiff1d(np.arange(len(original)),train)
    selected=np.random.default_rng(1409).choice(choices,CALIBRATION_ROWS,replace=False)
    base=transform.base_vectors(original[selected])
    probes=transform.queries(original[selected[:COST_QUERIES]])
    candidates=[]
    lines=(args.results/'allocation-quality.jsonl').read_text().splitlines()
    random.Random(14).shuffle(lines)
    for line in lines:
        row=json.loads(line)
        if row['seed']!=0:
            continue
        stem=f"{row['policy']}-b{row['budget']}-s0"
        raw=list((args.results/f'{stem}.bin').read_bytes())
        codec=native.Codec.load(raw)
        restored=np.empty_like(base)
        codec.reconstruct(base,restored)
        mse=float(np.mean((base.astype(np.float64)-restored)**2))
        index=native.Index.load(str(args.results/f'{stem}.tq'))
        scores,ids=output_arrays(1)
        durations=[]
        for repeat in range(COST_REPEATS+1):
            times=[]
            for i in range(COST_QUERIES):
                start=time.perf_counter_ns()
                index.search(probes[i:i+1],scores,ids)
                times.append((time.perf_counter_ns()-start)*1e-6)
            if repeat:
                durations.append(float(np.median(times)))
        cost=dict(policy=row['policy'],budget=row['budget'],bytes_per_vector=row['bytes_per_vector'],
                  base_reconstruction_mse=mse,base_single8_p50_ms=float(np.median(durations)),cost_repeat_p50_ms=durations)
        candidates.append(cost)
        append_json(args.results/args.output_name,cost)
        print(cost,flush=True)
    selected_cases=[]
    for baseline in candidates:
        if baseline['policy']!='marginal':
            continue
        feasible=[c for c in candidates if c['bytes_per_vector']<=baseline['bytes_per_vector']
                  and c['base_single8_p50_ms']<=baseline['base_single8_p50_ms']]
        chosen=min(feasible,key=lambda c:c['base_reconstruction_mse'])
        selected_cases.append(dict(baseline=baseline,selected=chosen))
    constrained=[]
    for memory in [c['bytes_per_vector'] for c in candidates if c['policy']=='marginal']:
        for latency in (0.75,1.0,1.5,2.0,3.0):
            feasible=[c for c in candidates if c['bytes_per_vector']<=memory and c['base_single8_p50_ms']<=latency]
            current=[c for c in feasible if c['policy']=='marginal']
            if not current:
                continue
            constrained.append(dict(memory_ceiling=memory,latency_ceiling_ms=latency,
                               current=min(current,key=lambda c:c['base_reconstruction_mse']),
                               selected=min(feasible,key=lambda c:c['base_reconstruction_mse'])))
    write_json(args.results/'serving-aware-selection.json',dict(calibration_rows=CALIBRATION_ROWS,
               base_query_proxy_rows=COST_QUERIES,calibration_base_ids=selected.tolist(),
               excludes_codec_training_rows=True,heldout_queries_or_truth_used=False,
               objective='minimize base reconstruction MSE subject to measured actual memory and base-query latency ceilings',
               selections=selected_cases, constrained_selections=constrained))
