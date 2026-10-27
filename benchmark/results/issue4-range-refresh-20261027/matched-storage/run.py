"""Measure one OpenAI TieredQ fast point selected by serving storage, never recall."""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'source'
OUTPUT = ROOT / 'matched-storage'
sys.path[:0] = [str(OUTPUT / 'build'), str(SOURCE / 'benchmark')]
import numpy as np
from threadpoolctl import threadpool_limits
import codec_native
from data import prepare
from measure import measure, recall
from build_policy import audit_compile_commands

CAP_BYTES = 404.07024
TARGET_FRACTION = 0.995
DIMENSIONS = 1536
SEEDS = (0, 1, 2)
THREADS = (1, 8)
BATCHES = (1, 32)
DATA = Path('/home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze/benchmark/data')


def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + '\n')


def build(base, budget, seed):
    codec_native.set_threads(8)
    start = time.perf_counter()
    index = codec_native.Index(base, 1, seed, True, False, True, budget)
    elapsed = time.perf_counter() - start
    storage = index.storage(str(OUTPUT / 'selection-index.bin')) / len(base)
    return index, elapsed, storage


def main():
    assert sorted(os.sched_getaffinity(0)) == list(range(8))
    reference = list(csv.DictReader((ROOT / 'output/results.csv').open()))
    baseline = next(r for r in reference if (r['dataset'],r['codec'],r['target_bits'],r['seed'],r['threads'],r['batch']) == ('openai','rabitq','2','0','8','32'))
    cap = float(baseline['effective_bytes_per_vector'])
    assert abs(cap - CAP_BYTES) < 0.001
    meta = json.loads((ROOT / 'output/meta.json').read_text())
    for filename, expected in meta['source_sha256'].items():
        assert hashlib.sha256((SOURCE / filename).read_bytes()).hexdigest() == expected, filename
    commands = audit_compile_commands(json.loads((OUTPUT / 'build/compile_commands.json').read_text()))
    write_json('meta.json', dict(status='running', purpose='OpenAI fast point below RaBitQ 2-bit serving storage',
        cap_bytes=cap, target_fraction=TARGET_FRACTION, source_commit=meta['tieredq_commit'],
        source_hashes_match=True, affinity=sorted(os.sched_getaffinity(0)),
        compiler_commands=commands, host=platform.node(), date=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
    with threadpool_limits(limits=1):
        base, queries, truth = prepare(DATA, 'openai', None, 1000)
        hashes = {k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in [('base',base),('queries',queries),('truth',truth)]}
        assert hashes == json.loads((ROOT / 'output/inputs.json').read_text())['openai']['sha256']
        print('Verified source, compiler policy, CPU affinity and input hashes', flush=True)
        control, _, control_storage = build(base, DIMENSIONS, 0)
        ids = np.empty((len(queries),10), dtype=np.uint32)
        for start in range(0,len(queries),32): control.search(queries[start:start+32], ids[start:start+32],8)
        expected_ids = np.load(ROOT / 'output/cell-openai-tieredq-1-0-t8-b32-ids.npy')
        assert np.array_equal(ids,expected_ids)
        expected_storage = next(float(r['effective_bytes_per_vector']) for r in reference if (r['dataset'],r['codec'],r['target_bits'],r['seed'],r['threads'],r['batch']) == ('openai','tieredq','1','0','8','32'))
        assert control_storage == expected_storage
        write_json('control.json',dict(code_bits=DIMENSIONS, identical_ids=True, bytes_per_vector=control_storage, recall10=recall(truth,ids)))
        del control
        lower, upper = DIMENSIONS, 2*DIMENSIONS
        budget = 2259
        selection=[]
        for attempt in range(12):
            index, build_s, storage = build(base,budget,0)
            selection.append(dict(code_bits=budget,bytes_per_vector=storage))
            print('Storage selection',selection[-1],flush=True)
            if cap*0.99 <= storage <= cap*TARGET_FRACTION: break
            if storage > cap*TARGET_FRACTION: upper=budget
            else: lower=budget
            del index
            budget=(lower+upper)//2
        else: raise RuntimeError('No storage match within selection bounds')
        write_json('selection.json',dict(policy='Storage only; target 99.0–99.5% of baseline cap; initial code budget 2259; integer bisection within 1536–3072', attempts=selection,selected_code_bits=budget))
        rows=[]
        for seed in SEEDS:
            if seed: index,build_s,storage=build(base,budget,seed)
            common=dict(dataset='openai',d=DIMENSIONS,base_n=len(base),queries=len(queries),metric='cosine',codec='tieredq',config='fast-native-exact-code-budget',target_bits=budget/DIMENSIONS,effective_bits_per_dim=storage*8/DIMENSIONS,effective_bytes_per_vector=storage,build_s=build_s,encode_vps=len(base)/index.encode_seconds,rerank=False,seed=seed,code_bits=budget)
            write_json(f'diagnostics-seed{seed}.json',index.diagnostics())
            if seed:
                for start in range(0,len(queries),32): index.search(queries[start:start+32],ids[start:start+32],8)
                recall_only=dict(recall10=recall(truth,ids),qps=float('nan'),p50_ms=float('nan'),p95_ms=float('nan'))
            for threads in THREADS:
                for batch in BATCHES:
                    metrics=recall_only if seed else measure(index,queries,truth,threads,batch)
                    row=common | metrics | dict(threads=threads,batch=batch)
                    rows.append(row)
                    print('Measured',json.dumps(row),flush=True)
            del index
        with (OUTPUT/'results.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        metadata=json.loads((OUTPUT/'meta.json').read_text())
        metadata.update(status='complete',input_sha256=hashes,selected_code_bits=budget,control_ids_match=True,rows=len(rows),timed_rows=sum(r['seed']==0 for r in rows))
        write_json('meta.json',metadata)
        print('Completed supplemental point',flush=True)


if __name__ == '__main__':
    main()
