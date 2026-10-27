"""Controlled SIFT lookup and allocation study using frozen experimental native sources."""
import argparse
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tiny_budgets'))
from common import (append_json, digest, load_cache, output_arrays, parse_model,
                    read_transform, write_json)

BATCH = 32
QUALITY_QUERIES = 1000
TIMING_QUERIES = 128
WARMUPS = 3
REPEATS = 3
THREADS = 8
TRAINING_ROWS = 32768
BUDGETS = (128, 160, 192, 224, 256, 288, 384, 512, 1032)
PRIMARY = (128, 192, 256, 512, 1032)


def search(index, probes, k=10):
    scores, ids = output_arrays(len(probes), k)
    for first in range(0, len(probes), BATCH):
        index.search(probes[first:first+BATCH], scores[first:first+BATCH], ids[first:first+BATCH])
    return scores, ids


def recall(ids, truth):
    return np.array([len(set(a) & set(b))/10 for a,b in zip(ids, truth, strict=True)])


def metrics(index, adapter_bytes, n):
    return dict(stride=index.stride(), storage=index.storage(), model_memory=index.model_memory(),
                bytes_per_vector=(index.storage()+index.model_memory()+adapter_bytes)/n)


def timing(native, index, queries, transform, truth, batch, threads):
    native.threads(threads)
    def call(first, size):
        q=transform.queries(queries[first:first+size])
        scores,ids=output_arrays(len(q))
        index.search(q,scores,ids)
        return ids
    for _ in range(WARMUPS):
        call(0,batch)
    elapsed=[]
    found=[]
    count=TIMING_QUERIES if batch==1 else len(queries)
    for first in range(0,count,batch):
        start=time.perf_counter_ns()
        ids=call(first,batch)
        elapsed.append((time.perf_counter_ns()-start)*1e-6)
        found.extend(ids)
    return dict(batch=batch,threads=threads,p50_ms=float(np.median(elapsed)),
                p95_ms=float(np.percentile(elapsed,95)),qps=count/(sum(elapsed)*1e-3),
                recall10=float(recall(found,truth[:count]).mean()),call_ms=elapsed)


def lookup(args,native,base,probes,queries,truth,transform):
    rows=[]
    for budget in (64,80,128):
        index=native.Index.load(str(args.previous/f'b{budget}-s0-fast.tq'))
        # All document scores and full deterministic order, before speed measurements.
        full=[]
        for grouping in ('register','byte'):
            os.environ['TIEREDQ_EXPERIMENT_GROUPING']=grouping
            scores,ids=output_arrays(1,len(base))
            index.search(probes[:1],scores,ids)
            full.append((scores,ids))
        np.testing.assert_array_equal(full[0][0],full[1][0])
        np.testing.assert_array_equal(full[0][1],full[1][1])
        ranked=[]
        for grouping in ('register','byte'):
            os.environ['TIEREDQ_EXPERIMENT_GROUPING']=grouping
            scores,ids=search(index,probes)
            ranked.append((scores,ids))
            np.savez_compressed(args.output/f'lookup-b{budget}-{grouping}.npz',scores=scores,ids=ids)
            row=dict(budget=budget,grouping=grouping,all_document_scores_equal=True,
                     recall10=float(recall(ids,truth).mean()),**metrics(index,transform.storage_bytes,len(base)))
            append_json(args.output/'lookup-quality.jsonl',row)
        np.testing.assert_array_equal(ranked[0][0],ranked[1][0])
        np.testing.assert_array_equal(ranked[0][1],ranked[1][1])
        cases=list(itertools.product(range(REPEATS),('register','byte'),((1,1),(1,8),(32,8))))
        random.Random(budget).shuffle(cases)
        for repeat,grouping,(batch,threads) in cases:
            os.environ['TIEREDQ_EXPERIMENT_GROUPING']=grouping
            row=dict(budget=budget,grouping=grouping,repeat=repeat,
                     **timing(native,index,queries,transform,truth,batch,threads))
            append_json(args.output/'lookup-timing.jsonl',row)
            print('LOOKUP',budget,grouping,repeat,batch,threads,row['p50_ms'],flush=True)
    os.environ['TIEREDQ_EXPERIMENT_GROUPING']='register'


def allocation(args,native,base,probes,queries,truth,transform):
    for seed in (0,1,2):
        selected=np.random.default_rng(seed).choice(len(base),TRAINING_ROWS,replace=False)
        sample=np.ascontiguousarray(base[selected])
        for budget in (BUDGETS if seed==0 else PRIMARY):
            for policy in ('marginal','log','cap4'):
                if budget>4*base.shape[1] and policy=='cap4':
                    continue
                os.environ['TIEREDQ_EXPERIMENT_ALLOCATION']=policy
                raw=native.train_total(sample,budget,True,seed)
                parsed=parse_model(raw)
                stem=f'{policy}-b{budget}-s{seed}'
                (args.output/f'{stem}.bin').write_bytes(bytes(raw))
                reference=args.previous/f'b{budget}-s{seed}.bin'
                if policy=='marginal' and reference.exists():
                    assert bytes(raw)==reference.read_bytes(), 'default source differs from prior recipe'
                if budget==8*base.shape[1] and policy=='log':
                    assert bytes(raw)==(args.output/f'marginal-b{budget}-s{seed}.bin').read_bytes()
                codec=native.Codec.load(raw)
                assert codec.roundtrip()==raw
                index=native.Index(codec,native.Mode.fast)
                start=time.perf_counter()
                index.add(base)
                add_s=time.perf_counter()-start
                path=args.output/f'{stem}.tq' if seed==0 else args.output/'reload-scratch.tq'
                index.save(str(path))
                index=native.Index.load(str(path))
                scores,ids=search(index,probes)
                per_query=recall(ids,truth)
                np.savez_compressed(args.output/f'{stem}-quality.npz',ids=ids,scores=scores,recall10=per_query)
                row=dict(policy=policy,budget=budget,seed=seed,recall10=float(per_query.mean()),
                         tiers=parsed['tiers'].tolist(),coded_dimensions=len(parsed['sigma']),
                         add_s=add_s,serialized_bytes=path.stat().st_size,
                         **metrics(index,transform.storage_bytes,len(base)))
                append_json(args.output/'allocation-quality.jsonl',row)
                print('ALLOCATION',json.dumps(row),flush=True)
    (args.output/'quality-exit-code').write_text('0\n')


def allocation_timing(args,native,queries,truth,transform):
    cases=list(itertools.product(range(REPEATS),('marginal','log','cap4'),BUDGETS,((1,1),(1,8),(32,8))))
    random.Random(14).shuffle(cases)
    for repeat,policy,budget,(batch,threads) in cases:
        path=args.output/f'{policy}-b{budget}-s0.tq'
        if not path.exists():
            continue
        index=native.Index.load(str(path))
        row=dict(policy=policy,budget=budget,repeat=repeat,
                 **timing(native,index,queries,transform,truth,batch,threads))
        append_json(args.output/'allocation-timing.jsonl',row)
        print('TIMING',policy,budget,repeat,batch,threads,row['p50_ms'],flush=True)
    (args.output/'timing-exit-code').write_text('0\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','native','previous','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mode',choices=('lookup','allocation','timing'),required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=True)
    sys.path.insert(0,str(args.native.resolve()))
    import tiny_native as native
    native.threads(THREADS)
    with threadpool_limits(limits=1,user_api='blas'):
        original,queries,truth=load_cache(args.data)
        transform=read_transform(args.previous/'adapter.npz')
        probes=transform.queries(queries)
        base=transform.base_vectors(original) if args.mode=='allocation' else original
        write_json(args.output/f'{args.mode}-manifest.json',dict(mode=args.mode,python=sys.version,
                   affinity=sorted(os.sched_getaffinity(0)),dataset_sha256=digest(args.data/'sift-1000000-1000-v2.npz'),
                   queries=len(queries),training_rows=TRAINING_ROWS,source_sha256={str(p):digest(p)
                   for folder in ('src','include','benchmark/experiments') for p in Path(folder).rglob('*')
                   if p.is_file() and '__pycache__' not in str(p)}))
        if args.mode=='lookup':
            lookup(args,native,base,probes,queries,truth,transform)
        elif args.mode=='allocation':
            allocation(args,native,base,probes,queries,truth,transform)
        else:
            allocation_timing(args,native,queries,truth,transform)


if __name__=='__main__':
    main()
