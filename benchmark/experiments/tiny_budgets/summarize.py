"""Validate the tiny-budget study and aggregate recall, storage, timing, and phases."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (BASE_ROWS, BUDGETS, QUERY_COUNT, RECALL_R, SEEDS, TOP_K,
                    parse_model, read_rows, write_json)


def quality_summary(root):
    rows = read_rows(root / 'quality.jsonl')
    expected = {(b, s, fast) for b in BUDGETS for s in SEEDS for fast in (False, True)}
    actual = {(r['budget_total_bits'], r['seed'], r['fast']) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError('quality matrix incomplete or duplicated')
    reference_books = None
    for budget, seed in {(b, s) for b in BUDGETS for s in SEEDS}:
        raw = (root / f'b{budget}-s{seed}.bin').read_bytes()
        model = parse_model(raw)
        if model['budget'] != budget or sum(int(t[1])*int(t[2]) for t in model['tiers']) != budget:
            raise ValueError('wrong total-bit allocation')
        if reference_books is None:
            reference_books = model['levels']
        for a,b in zip(model['levels'], reference_books, strict=True):
            np.testing.assert_array_equal(a,b)
    groups = defaultdict(list)
    for row in rows:
        budget, seed, fast = row['budget_total_bits'], row['seed'], row['fast']
        label = 'fast' if fast else 'fp32'
        with np.load(root / f'b{budget}-s{seed}-{label}-quality.npz') as data:
            ranks, per_query = data['nearest_neighbor_rank'], data['recall10_per_query']
            if len(ranks) != QUERY_COUNT or data['top10_ids'].shape != (QUERY_COUNT, TOP_K):
                raise ValueError('query output shape mismatch')
            if not np.isclose(row['recall10'], per_query.mean()):
                raise ValueError('Recall@10 mismatch')
            for r in RECALL_R:
                if row['recall_at_r'][str(r)] != float(np.mean(ranks <= r)):
                    raise ValueError('nearest-neighbor recall curve mismatch')
        measured = (row['vector_storage_bytes']+row['model_memory_bytes']+row['adapter_bytes'])/BASE_ROWS
        if not np.isclose(measured, row['serving_bytes_per_vector']):
            raise ValueError('storage accounting mismatch')
        groups[budget, fast].append(row)
    summary = []
    for (budget, fast), values in sorted(groups.items()):
        summary.append(dict(budget_total_bits=budget, fast=fast,
                           recall10_mean=float(np.mean([r['recall10'] for r in values])),
                           recall10_min=min(r['recall10'] for r in values), recall10_max=max(r['recall10'] for r in values),
                           recall_at_r_mean={str(n):float(np.mean([r['recall_at_r'][str(n)] for r in values])) for n in RECALL_R},
                           nominal_code_bytes=budget/8,
                           serving_bytes_per_vector_mean=float(np.mean([r['serving_bytes_per_vector'] for r in values])),
                           vector_stride=sorted(set(r['bytes_per_vector_stride'] for r in values)),
                           retained_dimensions=sorted(set(r['coded_dimensions'] for r in values)),
                           representations=sorted(set(r['representation'] for r in values)),
                           non_stride_serving_bytes_mean=float(np.mean([r['vector_storage_bytes']-r['bytes_per_vector_stride']*BASE_ROWS+r['model_memory_bytes']+r['adapter_bytes'] for r in values])),
                           train_s_mean=float(np.mean([r['train_s'] for r in values])),
                           add_s_mean=float(np.mean([r['add_s'] for r in values]))))
    return summary


def timing_summary(root):
    path = root / 'timing.jsonl'
    if not path.exists():
        return []
    rows = read_rows(path)
    expected = {(b,f,batch,threads,repeat) for b in BUDGETS for f in (False,True)
                for batch,threads in ((1,1),(1,8),(32,8)) for repeat in range(3)}
    actual = {(r['budget_total_bits'],r['fast'],r['batch'],r['threads'],r['repeat']) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError('timing matrix incomplete or duplicated')
    groups = defaultdict(list)
    for row in rows:
        groups[row['budget_total_bits'],row['fast'],row['batch'],row['threads']].append(row)
    summary = []
    for (budget,fast,batch,threads), values in sorted(groups.items()):
        durations=np.concatenate([r['call_durations_s'] for r in values])
        summary.append(dict(budget_total_bits=budget,fast=fast,batch=batch,threads=threads,
                            qps=sum(r['queries'] for r in values)/sum(r['total_s'] for r in values),
                            qps_min=min(r['qps'] for r in values),qps_max=max(r['qps'] for r in values),
                            p50_ms=float(np.percentile(durations,50)*1000),
                            p95_ms=float(np.percentile(durations,95)*1000),
                            recall10_by_repeat=[r['recall10'] for r in values]))
    return summary


def plot(summary, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for fast,marker,label in ((False,'o','FP32 scoring'),(True,'s','Fast scoring')):
        rows=[r for r in summary if r['fast']==fast]
        axes[0].plot([r['serving_bytes_per_vector_mean'] for r in rows],
                     [r['recall10_mean'] for r in rows], marker=marker,label=label)
        for r in rows:
            axes[0].annotate(str(r['budget_total_bits']),
                             (r['serving_bytes_per_vector_mean'],r['recall10_mean']),
                             xytext=(4,-12 if fast else 6),textcoords='offset points',fontsize=8)
    axes[0].set(xlabel='Actual serving bytes/vector (including shared state)',ylabel='Recall@10',
                title='SIFT1M: 64–512 total code bits',ylim=(0,1),xlim=(10,80))
    axes[0].legend()
    for row in summary:
        if not row['fast']:continue
        axes[1].plot(RECALL_R,[row['recall_at_r_mean'][str(r)] for r in RECALL_R],
                     marker='.',label=f"{row['budget_total_bits']} bits")
    axes[1].set(xscale='log',xlabel='Retrieved candidates R',ylabel='True nearest-neighbor Recall@R',
                title='Fast scoring: candidate retrieval',ylim=(0,1.02))
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    figure.savefig(root/'recall-storage.png',dpi=180)
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    quality=quality_summary(args.results)
    timing=timing_summary(args.results)
    phases=read_rows(args.results/'phases.jsonl') if (args.results/'phases.jsonl').exists() else []
    if phases:
        expected={(b,f,n) for b in BUDGETS for f in (False,True) for n in (1,32)}
        if len(phases)!=len(expected) or {(r['budget_total_bits'],r['fast'],r['batch']) for r in phases}!=expected:
            raise ValueError('profiling matrix incomplete or duplicated')
    write_json(args.results/'summary.json',dict(quality=quality,timing=timing,phases=phases))
    plot(quality,args.results)
    print('bits fast recall10 nn@100 bytes/vector retained')
    for row in quality:
        print(f"{row['budget_total_bits']:3} {int(row['fast'])} {row['recall10_mean']:.5f} "
              f"{row['recall_at_r_mean']['100']:.5f} {row['serving_bytes_per_vector_mean']:.3f} {row['retained_dimensions']}")
    print('Validated quality matrix; timing cases:',len(timing),'phase rows:',len(phases))


if __name__=='__main__':
    main()
