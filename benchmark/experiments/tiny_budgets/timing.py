"""Interleave seed-0 end-to-end timings; optionally collect separate diagnostic phases."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from common import (BATCH, BUDGETS, CACHE_NAME, THREADS, TOP_K, append_json,
                    digest, index_path, output_arrays, read_transform, write_json)

WARMUPS = 3
REPEATS = 3
SINGLE_QUERIES = 128
TIMING_CASES = ((1, 1), (1, THREADS), (BATCH, THREADS))
PROFILE_BATCHES = (1, BATCH)
PROFILE_PHASES = ('query_preparation_wall_s', 'scan_worker_s', 'selection_worker_s', 'merge_worker_s')


def recall(truth, ids):
    return float(np.mean([len(set(a) & set(b)) / TOP_K for a,b in zip(truth, ids, strict=True)]))


def run_calls(index, transform, queries, scores, ids, batch):
    durations = []
    for first in range(0, len(queries), batch):
        tick = time.perf_counter()
        probes = transform.queries(queries[first:first+batch])
        index.search(probes, scores[first:first+batch], ids[first:first+batch])
        durations.append(time.perf_counter() - tick)
    return durations


def profiles(native, indices, transform, queries, root):
    native.threads(1)
    for (budget, fast), index in indices.items():
        for batch in PROFILE_BATCHES:
            probes = transform.queries(queries[:batch])
            scores, ids = output_arrays(batch)
            index.search(probes, scores, ids)
            reference = ids.copy()
            native.profile_reset()
            start = time.perf_counter()
            index.search(probes, scores, ids)
            wall_s = time.perf_counter() - start
            phases = native.profile_read()
            np.testing.assert_array_equal(reference, ids)
            if len(phases) != len(PROFILE_PHASES) or not all(v >= 0 for v in phases) or phases[1] == 0:
                raise ValueError('invalid profiling counters')
            append_json(root / 'phases.jsonl', dict(budget_total_bits=budget, fast=fast,
                        batch=batch, threads=1, wall_s=wall_s, **dict(zip(PROFILE_PHASES, phases, strict=True))))
            print(f'PROFILE total_bits={budget} fast={fast} batch={batch} wall_ms={wall_s*1000:.3f}', flush=True)


def timings(native, indices, transform, queries, truth, root):
    cases = [(key, batch, threads) for key in indices for batch, threads in TIMING_CASES]
    for key, batch, threads in cases:
        native.threads(threads)
        scores, ids = output_arrays(batch)
        for _ in range(WARMUPS):
            run_calls(indices[key], transform, queries[:batch], scores, ids, batch)
    rng = np.random.default_rng(42)
    for repeat in range(REPEATS):
        for position in rng.permutation(len(cases)):
            (budget, fast), batch, threads = cases[position]
            native.threads(threads)
            count = SINGLE_QUERIES if batch == 1 else len(queries)
            scores, ids = output_arrays(count)
            durations = run_calls(indices[budget, fast], transform, queries[:count], scores, ids, batch)
            row = dict(budget_total_bits=budget, fast=fast, batch=batch, threads=threads,
                       repeat=repeat, queries=count, call_durations_s=durations,
                       total_s=sum(durations), qps=count/sum(durations),
                       p50_ms=float(np.percentile(durations, 50)*1000),
                       p95_ms=float(np.percentile(durations, 95)*1000), recall10=recall(truth[:count], ids))
            append_json(root / 'timing.jsonl', row)
            print(f'TIME repeat={repeat} total_bits={budget} fast={fast} batch={batch} threads={threads} qps={row["qps"]:.1f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    target = args.results / ('phases.jsonl' if args.profile else 'timing.jsonl')
    if target.exists():
        parser.error('use a results directory without the target measurement file')
    sys.path.insert(0, str(args.native.resolve()))
    import tiny_native as native
    with np.load(args.data / CACHE_NAME) as cache:
        queries, truth = cache['queries'], cache['truth']
    transform = read_transform(args.results / 'adapter.npz')
    with threadpool_limits(limits=1, user_api='blas'):
        indices = {(budget, fast): native.Index.load(str(index_path(args.results, budget, 0, fast)))
                   for budget in BUDGETS for fast in (False, True)}
        meta = dict(affinity=sorted(os.sched_getaffinity(0)), host=os.uname().nodename,
                    warmups=WARMUPS, repeats=REPEATS, single_queries=SINGLE_QUERIES,
                    include_query_transform=not args.profile, native_module_sha256=digest(Path(native.__file__)),
                    sources={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
        write_json(args.results / ('profile-manifest.json' if args.profile else 'timing-manifest.json'), meta)
        if args.profile:
            profiles(native, indices, transform, queries, args.results)
        else:
            timings(native, indices, transform, queries, truth, args.results)
    (args.results / ('profile-exit-code' if args.profile else 'timing-exit-code')).write_text('0\n')


if __name__ == '__main__':
    main()
