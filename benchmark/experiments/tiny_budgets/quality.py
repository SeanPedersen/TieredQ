"""Measure tiny-budget SIFT recall and canonical serving storage on full base data."""
import argparse
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from common import (BASE_ROWS, BATCH, BUDGETS, CACHE_NAME, CURVE_K, ORIGINAL_DIMENSIONS,
                    QUERY_COUNT, RECALL_R, SEEDS, THREADS, TOP_K, TRAINING_ROWS,
                    L2Transform, append_json, digest, index_path, load_cache,
                    output_arrays, parse_model, write_json)



def retrieve(index, probes, truth):
    scores, ids = output_arrays(len(probes), CURVE_K)
    for first in range(0, len(probes), BATCH):
        index.search(probes[first:first+BATCH], scores[first:first+BATCH], ids[first:first+BATCH])
    if not np.isfinite(scores).all() or np.any(scores[:, 1:] > scores[:, :-1]) or np.any(ids >= BASE_ROWS):
        raise ValueError('invalid ranked results')
    small_scores, small_ids = output_arrays(BATCH)
    index.search(probes[:BATCH], small_scores, small_ids)
    np.testing.assert_array_equal(small_ids, ids[:BATCH, :TOP_K])
    np.testing.assert_array_equal(small_scores, scores[:BATCH, :TOP_K])
    found = ids == truth[:, :1]
    nn_rank = np.where(found.any(axis=1), found.argmax(axis=1) + 1, CURVE_K + 1)
    recall_per_query = np.array([len(set(a) & set(b)) / TOP_K for a, b in zip(ids[:, :TOP_K], truth, strict=True)])
    return dict(recall10=float(recall_per_query.mean()),
                recall_at_r={str(r): float(np.mean(nn_rank <= r)) for r in RECALL_R}), ids[:, :TOP_K], nn_rank, recall_per_query


def representation(path, fast):
    with path.open('rb') as handle:
        header = handle.read(80)
        if not fast:
            return 'fp32-packed-planes'
        payload_offset, = struct.unpack_from('<Q', header, 32)
        handle.seek(payload_offset)
        version, kind, _ = struct.unpack('<3I', handle.read(12))
    if version != 1:
        raise ValueError('unexpected fast payload')
    return {1: 'narrow-byte-LUT', 2: 'integer-levels', 3: 'byte-code-IDs', 4: 'packed-code-IDs'}[kind]


def evaluate(native, raw, base, probes, truth, root, budget, seed, fast, adapter_bytes):
    index = native.Index(native.Codec.load(raw), fast)
    start = time.perf_counter()
    index.add(base)
    add_s = time.perf_counter() - start
    build_storage = dict(vector_storage_bytes=index.storage(), model_memory_bytes=index.model_memory())
    path = index_path(root, budget, seed, fast) if seed == SEEDS[0] else root / 'reload-scratch.tq'
    index.save(str(path))
    index = native.Index.load(str(path))
    metrics, ids, rank, per_query = retrieve(index, probes, truth)
    stem = f'b{budget}-s{seed}-{"fast" if fast else "fp32"}'
    np.savez_compressed(root / f'{stem}-quality.npz', top10_ids=ids, nearest_neighbor_rank=rank,
                        recall10_per_query=per_query)
    storage = index.storage()
    model = index.model_memory()
    return dict(budget_total_bits=budget, seed=seed, fast=fast, add_s=add_s,
                representation=representation(path, fast), scalar_fast=index.scalar_fast() if fast else False,
                bytes_per_vector_stride=index.stride(), vector_storage_bytes=storage,
                model_memory_bytes=model, adapter_bytes=adapter_bytes,
                serving_bytes_per_vector=(storage+model+adapter_bytes)/BASE_ROWS,
                serialized_index_bytes=path.stat().st_size, build_memory=build_storage, **metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.native.resolve()))
    import tiny_native as native
    native.threads(THREADS)
    original, queries, truth = load_cache(args.data)
    with threadpool_limits(limits=1, user_api='blas'):
        start = time.perf_counter()
        transform = L2Transform.fit(original)
        base = transform.base_vectors(original)
        probes = transform.queries(queries)
        transform_s = time.perf_counter() - start
        np.savez(args.output / 'adapter.npz', scale=transform.scale,
                 base_map=transform.base_map, query_map=transform.query_map)
        manifest = dict(dataset='SIFT1M', original_dimensions=ORIGINAL_DIMENSIONS,
                        indexed_dimensions=base.shape[1], base_rows=BASE_ROWS, query_count=QUERY_COUNT,
                        training_rows=TRAINING_ROWS, centroids=256, rotate_tiers=True,
                        seeds=SEEDS, budgets_total_bits=BUDGETS, recall_r=RECALL_R,
                        adapter='query-metric-L2-lift', adapter_fit_and_base_transform_s=transform_s,
                        threads=THREADS, affinity=sorted(os.sched_getaffinity(0)),
                        dataset_cache_sha256=digest(args.data / CACHE_NAME),
                        source_sha256={str(p): digest(p) for folder in ('src', 'include', 'cmake', 'benchmark/experiments')
                                       for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)},
                        python=sys.version, host=os.uname().nodename)
        write_json(args.output / 'manifest.json', manifest)
        for seed in SEEDS:
            selected = np.random.default_rng(seed).choice(len(base), TRAINING_ROWS, replace=False)
            sample = np.ascontiguousarray(base[selected])
            for budget in BUDGETS:
                print(f'TRAIN seed={seed} total_bits={budget}', flush=True)
                start = time.perf_counter()
                raw = native.train_total(sample, budget, True, seed)
                train_s = time.perf_counter() - start
                codec = native.Codec.load(raw)
                if codec.roundtrip() != raw:
                    raise ValueError('codec roundtrip mismatch')
                parsed = parse_model(raw)
                if parsed['budget'] != budget:
                    raise ValueError('total-bit budget mismatch')
                (args.output / f'b{budget}-s{seed}.bin').write_bytes(bytes(raw))
                for fast in (False, True):
                    print(f'ENCODE/RETRIEVE seed={seed} total_bits={budget} fast={fast}', flush=True)
                    row = evaluate(native, raw, base, probes, truth, args.output, budget, seed, fast, transform.storage_bytes)
                    row.update(train_s=train_s, coded_dimensions=len(parsed['sigma']), tiers=parsed['tiers'].tolist())
                    append_json(args.output / 'quality.jsonl', row)
                    print(json.dumps({k: row[k] for k in ('budget_total_bits','seed','fast','recall10','recall_at_r','serving_bytes_per_vector','representation')}), flush=True)
    (args.output / 'quality-exit-code').write_text('0\n')


if __name__ == '__main__':
    main()
