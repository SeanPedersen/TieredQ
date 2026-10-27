"""Evaluate a storage-selected point using the archived recall and timing procedure."""
import csv
import math
from pathlib import Path

from runtime import build, search_ids, write_json
from measure import measure, recall

SEEDS = (0, 1, 2)
THREADS = (1, 8)
BATCHES = (1, 32)


def write_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(dataset, base, queries, truth, anchor_bits, selected, directory):
    rows = []
    built = selected.built
    dimensions = base.shape[1]
    for seed in SEEDS:
        if seed:
            built = build(base, selected.code_bits, seed, directory)
        common = dict(dataset=dataset, d=dimensions, base_n=len(base), queries=len(queries),
                      metric='cosine', codec='tieredq', config='fast-native-exact-code-budget',
                      target_bits=selected.code_bits / dimensions, code_bits=selected.code_bits,
                      anchor_bits=anchor_bits, effective_bits_per_dim=built.bytes_per_vector * 8 / dimensions,
                      effective_bytes_per_vector=built.bytes_per_vector, build_s=built.build_s,
                      encode_vps=built.encode_vps, rerank=False, seed=seed)
        write_json(directory / f'diagnostics-seed{seed}.json', built.index.diagnostics())
        if seed:
            ids = search_ids(built.index, queries)
            recall_only = dict(recall10=recall(truth, ids), qps=math.nan, p50_ms=math.nan, p95_ms=math.nan)
        for threads in THREADS:
            for batch in BATCHES:
                metrics = recall_only if seed else measure(built.index, queries, truth, threads, batch)
                row = common | metrics | dict(threads=threads, batch=batch)
                rows.append(row)
                print(f'MEASURED {dataset} cap={anchor_bits} seed={seed} threads={threads} batch={batch} '
                      f'recall={row["recall10"]:.6f} qps={row["qps"]:.2f}', flush=True)
        if not seed:
            selected.built = None
        del built
    write_rows(directory / 'results.csv', rows)
    return rows
