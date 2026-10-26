"""Generate concise measured tables from the audited core matrix."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    rows=list(csv.DictReader((ROOT/'results.csv').open()))
    primary=[r for r in rows if r['threads']=='8' and r['batch']=='32']
    meta=json.loads((ROOT/'meta.json').read_text())
    audit=json.loads((ROOT/'audit.json').read_text())
    summary=json.loads((ROOT/'summary.json').read_text())
    differences=json.loads((ROOT/'historical-differences.json').read_text())
    lines=['# Issue #4 refresh: current fast mode, seed 0', '',
        'Full SIFT1M (1M × 128), GloVe (100K × 200) and OpenAI3 (100K × 1536), 1,000 held-out queries each; budgets 1/2/4/8, threads 1/8, batches 1/32. TieredQ fast mode only versus pinned native direct-flat RaBitQ with bounds and HACC enabled. Three warmup queries and three complete timing repeats. This is one single-seed run; it establishes no statistical confidence.', '',
        f'Core matrix wall time: **{meta["matrix_wall_s"]:.2f} seconds ({meta["matrix_wall_s"]/60:.2f} minutes)**. Includes index construction and the whole scheduling matrix; excludes compilation, dataset verification/preparation and figure generation. No profiling or optimization runs were added. The sole follow-up diagnostic investigates the observed SIFT batch-dependent recall.', '',
        '## Current measurements', '',
        'QPS is the primary 8-thread/batch-32 point. Construction includes training, encoding/packing and SIFT adapter fitting/transformation. Encode throughput measures native add/encode/pack only, excluding training and adapter fit. Bytes/vector are freshly measured serving representations, including shared model/cache state and the SIFT adapter; these are not process RSS.', '',
        '| Dataset | Codec | Bits | Recall@10 | Bytes/vector | QPS | Build s | Encode vectors/s |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in primary:
        lines.append(f'| {r["dataset"]} | {r["codec"]} | {r["target_bits"]} | {float(r["recall10"]):.4f} | {float(r["effective_bytes_per_vector"]):.3f} | {float(r["qps"]):,.0f} | {float(r["build_s"]):.3f} | {float(r["encode_vps"]):,.0f} |')
    lines += ['', '## Single-query response latency', '', 'Batch 1, milliseconds per complete query, including preprocessing, scan, selection and binding overhead. TieredQ can parallelize document scanning for one query; native RaBitQ parallelizes across queries, so batch 1 has no query-level speedup.', '', '| Dataset | Codec | Bits | 1 thread p50 / p95 ms | 8 threads p50 / p95 ms |', '|---|---|---:|---:|---:|']
    for r in primary:
        pair=[next(x for x in rows if all(x[k]==r[k] for k in ('dataset','codec','target_bits')) and x['threads']==t and x['batch']=='1') for t in ('1','8')]
        cells=[f'{float(x["p50_ms"]):.3f} / {float(x["p95_ms"]):.3f}' for x in pair]
        lines.append(f'| {r["dataset"]} | {r["codec"]} | {r["target_bits"]} | {cells[0]} | {cells[1]} |')
    lines += ['', '## Discrete within-memory comparisons', '', 'At each current RaBitQ memory anchor, select the highest-recall measured TieredQ point that fits. QPS is 8 threads/batch 32. A dash means no measured TieredQ point fits. These are discrete budget-constrained comparisons, with no interpolation or exact-equal-memory claim.', '', '| Dataset | RaBitQ bits / bytes / recall / QPS | TieredQ bits / bytes / recall / QPS within memory |', '|---|---|---|']
    def point(p):
        return '—' if p is None else f'{p["bits"]} / {p["bytes_per_vector"]:.3f} / {p["recall_mean"]:.4f} / {p["qps_seed0"]:,.0f}'
    for c in summary['comparisons']:
        if c['threads']==8: lines.append(f'| {c["dataset"]} | {point(c["rabitq_anchor"])} | {point(c["tieredq_best_recall_within_budget"])} |')
    lines += ['', 'Minimum measured bytes/vector reaching Recall@10 ≥ 0.95 (not exact equal-recall optima):', '', '| Dataset | TieredQ bytes (bits) | RaBitQ bytes (bits) |', '|---|---:|---:|']
    for dataset in ('sift','glove','openai'):
        cells=[]
        for codec in ('tieredq','rabitq'):
            p=next(p for p in summary['operating_points'] if p['dataset']==dataset and p['codec']==codec and p['threads']==8)['minimum_measured_at_095']
            cells.append('unreached' if p is None else f'{p["bytes_per_vector"]:.3f} ({p["bits"]})')
        lines.append(f'| {dataset} | {cells[0]} | {cells[1]} |')
    lines += ['', '## Changes from historical issue #4', '', 'The original [2026-10-23 results](../hartmut-full-single-run-20261023/README.md) remain unchanged. Dataset manifests and base/query hashes match exactly. Current measurements combine the high-bit FP32/code-ID precision fix, threshold-5 routing, packed payloads, the query-sensitive metric-preserving SIFT adapter, and corrected serving-memory accounting. Historical memory totals are comparison evidence only; none were reused for current points. Timing ratios below compare separate runs and do not isolate an individual change.', '', '| Dataset | Codec | Bits | Recall delta | Bytes old → current | QPS current / old |', '|---|---|---:|---:|---:|---:|']
    for d in differences:
        if d['threads']=='8' and d['batch']=='32': lines.append(f'| {d["dataset"]} | {d["codec"]} | {d["target_bits"]} | {d["recall10_delta"]:+.4f} | {d["bytes_per_vector_old"]:.3f} → {d["bytes_per_vector_current"]:.3f} | {d["qps_current_over_old"]:.2f}× |')
    lines += ['', '## Audit and provenance', '',
        '- `audit.py` / `audit.json`: all 96 unique rows, complete Cartesian grid, finite positive timings, correct dimensions/counts/metrics, and identical storage/construction/encoding across scheduling rows. Recall is identical across thread counts within each batch.',
        '- SIFT TieredQ 1-bit recall is 0.5005 at batch 1 and 0.5006 at batch 32. The targeted diagnostic isolates FP32 adapter matrix-multiplication rounding: transformed coordinates differ by at most 1.1444e-5; one query changes its top-10 set. Holding transformed probes fixed gives identical IDs at native batches 1/32. All other recall points agree across scheduling rows. Raw values remain unchanged; primary comparisons use batch 32, while the storage plot follows batch 1. See `batch-recall-diagnostic.json`.',
        '- `validation.log`: all 16 tests passed, including native-reference rankings and flat storage. All five effective compiler commands in `meta.json` pass `-O3 -march=native -ffp-contract=off`; source uses nanobind `NOMINSIZE`. Pinned RaBitQ revision is `ced3763857e7e4e48553951bf66fabc8517bc4f0`, with only the recorded deterministic rotation overlay; bounds/HACC remain enabled.',
        '- Intel i9-7980XE, AVX-512BW, GCC 16.2.1; affinity CPUs 0–7. BLAS is held at one thread inside timed workers; construction uses eight OpenMP workers. Unrelated services were active; this run does not establish exclusive-host timing.',
        '- `data-verification.json`: full cache array hashes, truth hashes, manifests and split-index hashes. SIFT/GloVe raw checksums verified before timing; pinned OpenAI dataset revision and original manifest preserved. Split arrays are in `split-indices/`.',
        '- `committed-source.tar` exports clean local HEAD `3c9dbf590209dabf8ecccd3385fd56254779c70d`. `source-sha256.json` and `source-verification.json` identify the as-run source; `scope-only.patch` removes automatic post-matrix profiling only. `SOURCE_REVISION` identifies the export; there is no remote Git checkout. `provenance.json` records archive hash, branch and reconciliation. Fresh native build and isolated uv environment; old workspaces/builds were not modified.',
        '- All three generated figures were rendered and visually inspected; panels separate thread counts and label every measured budget. Lines connect discrete points and do not imply interpolated equal-memory results.',
        '- `job.log` preserves the initial wrapper import-path failure after successful validation; `resume.log`, `data-verification.log`, `matrix.log` and `exit-code` document recovery and successful completion. No matrix timings preceded the correction. Generic `recall_std=0` fields in `summary.json` reflect one seed and are not uncertainty estimates.', '',
        'Remote workspace: `sean@hartmut:/home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze`. Reproduce from the archive, apply `scope-only.patch` to omit automatic profiling, reuse verified caches, then from `benchmark/`:', '', '```sh', './validate.sh', 'taskset -c 0-7 ./run.sh --bits 1 2 4 8 --seeds 0 --threads 1 8 --batches 1 32 --output output-current-<new-timestamp>', '```', '',
        '![Recall versus measured storage](recall-storage.png)', '', '![Batch throughput versus recall](qps-recall.png)', '', '![Single-query latency versus recall](latency-recall.png)', '']
    (ROOT/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
