"""Faiss IVF-SQ8 baselines of the #52 scorecard with full held-out curves and 32-thread batch QPS (#51), on Qwen/MS
MARCO 8.84M x 1024 (shared SQ8 codes). HNSW and flat (exact) coarse quantizers at 64K and 128K lists, each trained by
Faiss itself on 40 rows per list with 4 Lloyd iterations (build time end to end). Every budget is replayed on the 750
held-out queries one query at a time on one pinned core, then as one 32-thread batch call. Same budgets as the
finelists study. Runs under the shared benchmark lock; resumable per step (exit.json)."""
import datetime, fcntl, json, subprocess
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
CALIBRATION_QUERIES, HELDOUT_QUERIES = 250, 750
THREADS, SEARCH_CPU = 32, 2
ROWS_PER_CENTROID = 40
BUDGETS = {65536: [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072],
           131072: [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144]}
STEPS = [('ivf-hnsw', 65536), ('ivf-hnsw', 131072), ('ivf', 65536), ('ivf', 131072)]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def faiss(out, variant, nlist):
    if (out / 'exit.json').exists():
        return
    args = ['uv', 'run', '--no-project', '--python', '3.12', '--with', 'faiss-cpu==1.15.1', '--with', 'numpy',
            str(ROOT / 'source/compare_faiss.py'),
            '--base', str(QWEN / 'data/base.rawview'), '--queries', str(QWEN / 'data/calibration_queries.fvecs'),
            '--truth', str(QWEN / 'data/calibration_truth.ivecs'), '--nq', str(CALIBRATION_QUERIES), '--variant', variant,
            '--nlist', str(nlist), '--sample', str(ROWS_PER_CENTROID * nlist), '--budgets', ','.join(map(str, BUDGETS[nlist])),
            '--int8-base', str(QWEN / 'base.i8'), '--int8-scales', str(QWEN / 'scales.f32'), '--int8-offsets', str(QWEN / 'offsets.f32'),
            '--evaluation-queries', str(QWEN / 'data/test_queries.fvecs'), '--evaluation-truth', str(QWEN / 'data/test_truth.ivecs'),
            '--evaluation-nq', str(HELDOUT_QUERIES), '--evaluation-output', str(out / 'heldout.csv'),
            '--targets', '0.95,0.99,0.995', '--build-threads', str(THREADS), '--search-cpu', str(SEARCH_CPU),
            '--heldout-all-budgets', '--batch-threads', str(THREADS), '--memory-output', str(out / 'memory.json')]
    status('running', step=out.name)
    with (out / 'run.log').open('w') as log, (out / 'calibration.csv').open('w') as stdout:
        code = subprocess.run(args, stdout=stdout, stderr=log).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code)) + '\n')
    if code:
        raise RuntimeError(f'{out.name} failed: {code}')


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (ROOT / 'results').mkdir(exist_ok=True)
        for kind, k in STEPS:
            out = ROOT / 'results' / f'{kind}-{k // 1024}k'; out.mkdir(exist_ok=True)
            faiss(out, kind, k)
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
