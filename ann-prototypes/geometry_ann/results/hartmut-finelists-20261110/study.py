"""Experiment 3, stage A: flat Lloyd partitions at 64K and 128K lists, plus Faiss IVF-SQ8 baselines at the same list
counts with a flat (exact) and an HNSW coarse quantizer. Same recipe as the 32K runs: 4 Lloyd iterations, 40 training
rows per centroid, exact final assignment for our labels; Faiss trains on the same number of rows. Our index is measured
in stage B (separate study) once the router changes are final. Runs under the shared benchmark lock; every step is
resumable (partition.json / exit.json)."""
import datetime, fcntl, json, subprocess
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
ROWS, DIM = 8840823, 1024
CALIBRATION_QUERIES, HELDOUT_QUERIES = 250, 750
THREADS, SEARCH_CPU = 32, 2
ROWS_PER_CENTROID = 40
# Probe budgets keep the scanned fraction comparable to the 32K runs (nprobe 32..1536 at 32K).
BUDGETS = {65536: [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072],
           131072: [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144]}
STEPS = [('lloyd', 65536), ('ivf', 65536), ('ivf-hnsw', 65536), ('lloyd', 131072), ('ivf-hnsw', 131072), ('ivf', 131072)]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def run(out, args, csv_stdout):
    """Runs one step unless it already finished; Faiss writes its calibration table to stdout."""
    if (out / 'exit.json').exists():
        return
    status('running', step=out.name, command=args)
    with (out / 'run.log').open('w') as log, (out / ('calibration.csv' if csv_stdout else 'stdout.log')).open('w') as stdout:
        code = subprocess.run(args, stdout=stdout, stderr=log).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code)) + '\n')
    if code:
        raise RuntimeError(f'{out.name} failed: {code}')


def lloyd(out, k):
    run(out, ['uv', 'run', '--no-project', '--python', '3.12', str(ROOT / 'source/train_partitions.py'),
              '--codes', str(QWEN / 'base.i8'), '--scales', str(QWEN / 'scales.f32'), '--offsets', str(QWEN / 'offsets.f32'),
              '--rows', str(ROWS), '--dim', str(DIM), '--k', str(k), '--method', 'lloyd', '--iterations', '4',
              '--sample-per-centroid', str(ROWS_PER_CENTROID), '--threads', str(THREADS),
              '--labels-output', str(out / 'labels.u32'), '--report', str(out / 'partition.json')], csv_stdout=False)


def faiss(out, variant, nlist):
    run(out, ['uv', 'run', '--no-project', '--python', '3.12', '--with', 'faiss-cpu==1.15.1', '--with', 'numpy',
              str(ROOT / 'source/compare_faiss.py'),
              '--base', str(QWEN / 'data/base.rawview'), '--queries', str(QWEN / 'data/calibration_queries.fvecs'),
              '--truth', str(QWEN / 'data/calibration_truth.ivecs'), '--nq', str(CALIBRATION_QUERIES), '--variant', variant,
              '--nlist', str(nlist), '--sample', str(ROWS_PER_CENTROID * nlist), '--budgets', ','.join(map(str, BUDGETS[nlist])),
              '--int8-base', str(QWEN / 'base.i8'), '--int8-scales', str(QWEN / 'scales.f32'), '--int8-offsets', str(QWEN / 'offsets.f32'),
              '--evaluation-queries', str(QWEN / 'data/test_queries.fvecs'), '--evaluation-truth', str(QWEN / 'data/test_truth.ivecs'),
              '--evaluation-nq', str(HELDOUT_QUERIES), '--evaluation-output', str(out / 'heldout.csv'),
              '--targets', '0.95,0.99,0.995', '--build-threads', str(THREADS), '--search-cpu', str(SEARCH_CPU),
              '--memory-output', str(out / 'memory.json')], csv_stdout=True)


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (ROOT / 'results').mkdir(exist_ok=True)
        for kind, k in STEPS:
            out = ROOT / 'results' / f'{kind}-{k // 1024}k'; out.mkdir(exist_ok=True)
            if kind == 'lloyd':
                lloyd(out, k)
            else:
                faiss(out, kind, k)
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
