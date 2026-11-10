"""Coarse router duel (#50): Faiss HNSW (flat and SQ8 with exact re-rank) against TieredQ's two-level router over the
Lloyd centroids at 32K/64K/128K lists (coverage of the true neighbours' lists, from the row labels, and routing recall
against the exact order) and over 1M centroids (1M random SQ8-decoded base rows; routing recall only). The tieredq
Python package is built from the source snapshot. 750 held-out queries, routed one at a time on one pinned core; builds
on 32 threads. Runs under the shared benchmark lock; resumable per size (exit.json)."""
import datetime, fcntl, json, subprocess
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
DUEL = EXPERIMENTS / 'TieredQ-issue44-routerduel-20261110/results'
STAGE_A = EXPERIMENTS / 'TieredQ-issue44-finelists-20261110/results'
ROWS, DIM, THREADS, SEARCH_CPU = 8840823, 1024, 32, 2
SIZES = [('32k', DUEL / '32k/centroids.f32', EXPERIMENTS / 'TieredQ-issue44-emeans-20261110/results/lloyd-32k/labels.u32'),
         ('64k', DUEL / '64k/centroids.f32', STAGE_A / 'lloyd-64k/labels.u32'),
         ('128k', DUEL / '128k/centroids.f32', STAGE_A / 'lloyd-128k/labels.u32'),
         ('1m', ROOT / 'results/1m/centroids.f32', None)]
MILLION = 1 << 20
UV = ['uv', 'run', '--no-project', '--python', '3.12', '--with', 'faiss-cpu==1.15.1', '--with', 'numpy',
      '--with', str(ROOT / 'source/tieredq')]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def million_centroids(path):
    """1M distinct random base rows, SQ8-decoded, as stand-in centroids for the large-list case."""
    if path.exists():
        return
    script = (f"import numpy as np; rng=np.random.default_rng(42); ids=np.sort(rng.choice({ROWS},{MILLION},replace=False)); "
              f"c=np.memmap('{QWEN / 'base.i8'}',dtype=np.int8,mode='r',shape=({ROWS},{DIM})); "
              f"s=np.fromfile('{QWEN / 'scales.f32'}',dtype=np.float32); o=np.fromfile('{QWEN / 'offsets.f32'}',dtype=np.float32); "
              f"(c[ids].astype(np.float32)*s+o).astype('<f4').tofile('{path}')")
    subprocess.run(['uv', 'run', '--no-project', '--python', '3.12', '--with', 'numpy', 'python', '-c', script], check=True)


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, centroids, labels in SIZES:
            out = ROOT / 'results' / name; out.mkdir(parents=True, exist_ok=True)
            if (out / 'exit.json').exists():
                continue
            status('running', size=name)
            if name == '1m':
                million_centroids(centroids)
            args = [*UV, 'python', str(ROOT / 'source/hnsw_duel.py'), '--centroids', str(centroids),
                    '--queries', str(QWEN / 'data/test_queries.fvecs'), '--truth', str(QWEN / 'data/test_truth.ivecs'),
                    '--nq', '750', '--dim', str(DIM), '--threads', str(THREADS), '--search-cpu', str(SEARCH_CPU),
                    '--output', str(out / 'duel.csv'), *(['--labels', str(labels)] if labels else [])]
            with (out / 'run.log').open('w') as log:
                code = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT).returncode
            (out / 'exit.json').write_text(json.dumps(dict(exit_code=code)) + '\n')
            if code:
                raise RuntimeError(f'{name} failed: {code}')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
