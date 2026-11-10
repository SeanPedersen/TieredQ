"""Offline scan-policy simulation on Qwen 8.84M x 1024: boundary spill by bisector margin and patience stopping.
Runs on the best flat 32K partitions (Lloyd, E-means 4 epochs) from the E-means study. Runs under the shared lock."""
import datetime, fcntl, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
QUERIES, THREADS = 250, 32
CAPS = [35000, 50000, 70000, 88408, 120000, 176816, 250000, 350000, 442041, 600000]
EMEANS = EXPERIMENTS / 'TieredQ-issue44-emeans-20261110' / 'results'
MODELS = [('qwen-32k-emeans4', EMEANS / 'emeans-32k-s40-e4' / 'labels.u32'), ('qwen-32k-lloyd', EMEANS / 'lloyd-32k' / 'labels.u32')]
PATIENCE = [4, 8, 16, 32, 64, 128, 256]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def checked(args, log):
    with log.open('w') as out:
        subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, check=True)


def simulate(name, labels):
    out = ROOT / 'results' / name; out.mkdir(parents=True)
    args = [str(ROOT / 'build/cell_ceiling'), '--base', str(QWEN / 'data/base.rawview'),
            '--queries', str(QWEN / 'data/calibration_queries.fvecs'), '--truth', str(QWEN / 'data/calibration_truth.ivecs'),
            '--nq', str(QUERIES), '--variant', 'V0', '--coarse', '128', '--children', '256', '--leaves', '32768', '--iterations', '4',
            '--build-threads', str(THREADS), '--assignments', str(labels), '--patience', ','.join(map(str, PATIENCE)),
            '--cell-queries', str(QUERIES), '--threads', str(THREADS), '--slabs', '0', '--hull', '0', '--statistical', '0',
            '--policies', '1', '--coverage-budgets', ','.join(map(str, CAPS)), '--cell-output', str(out / 'cells.csv')]
    for flag, file in [('base', 'base.i8'), ('scales', 'scales.f32'), ('offsets', 'offsets.f32')]:
        args += ['--int8-' + flag, str(QWEN / file)]
    status('running', model=name, command=args)
    with (out / 'run.log').open('x') as log:
        code = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code)) + '\n')
    if code:
        raise RuntimeError(f'{name} failed: {code}')


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status('building_and_validating')
        checked(['cmake', '-S', str(ROOT / 'source'), '-B', str(ROOT / 'build'), '-DCMAKE_BUILD_TYPE=Release'], ROOT / 'configure.log')
        checked(['cmake', '--build', str(ROOT / 'build'), '-j', '16'], ROOT / 'build.log')
        checked(['ctest', '--test-dir', str(ROOT / 'build'), '--output-on-failure'], ROOT / 'ctest.log')
        (ROOT / 'results').mkdir(exist_ok=False)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'source').iterdir()) if p.is_file()}
        (ROOT / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
        for name, labels in MODELS:
            simulate(name, labels)
        checked([sys.executable, str(ROOT / 'summarize_policies.py'), str(ROOT)], ROOT / 'summary.log')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
