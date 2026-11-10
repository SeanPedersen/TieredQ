"""Cell-ceiling study: how many leaves can convex (hyperplane) lower bounds rule out? Runs under the shared benchmark lock."""
import datetime, fcntl, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
DATASETS = {
    'sift-cells': EXPERIMENTS / 'TieredQ-issue44-sift-sq8-20261107',
    'qwen-cells': EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107',
}
QUERIES = 50
THREADS = 32
HULL_ITERATIONS = 128


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def checked(args, log):
    with log.open('w') as out:
        subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, check=True)


def run(name, data_root):
    out = ROOT / 'results' / name; out.mkdir()
    args = [str(ROOT / 'build/cell_ceiling'),
            '--base', str(data_root / 'data/base.rawview'),
            '--queries', str(data_root / 'data/calibration_queries.fvecs'),
            '--truth', str(data_root / 'data/calibration_truth.ivecs'),
            '--nq', str(QUERIES), '--variant', 'V0', '--coarse', '64', '--children', '128', '--leaves', '8192',
            '--iterations', '4', '--cell-queries', str(QUERIES), '--threads', str(THREADS),
            '--hull-iterations', str(HULL_ITERATIONS), '--cell-output', str(out / 'cells.csv')]
    for flag, file in [('base', 'base.i8'), ('scales', 'scales.f32'), ('offsets', 'offsets.f32')]:
        args += ['--int8-' + flag, str(data_root / file)]
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
        for name, data_root in DATASETS.items():
            run(name, data_root)
        checked([sys.executable, str(ROOT / 'summarize.py'), str(ROOT)], ROOT / 'summary.log')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
