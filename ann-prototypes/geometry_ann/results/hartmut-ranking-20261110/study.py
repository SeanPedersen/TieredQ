"""Ranking study: (A) offline statistical leaf scores on 8192 leaves; (B) finer partitions with real search.
Runs serially under the shared benchmark lock on hartmut."""
import csv, datetime, fcntl, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
SIFT = EXPERIMENTS / 'TieredQ-issue44-sift-sq8-20261107'
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
CALIBRATION_QUERIES = 250
THREADS = 32
QWEN_CAPS = [35000, 88408, 176816, 442041]
SIFT_CAPS = [5000, 10000, 20000, 35000]
# Same sample-per-centroid ratio as the 64x128 baseline (4096 samples / 128 children).
PARTITIONS = [('qwen-32k', 128, 256, 8192), ('qwen-64k', 256, 256, 8192)]
GRAPH_BUDGETS = [2048, 4096, 8192, 16384]
STATISTICAL = ROOT / 'statistical' / 'results'  # layout expected by summarize_cells.py


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def checked(args, log):
    with log.open('w') as out:
        subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, check=True)


def data_args(data_root):
    args = ['--base', str(data_root / 'data/base.rawview'),
            '--queries', str(data_root / 'data/calibration_queries.fvecs'),
            '--truth', str(data_root / 'data/calibration_truth.ivecs'), '--nq', str(CALIBRATION_QUERIES)]
    for flag, file in [('base', 'base.i8'), ('scales', 'scales.f32'), ('offsets', 'offsets.f32')]:
        args += ['--int8-' + flag, str(data_root / file)]
    return args


def execute(base, name, args, stdout_name='run.out'):
    out = base / name; out.mkdir(parents=True)
    status('running', model=name, command=args)
    with (out / stdout_name).open('x') as stdout, (out / 'run.log').open('x') as stderr:
        code = subprocess.run(args, stdout=stdout, stderr=stderr, cwd=out).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code)) + '\n')
    if code:
        raise RuntimeError(f'{name} failed: {code}')
    return out


def statistical(name, data_root, caps):
    out_csv = STATISTICAL / name / 'cells.csv'
    execute(STATISTICAL, name, [str(ROOT / 'build/cell_ceiling'), *data_args(data_root), '--variant', 'V0',
                   '--coarse', '64', '--children', '128', '--leaves', '8192', '--iterations', '4',
                   '--cell-queries', str(CALIBRATION_QUERIES), '--threads', str(THREADS),
                   '--slabs', '0', '--hull', '0', '--coverage-budgets', ','.join(map(str, caps)),
                   '--cell-output', str(out_csv)])


def partition(name, coarse, children, sample):
    leaves = coarse * children
    settings = ROOT / 'settings' / f'{name}.csv'
    with settings.open('x') as f:
        writer = csv.writer(f); writer.writerow(['routing', 'bounds', 'weight', 'metadata_cap', 'vector_cap', 'scan_order'])
        for cap in QWEN_CAPS:
            writer.writerow(['flat', 'none', 0, leaves + coarse, cap, 'centroid'])
            for budget in GRAPH_BUDGETS:
                writer.writerow(['graph', 'none', 0, budget, cap, 'centroid'])
    out = ROOT / 'results' / name
    execute(ROOT / 'results', name, ['taskset', '-c', '2', str(ROOT / 'build/graph_bench'), *data_args(QWEN), '--variant', 'V0',
                   '--coarse', str(coarse), '--children', str(children), '--leaves', str(leaves), '--iterations', '4',
                   '--sample', str(sample), '--graph-degree', '8', '--settings', str(settings),
                   '--evaluation-queries', str(QWEN / 'data/test_queries.fvecs'),
                   '--evaluation-truth', str(QWEN / 'data/test_truth.ivecs'), '--evaluation-nq', '750',
                   '--evaluation-output', str(out / 'heldout.csv'), '--repeats', '3'], stdout_name='calibration.csv')


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status('building_and_validating')
        checked(['cmake', '-S', str(ROOT / 'source'), '-B', str(ROOT / 'build'), '-DCMAKE_BUILD_TYPE=Release'], ROOT / 'configure.log')
        checked(['cmake', '--build', str(ROOT / 'build'), '-j', '16'], ROOT / 'build.log')
        checked(['ctest', '--test-dir', str(ROOT / 'build'), '--output-on-failure'], ROOT / 'ctest.log')
        (ROOT / 'results').mkdir(exist_ok=False); (ROOT / 'settings').mkdir(exist_ok=False)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'source').iterdir()) if p.is_file()}
        (ROOT / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
        statistical('sift-statistical', SIFT, SIFT_CAPS)
        statistical('qwen-statistical', QWEN, QWEN_CAPS)
        checked([sys.executable, str(ROOT / 'summarize_cells.py'), str(ROOT / 'statistical')], ROOT / 'summary-statistical.log')
        for spec in PARTITIONS:
            partition(*spec)
        checked([sys.executable, str(ROOT / 'summarize_partitions.py'), str(ROOT)], ROOT / 'summary-partitions.log')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
