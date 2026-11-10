"""Finer partitions (32K/64K leaves) on Qwen 8.84M x 1024 with multi-threaded k-means and single-core search.
Gate: SIFT1M built with 1 and BUILD_THREADS threads must give identical search measurements."""
import csv, datetime, fcntl, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
SIFT = EXPERIMENTS / 'TieredQ-issue44-sift-sq8-20261107'
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
CALIBRATION_QUERIES = 250
BUILD_THREADS = 32
SEARCH_CPU = 2
QWEN_CAPS = [35000, 88408, 176816, 442041]
# Same 32 samples per fine centroid as the 64x128 baseline (4096 samples / 128 children).
PARTITIONS = [('qwen-32k', 128, 256, 8192), ('qwen-64k', 256, 256, 8192)]
GRAPH_BUDGETS = [2048, 4096, 8192, 16384]
# Columns that depend only on the partition and routing, not on timing.
DETERMINISTIC = ['routing', 'metadata_cap', 'vector_cap', 'recall10', 'mean_vectors', 'mean_metadata', 'mean_leaves', 'leaves', 'graph_build_distances']


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


def write_settings(path, leaves, coarse, caps):
    with path.open('x') as f:
        writer = csv.writer(f); writer.writerow(['routing', 'bounds', 'weight', 'metadata_cap', 'vector_cap', 'scan_order'])
        for cap in caps:
            writer.writerow(['flat', 'none', 0, leaves + coarse, cap, 'centroid'])
            for budget in GRAPH_BUDGETS:
                writer.writerow(['graph', 'none', 0, budget, cap, 'centroid'])


def graph_bench(out, data_root, coarse, children, sample, threads, settings, extra=()):
    out.mkdir(parents=True)
    args = [str(ROOT / 'build/graph_bench'), *data_args(data_root), '--variant', 'V0', '--coarse', str(coarse),
            '--children', str(children), '--leaves', str(coarse * children), '--iterations', '4', '--sample', str(sample),
            '--graph-degree', '8', '--settings', str(settings), '--build-threads', str(threads),
            '--search-cpu', str(SEARCH_CPU), *extra]
    status('running', run=str(out.relative_to(ROOT)), command=args)
    with (out / 'calibration.csv').open('x') as stdout, (out / 'run.log').open('x') as stderr:
        code = subprocess.run(args, stdout=stdout, stderr=stderr).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code)) + '\n')
    if code:
        raise RuntimeError(f'{out.name} failed: {code}')
    return [{k: r[k] for k in DETERMINISTIC} for r in csv.DictReader((out / 'calibration.csv').open())]


def threading_gate():
    settings = ROOT / 'settings' / 'sift-gate.csv'
    write_settings(settings, 8192, 64, [35000])
    serial = graph_bench(ROOT / 'gate' / 'sift-1thread', SIFT, 64, 128, 4096, 1, settings)
    threaded = graph_bench(ROOT / 'gate' / f'sift-{BUILD_THREADS}threads', SIFT, 64, 128, 4096, BUILD_THREADS, settings)
    if serial != threaded:
        raise RuntimeError('multi-threaded build changed search results')
    (ROOT / 'gate' / 'identical.json').write_text(json.dumps(dict(identical=True, rows=len(serial)), indent=2) + '\n')


def partition(name, coarse, children, sample):
    settings = ROOT / 'settings' / f'{name}.csv'
    write_settings(settings, coarse * children, coarse, QWEN_CAPS)
    out = ROOT / 'results' / name
    graph_bench(out, QWEN, coarse, children, sample, BUILD_THREADS, settings, [
        '--evaluation-queries', str(QWEN / 'data/test_queries.fvecs'), '--evaluation-truth', str(QWEN / 'data/test_truth.ivecs'),
        '--evaluation-nq', '750', '--evaluation-output', str(out / 'heldout.csv'), '--repeats', '3'])


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status('building_and_validating')
        checked(['cmake', '-S', str(ROOT / 'source'), '-B', str(ROOT / 'build'), '-DCMAKE_BUILD_TYPE=Release'], ROOT / 'configure.log')
        checked(['cmake', '--build', str(ROOT / 'build'), '-j', '16'], ROOT / 'build.log')
        checked(['ctest', '--test-dir', str(ROOT / 'build'), '--output-on-failure'], ROOT / 'ctest.log')
        for folder in ['results', 'settings', 'gate']:
            (ROOT / folder).mkdir(exist_ok=False)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'source').iterdir()) if p.is_file()}
        (ROOT / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
        threading_gate()
        for spec in PARTITIONS:
            partition(*spec)
        checked([sys.executable, str(ROOT / 'summarize_partitions.py'), str(ROOT)], ROOT / 'summary.log')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
