"""Flat partitions trained by Faiss Lloyd vs E-means, measured with the same index/search as our hierarchy.
Partitions: train_partitions.py (32 threads, BLAS). Index/search: graph_bench --assignments (search single-core).
Runs under the shared benchmark lock; completed models (exit.json) are skipped on resume."""
import csv, datetime, fcntl, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
ROWS, DIM = 8840823, 1024
CALIBRATION_QUERIES, HELDOUT_QUERIES = 250, 750
THREADS, SEARCH_CPU = 32, 2
VECTOR_CAPS = [35000, 50000, 70000, 88408, 120000, 176816, 250000, 350000, 442041, 600000]
# name, k, coarse parents for the graph, method arguments, graph metadata budget. Decisive 32K runs first.
MODELS = [
    ('lloyd-32k', 32768, 128, ['--method', 'lloyd', '--iterations', '4', '--sample-per-centroid', '40'], 8192),
    # Fair comparison: identical 40-per-centroid training rows and final assignment pass. 4 epochs equals
    # Lloyd's 4 iterations of compute; 1 and 2 test the claim of matching quality with fewer passes.
    ('emeans-32k-s40-e1', 32768, 128, ['--method', 'emeans', '--epochs', '1', '--sample-per-centroid', '40'], 8192),
    ('emeans-32k-s40-e2', 32768, 128, ['--method', 'emeans', '--epochs', '2', '--sample-per-centroid', '40'], 8192),
    ('emeans-32k-s40-e4', 32768, 128, ['--method', 'emeans', '--epochs', '4', '--sample-per-centroid', '40'], 8192),
    ('lloyd-8k', 8192, 64, ['--method', 'lloyd', '--iterations', '4', '--sample-per-centroid', '40'], 4096),
    ('emeans-8k-s40-e2', 8192, 64, ['--method', 'emeans', '--epochs', '2', '--sample-per-centroid', '40'], 4096),
]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def checked(args, log):
    with log.open('w') as out:
        subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, check=True)


def train(out, k, method_args):
    labels, report = out / 'labels.u32', out / 'partition.json'
    if report.exists():
        return labels
    status('training', run=out.name)
    checked(['uv', 'run', '--no-project', '--python', '3.12', str(ROOT / 'source/train_partitions.py'),
             '--codes', str(QWEN / 'base.i8'), '--scales', str(QWEN / 'scales.f32'), '--offsets', str(QWEN / 'offsets.f32'),
             '--rows', str(ROWS), '--dim', str(DIM), '--k', str(k), *method_args, '--emeans-path', str(ROOT / 'vendor'),
             '--threads', str(THREADS), '--labels-output', str(labels), '--report', str(report)], out / 'train.log')
    return labels


def measure(out, k, coarse, labels, graph_budget):
    settings = out / 'settings.csv'
    settings.unlink(missing_ok=True)
    with settings.open('x') as f:
        writer = csv.writer(f); writer.writerow(['routing', 'bounds', 'weight', 'metadata_cap', 'vector_cap', 'scan_order'])
        for cap in VECTOR_CAPS:
            writer.writerow(['flat', 'none', 0, k + coarse, cap, 'centroid'])
            writer.writerow(['graph', 'none', 0, graph_budget, cap, 'centroid'])
    args = [str(ROOT / 'build/graph_bench'), '--base', str(QWEN / 'data/base.rawview'),
            '--queries', str(QWEN / 'data/calibration_queries.fvecs'), '--truth', str(QWEN / 'data/calibration_truth.ivecs'),
            '--nq', str(CALIBRATION_QUERIES), '--int8-base', str(QWEN / 'base.i8'), '--int8-scales', str(QWEN / 'scales.f32'),
            '--int8-offsets', str(QWEN / 'offsets.f32'), '--variant', 'V0', '--coarse', str(coarse), '--children', str(k // coarse),
            '--leaves', str(k), '--iterations', '4', '--graph-degree', '8', '--settings', str(settings),
            '--build-threads', str(THREADS), '--search-cpu', str(SEARCH_CPU), '--assignments', str(labels),
            '--memory-output', str(out / 'memory.json'),
            '--evaluation-queries', str(QWEN / 'data/test_queries.fvecs'), '--evaluation-truth', str(QWEN / 'data/test_truth.ivecs'),
            '--evaluation-nq', str(HELDOUT_QUERIES), '--evaluation-output', str(out / 'heldout.csv'), '--repeats', '1']
    for stale in ['calibration.csv', 'run.log', 'memory.json', 'heldout.csv', 'heldout.csv.selection.csv', 'heldout.csv.memory.csv']:
        (out / stale).unlink(missing_ok=True)
    status('measuring', run=out.name, command=args)
    with (out / 'calibration.csv').open('x') as stdout, (out / 'run.log').open('x') as stderr:
        code = subprocess.run(args, stdout=stdout, stderr=stderr).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code)) + '\n')
    if code:
        raise RuntimeError(f'{out.name} failed: {code}')


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status('building_and_validating')
        checked(['cmake', '-S', str(ROOT / 'source'), '-B', str(ROOT / 'build'), '-DCMAKE_BUILD_TYPE=Release'], ROOT / 'configure.log')
        checked(['cmake', '--build', str(ROOT / 'build'), '-j', '16'], ROOT / 'build.log')
        checked(['ctest', '--test-dir', str(ROOT / 'build'), '--output-on-failure'], ROOT / 'ctest.log')
        (ROOT / 'results').mkdir(exist_ok=True)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'source').iterdir()) if p.is_file()}
        (ROOT / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
        for name, k, coarse, method_args, graph_budget in MODELS:
            out = ROOT / 'results' / name; out.mkdir(exist_ok=True)
            if (out / 'exit.json').exists():
                continue
            measure(out, k, coarse, train(out, k, method_args), graph_budget)
            checked([sys.executable, str(ROOT / 'summarize.py'), str(ROOT)], ROOT / 'summary.log')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
