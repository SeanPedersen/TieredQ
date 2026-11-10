"""Fast final assignment: rows are assigned among the members of their top-P super-centroids (k-means over the 32K
Lloyd centroids, grouped GEMMs) instead of all centroids. Same Lloyd centroids (seed 42, 40 rows/centroid, 4 iterations)
for every label set; exhaustive grouped assignment is the exact control and must reproduce the existing exact labels.
Each label set is measured with our index (compressed lazy routing; fixed caps and patience; held-out replay of all
settings), and Faiss IVF-SQ8 is built on the same centroids and fast labels, so both systems hold identical partitions.
Runs under the shared benchmark lock; every step is resumable."""
import array, csv, datetime, fcntl, json, subprocess
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
EXACT_LABELS = EXPERIMENTS / 'TieredQ-issue44-emeans-20261110/results/lloyd-32k/labels.u32'
ROWS, DIM, LEAVES, COARSE = 8840823, 1024, 32768, 128
CALIBRATION_QUERIES, HELDOUT_QUERIES = 250, 750
THREADS, SEARCH_CPU = 32, 2
SUPERS = 256
# name, super-centroid probes (SUPERS: exhaustive control).
ASSIGNMENTS = [('exhaustive', SUPERS), ('p16', 16), ('p8', 8)]
FIXED_CAPS, WINDOWS, TAIL_CAP, NO_CAP = [35000, 88408, 250000], [48, 96, 192, 768], 350000, 600000
FAISS_BUDGETS = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def checked(args, log):
    with log.open('w') as out:
        subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, check=True)


def step(out, args, stdout_name):
    """Runs one resumable step: exit.json marks completion, stdout goes to `stdout_name`."""
    if (out / 'exit.json').exists():
        return
    status('running', step=out.name, command=args)
    with (out / stdout_name).open('w') as stdout, (out / 'run.log').open('w') as log:
        code = subprocess.run(args, stdout=stdout, stderr=log).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code)) + '\n')
    if code:
        raise RuntimeError(f'{out.name} failed: {code}')


def quantized():
    return ['--codes', str(QWEN / 'base.i8'), '--scales', str(QWEN / 'scales.f32'), '--offsets', str(QWEN / 'offsets.f32')]


def train(out):
    step(out, ['uv', 'run', '--no-project', '--python', '3.12', str(ROOT / 'source/train_partitions.py'), *quantized(),
               '--rows', str(ROWS), '--dim', str(DIM), '--k', str(LEAVES), '--method', 'lloyd', '--iterations', '4',
               '--sample-per-centroid', '40', '--threads', str(THREADS), '--assign', 'none',
               '--centroids-output', str(out / 'centroids.f32'), '--labels-output', str(out / 'unused.u32'),
               '--report', str(out / 'partition.json')], 'stdout.log')
    return out / 'centroids.f32'


def assign(out, centroids, probes):
    step(out, [str(ROOT / 'build/fast_assign'), *quantized(), '--rows', str(ROWS), '--dim', str(DIM), '--k', str(LEAVES),
               '--centroids', str(centroids), '--supers', str(SUPERS), '--probes', str(probes), '--threads', str(THREADS),
               '--check-rows', '50000', '--labels-output', str(out / 'labels.u32'), '--report', str(out / 'assignment.json')], 'stdout.log')
    return out / 'labels.u32'


def settings_file(path):
    path.unlink(missing_ok=True)
    rows = [(cap, 0) for cap in FIXED_CAPS] + [(NO_CAP, w) for w in WINDOWS] + [(TAIL_CAP, WINDOWS[-1])]
    with path.open('x') as f:
        writer = csv.writer(f)
        writer.writerow(['routing', 'bounds', 'weight', 'metadata_cap', 'vector_cap', 'scan_order', 'patience'])
        for cap, window in rows:
            writer.writerow(['compressed', 'none', 0, 256, cap, 'centroid', window])


def ours(out, labels):
    settings_file(out / 'settings.csv')
    step(out, [str(ROOT / 'build/graph_bench'), '--base', str(QWEN / 'data/base.rawview'),
               '--queries', str(QWEN / 'data/calibration_queries.fvecs'), '--truth', str(QWEN / 'data/calibration_truth.ivecs'),
               '--nq', str(CALIBRATION_QUERIES), '--int8-base', str(QWEN / 'base.i8'), '--int8-scales', str(QWEN / 'scales.f32'),
               '--int8-offsets', str(QWEN / 'offsets.f32'), '--variant', 'V0', '--coarse', str(COARSE), '--children', str(LEAVES // COARSE),
               '--leaves', str(LEAVES), '--iterations', '4', '--routing-bits', '2', '--routing-scan-bits', '4',
               '--routing-candidates', '4096', '--routing-initial', '256', '--settings', str(out / 'settings.csv'),
               '--build-threads', str(THREADS), '--search-cpu', str(SEARCH_CPU), '--assignments', str(labels),
               '--memory-output', str(out / 'memory.json'), '--select-by', 'p50',
               '--evaluation-queries', str(QWEN / 'data/test_queries.fvecs'), '--evaluation-truth', str(QWEN / 'data/test_truth.ivecs'),
               '--evaluation-nq', str(HELDOUT_QUERIES), '--evaluation-output', str(out / 'heldout.csv'),
               '--evaluation-fixed-settings', str(out / 'settings.csv'), '--repeats', '1'], 'calibration.csv')


def faiss(out, centroids, labels):
    step(out, ['uv', 'run', '--no-project', '--python', '3.12', '--with', 'faiss-cpu==1.15.1', '--with', 'numpy',
               str(ROOT / 'source/compare_faiss.py'), '--base', str(QWEN / 'data/base.rawview'),
               '--queries', str(QWEN / 'data/calibration_queries.fvecs'), '--truth', str(QWEN / 'data/calibration_truth.ivecs'),
               '--nq', str(CALIBRATION_QUERIES), '--variant', 'ivf', '--nlist', str(LEAVES), '--budgets', ','.join(map(str, FAISS_BUDGETS)),
               '--int8-base', str(QWEN / 'base.i8'), '--int8-scales', str(QWEN / 'scales.f32'), '--int8-offsets', str(QWEN / 'offsets.f32'),
               '--centroids', str(centroids), '--assignments', str(labels),
               '--evaluation-queries', str(QWEN / 'data/test_queries.fvecs'), '--evaluation-truth', str(QWEN / 'data/test_truth.ivecs'),
               '--evaluation-nq', str(HELDOUT_QUERIES), '--evaluation-output', str(out / 'heldout.csv'),
               '--targets', '0.95,0.99,0.995', '--build-threads', str(THREADS), '--search-cpu', str(SEARCH_CPU),
               '--memory-output', str(out / 'memory.json')], 'calibration.csv')


def agreement(a, b):
    """Fraction of rows with the same label (BLAS summation order may flip exact near-ties)."""
    left, right = array.array('I'), array.array('I')
    left.frombytes(a.read_bytes()); right.frombytes(b.read_bytes())
    if len(left) != len(right):
        raise ValueError('label files differ in length')
    return sum(x == y for x, y in zip(left, right)) / len(left)


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status('building_and_validating')
        checked(['cmake', '-S', str(ROOT / 'source'), '-B', str(ROOT / 'build'), '-DCMAKE_BUILD_TYPE=Release',
                 f"-DTIEREDQ_SOURCE_DIR={ROOT / 'source/tieredq'}"], ROOT / 'configure.log')
        checked(['cmake', '--build', str(ROOT / 'build'), '-j', '16'], ROOT / 'build.log')
        checked(['ctest', '--test-dir', str(ROOT / 'build'), '--output-on-failure'], ROOT / 'ctest.log')
        results = ROOT / 'results'; results.mkdir(exist_ok=True)
        (results / 'train').mkdir(exist_ok=True)
        centroids = train(results / 'train')
        labels = {}
        for name, probes in ASSIGNMENTS:
            out = results / f'assign-{name}'; out.mkdir(exist_ok=True)
            labels[name] = assign(out, centroids, probes)
        # The exhaustive control should reproduce the existing exact labels (same seed, rows and centroids).
        check = {name: agreement(labels[name], EXACT_LABELS) for name, _ in ASSIGNMENTS}
        (results / 'agreement_with_existing_exact.json').write_text(json.dumps(check, indent=2) + '\n')
        for name, _ in ASSIGNMENTS:
            out = results / f'ours-{name}'; out.mkdir(exist_ok=True)
            ours(out, labels[name])
        out = results / 'faiss-p16'; out.mkdir(exist_ok=True)
        faiss(out, centroids, labels['p16'])
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
