"""Leaf-assignment study: does lifting the hierarchical restriction cut scanned clusters/vectors at fixed recall?
Compares hierarchical k-means with/without leaf reassignment (8K, 32K leaves) to Faiss IVF-SQ8 with the same nlist.
Builds use BUILD_THREADS threads; every search runs single-threaded on SEARCH_CPU. Runs under the shared lock."""
import csv, datetime, fcntl, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
SIFT = EXPERIMENTS / 'TieredQ-issue44-sift-sq8-20261107'
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
CALIBRATION_QUERIES, HELDOUT_QUERIES = 250, 750
BUILD_THREADS, SEARCH_CPU = 32, 2
VECTOR_CAPS = [35000, 50000, 70000, 88408, 120000, 176816, 250000, 350000, 442041, 600000]
# name, coarse, children, training sample (32 per fine centroid), reassignment parents, graph budgets
OURS = [
    ('ours-8k-p0', 64, 128, 4096, 0, [2048, 4096]),
    ('ours-8k-p4', 64, 128, 4096, 4, [2048, 4096]),
    ('ours-32k-p0', 128, 256, 8192, 0, [4096, 8192, 16384]),
    ('ours-32k-p8', 128, 256, 8192, 8, [4096, 8192, 16384]),
]
# Faiss: 40 training samples per list, as in the earlier 8K reference.
FAISS = [('faiss-8k', 8192, 327680, [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]),
         ('faiss-32k', 32768, 1310720, [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536])]
DETERMINISTIC = ['routing', 'metadata_cap', 'vector_cap', 'recall10', 'mean_vectors', 'mean_metadata', 'mean_leaves', 'leaves', 'graph_build_distances']


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def checked(args, log):
    with log.open('w') as out:
        subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, check=True)


def run(out, args):
    out.mkdir(parents=True)
    status('running', run=str(out.relative_to(ROOT)), command=args)
    with (out / 'calibration.csv').open('x') as stdout, (out / 'run.log').open('x') as stderr:
        code = subprocess.run(args, stdout=stdout, stderr=stderr).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code)) + '\n')
    if code:
        raise RuntimeError(f'{out.name} failed: {code}')


def int8_args(data_root):
    return [arg for flag, file in [('base', 'base.i8'), ('scales', 'scales.f32'), ('offsets', 'offsets.f32')]
            for arg in ('--int8-' + flag, str(data_root / file))]


def write_settings(path, leaves, coarse, caps, budgets):
    with path.open('x') as f:
        writer = csv.writer(f); writer.writerow(['routing', 'bounds', 'weight', 'metadata_cap', 'vector_cap', 'scan_order'])
        for cap in caps:
            writer.writerow(['flat', 'none', 0, leaves + coarse, cap, 'centroid'])
            for budget in budgets:
                writer.writerow(['graph', 'none', 0, budget, cap, 'centroid'])


def ours(out, data_root, coarse, children, sample, parents, budgets, threads, caps, heldout=True):
    settings = ROOT / 'settings' / f'{out.name}.csv'
    settings.unlink(missing_ok=True)  # regenerated deterministically on resume
    write_settings(settings, coarse * children, coarse, caps, budgets)
    args = [str(ROOT / 'build/graph_bench'), '--base', str(data_root / 'data/base.rawview'),
            '--queries', str(data_root / 'data/calibration_queries.fvecs'), '--truth', str(data_root / 'data/calibration_truth.ivecs'),
            '--nq', str(CALIBRATION_QUERIES), *int8_args(data_root), '--variant', 'V0', '--coarse', str(coarse),
            '--children', str(children), '--leaves', str(coarse * children), '--iterations', '4', '--sample', str(sample),
            '--graph-degree', '8', '--settings', str(settings), '--build-threads', str(threads), '--search-cpu', str(SEARCH_CPU),
            '--reassign-parents', str(parents), '--reassign-iterations', '2', '--memory-output', str(out / 'memory.json')]
    if heldout:
        args += ['--evaluation-queries', str(data_root / 'data/test_queries.fvecs'), '--evaluation-truth', str(data_root / 'data/test_truth.ivecs'),
                 '--evaluation-nq', str(HELDOUT_QUERIES), '--evaluation-output', str(out / 'heldout.csv'), '--repeats', '1']
    run(out, args)
    return [{k: r[k] for k in DETERMINISTIC} for r in csv.DictReader((out / 'calibration.csv').open())]


def faiss(out, nlist, sample, budgets):
    run(out, ['uv', 'run', '--no-project', '--python', '3.12', str(ROOT / 'source/compare_faiss.py'),
              '--base', str(QWEN / 'data/base.rawview'), '--queries', str(QWEN / 'data/calibration_queries.fvecs'),
              '--truth', str(QWEN / 'data/calibration_truth.ivecs'), '--nq', str(CALIBRATION_QUERIES), '--variant', 'ivf',
              '--nlist', str(nlist), '--sample', str(sample), '--budgets', ','.join(map(str, budgets)),
              '--int8-base', str(QWEN / 'base.i8'), '--int8-scales', str(QWEN / 'scales.f32'), '--int8-offsets', str(QWEN / 'offsets.f32'),
              '--evaluation-queries', str(QWEN / 'data/test_queries.fvecs'), '--evaluation-truth', str(QWEN / 'data/test_truth.ivecs'),
              '--evaluation-nq', str(HELDOUT_QUERIES), '--evaluation-output', str(out / 'heldout.csv'),
              '--targets', '0.95,0.99,0.995', '--build-threads', str(BUILD_THREADS), '--search-cpu', str(SEARCH_CPU),
              '--memory-output', str(out / 'memory.json')])


def threading_gate():
    gate = ROOT / 'gate'
    serial = ours(gate / 'sift-p4-1thread', SIFT, 64, 128, 4096, 4, [2048], 1, [35000], heldout=False)
    threaded = ours(gate / f'sift-p4-{BUILD_THREADS}threads', SIFT, 64, 128, 4096, 4, [2048], BUILD_THREADS, [35000], heldout=False)
    if serial != threaded:
        raise RuntimeError('multi-threaded reassignment changed search results')
    (gate / 'identical.json').write_text(json.dumps(dict(identical=True, rows=len(serial)), indent=2) + '\n')


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status('building_and_validating')
        checked(['cmake', '-S', str(ROOT / 'source'), '-B', str(ROOT / 'build'), '-DCMAKE_BUILD_TYPE=Release'], ROOT / 'configure.log')
        checked(['cmake', '--build', str(ROOT / 'build'), '-j', '16'], ROOT / 'build.log')
        checked(['ctest', '--test-dir', str(ROOT / 'build'), '--output-on-failure'], ROOT / 'ctest.log')
        # Resumable: completed models (exit.json present) and a passed gate are not rerun.
        for folder in ['results', 'settings', 'gate']:
            (ROOT / folder).mkdir(exist_ok=True)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'source').iterdir()) if p.is_file()}
        (ROOT / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
        if not (ROOT / 'gate' / 'identical.json').exists():
            threading_gate()
        for name, coarse, children, sample, parents, budgets in OURS:
            if (ROOT / 'results' / name / 'exit.json').exists():
                continue
            ours(ROOT / 'results' / name, QWEN, coarse, children, sample, parents, budgets, BUILD_THREADS, VECTOR_CAPS)
            checked([sys.executable, str(ROOT / 'summarize.py'), str(ROOT)], ROOT / 'summary.log')
        for name, nlist, sample, budgets in FAISS:
            if (ROOT / 'results' / name / 'exit.json').exists():
                continue
            faiss(ROOT / 'results' / name, nlist, sample, budgets)
            checked([sys.executable, str(ROOT / 'summarize.py'), str(ROOT)], ROOT / 'summary.log')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
