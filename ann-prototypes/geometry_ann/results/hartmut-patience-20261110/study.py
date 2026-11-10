"""Patience stopping vs fixed vector caps in real search on the flat Lloyd 32K partition (Qwen SQ8).
Same index build, codes and evaluation path for both policies; only the stop rule differs.
Calibration (250 queries) picks the cheapest setting per policy and recall target by mean scanned vectors,
frozen before held-out data is read; every setting is also replayed on the 750 held-out queries for curves.
Runs under the shared benchmark lock; completed runs (exit.json) are skipped on resume."""
import csv, datetime, fcntl, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
LABELS = EXPERIMENTS / 'TieredQ-issue44-emeans-20261110/results/lloyd-32k/labels.u32'
LEAVES, COARSE = 32768, 128
CALIBRATION_QUERIES, HELDOUT_QUERIES = 250, 750
THREADS, SEARCH_CPU = 32, 2
NO_CAP = 600000
FIXED_CAPS = [20000, 35000, 50000, 70000, 88408, 120000, 176816, 250000, 350000, 442041, NO_CAP]
PATIENCE_WINDOWS = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]
# Tighter ceilings bound the p99 tail of patience at the cost of some recall.
TAIL_CAPPED = [(window, cap) for window in (64, 128, 256) for cap in (120000, 250000)]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def checked(args, log):
    with log.open('w') as out:
        subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, check=True)


def write_settings(path):
    path.unlink(missing_ok=True)
    rows = [(cap, 0) for cap in FIXED_CAPS] + [(NO_CAP, w) for w in PATIENCE_WINDOWS] + [(cap, w) for w, cap in TAIL_CAPPED]
    with path.open('x') as f:
        writer = csv.writer(f)
        writer.writerow(['routing', 'bounds', 'weight', 'metadata_cap', 'vector_cap', 'scan_order', 'patience'])
        for cap, window in rows:
            writer.writerow(['flat', 'none', 0, LEAVES + COARSE, cap, 'centroid', window])


def measure(out):
    settings = out / 'settings.csv'
    write_settings(settings)
    args = [str(ROOT / 'build/graph_bench'), '--base', str(QWEN / 'data/base.rawview'),
            '--queries', str(QWEN / 'data/calibration_queries.fvecs'), '--truth', str(QWEN / 'data/calibration_truth.ivecs'),
            '--nq', str(CALIBRATION_QUERIES), '--int8-base', str(QWEN / 'base.i8'), '--int8-scales', str(QWEN / 'scales.f32'),
            '--int8-offsets', str(QWEN / 'offsets.f32'), '--variant', 'V0', '--coarse', str(COARSE), '--children', str(LEAVES // COARSE),
            '--leaves', str(LEAVES), '--iterations', '4', '--graph-degree', '8', '--settings', str(settings),
            '--build-threads', str(THREADS), '--search-cpu', str(SEARCH_CPU), '--assignments', str(LABELS),
            '--memory-output', str(out / 'memory.json'), '--select-by', 'vectors',
            '--evaluation-queries', str(QWEN / 'data/test_queries.fvecs'), '--evaluation-truth', str(QWEN / 'data/test_truth.ivecs'),
            '--evaluation-nq', str(HELDOUT_QUERIES), '--evaluation-output', str(out / 'heldout.csv'),
            '--evaluation-fixed-settings', str(settings), '--repeats', '1']
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
        out = ROOT / 'results' / 'lloyd-32k'; out.mkdir(exist_ok=True)
        if not (out / 'exit.json').exists():
            measure(out)
        checked([sys.executable, str(ROOT / 'summarize.py'), str(ROOT)], ROOT / 'summary.log')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
