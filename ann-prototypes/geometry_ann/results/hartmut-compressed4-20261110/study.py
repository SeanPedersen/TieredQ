"""Rerun with a 4-bit per-coordinate scan cap (x86 nibble-lookup kernel) and fewer candidates.
Flat routing over TieredQ-compressed leaf centroids vs exact FP32 flat routing (Qwen SQ8, Lloyd 32K labels).
The compressed table ranks `--routing-candidates` leaves; the first R (metadata_cap) are re-ranked with exact centroid
distances; scanning uses fixed caps or patience. FP32 flat controls run on the same build. Calibration (250) picks by p50
per routing, policy and target, frozen before held-out; every setting is replayed on the 750 held-out queries.
The TieredQ library is snapshotted in source/tieredq. Runs under the shared benchmark lock; resumable via exit.json."""
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
FIXED_CAPS = [35000, 88408, 250000]
PATIENCE_WINDOWS = [48, 96, 192, 768]
RERANKS = [256, 512]
SCAN_BITS = 4
# name, bits per dimension of the compressed centroid table, candidates ranked by the table.
MODELS = [('tq2-s4-c1024', 2, 1024), ('tq2-s4-c2048', 2, 2048), ('tq1-s4-c1024', 1, 1024)]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def checked(args, log):
    with log.open('w') as out:
        subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, check=True)


def write_settings(path):
    path.unlink(missing_ok=True)
    rows = [(cap, 0) for cap in FIXED_CAPS] + [(NO_CAP, w) for w in PATIENCE_WINDOWS]
    with path.open('x') as f:
        writer = csv.writer(f)
        writer.writerow(['routing', 'bounds', 'weight', 'metadata_cap', 'vector_cap', 'scan_order', 'patience'])
        for rerank in RERANKS:
            for cap, window in rows:
                writer.writerow(['compressed', 'none', 0, rerank, cap, 'centroid', window])
        for cap, window in [(35000, 0), (250000, 0), (NO_CAP, 64), (NO_CAP, 768)]:
            writer.writerow(['flat', 'none', 0, LEAVES + COARSE, cap, 'centroid', window])


def measure(out, bits, candidates):
    settings = out / 'settings.csv'
    write_settings(settings)
    args = [str(ROOT / 'build/graph_bench'), '--base', str(QWEN / 'data/base.rawview'),
            '--queries', str(QWEN / 'data/calibration_queries.fvecs'), '--truth', str(QWEN / 'data/calibration_truth.ivecs'),
            '--nq', str(CALIBRATION_QUERIES), '--int8-base', str(QWEN / 'base.i8'), '--int8-scales', str(QWEN / 'scales.f32'),
            '--int8-offsets', str(QWEN / 'offsets.f32'), '--variant', 'V0', '--coarse', str(COARSE), '--children', str(LEAVES // COARSE),
            '--leaves', str(LEAVES), '--iterations', '4', '--routing-bits', str(bits), '--routing-candidates', str(candidates), '--routing-scan-bits', str(SCAN_BITS), '--settings', str(settings),
            '--build-threads', str(THREADS), '--search-cpu', str(SEARCH_CPU), '--assignments', str(LABELS),
            '--memory-output', str(out / 'memory.json'), '--select-by', 'p50',
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
        checked(['cmake', '-S', str(ROOT / 'source'), '-B', str(ROOT / 'build'), '-DCMAKE_BUILD_TYPE=Release',
                 f"-DTIEREDQ_SOURCE_DIR={ROOT / 'source/tieredq'}"], ROOT / 'configure.log')
        checked(['cmake', '--build', str(ROOT / 'build'), '-j', '16'], ROOT / 'build.log')
        checked(['ctest', '--test-dir', str(ROOT / 'build'), '--output-on-failure'], ROOT / 'ctest.log')
        (ROOT / 'results').mkdir(exist_ok=True)
        hashes = {str(p.relative_to(ROOT / 'source')): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((ROOT / 'source').rglob('*')) if p.is_file()}
        (ROOT / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
        for name, bits, candidates in MODELS:
            out = ROOT / 'results' / name; out.mkdir(exist_ok=True)
            if not (out / 'exit.json').exists():
                measure(out, bits, candidates)
        checked([sys.executable, str(ROOT / 'summarize.py'), str(ROOT)], ROOT / 'summary.log')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
