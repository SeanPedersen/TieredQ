"""Experiment 3, stage B: our index on flat Lloyd partitions at 32K/64K/128K lists, routed by a TieredQ-compressed
centroid table (2 bits/dim, 4-bit scan cap) with lazy candidate expansion and an exact re-rank of the head. Scanning uses
fixed vector caps or patience (windows scale with the list count, since leaves shrink). Controls: one-shot ranking and FP32
flat routing at 32K, FP32 flat at 64K, degree-16 graph routing at 32K/64K/128K. Routing distances (graph walk, exact
re-rank head) and graph construction use the FP32 SIMD kernel (--float-routing 1). Faiss baselines and the 64K/128K labels come from
stage A (TieredQ-issue44-finelists-20261110). Calibration (250) picks by p50; every setting is replayed on 750 held-out
queries. Runs under the shared benchmark lock; resumable via exit.json."""
import csv, datetime, fcntl, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
STAGE_A = EXPERIMENTS / 'TieredQ-issue44-finelists-20261110/results'
LABELS = {32768: EXPERIMENTS / 'TieredQ-issue44-emeans-20261110/results/lloyd-32k/labels.u32',
          65536: STAGE_A / 'lloyd-64k/labels.u32', 131072: STAGE_A / 'lloyd-128k/labels.u32'}
COARSE = 128
CALIBRATION_QUERIES, HELDOUT_QUERIES = 250, 750
THREADS, SEARCH_CPU = 32, 2
NO_CAP = 600000
FIXED_CAPS = [35000, 88408, 250000]
TAIL_CAP = 350000
ROUTING_BITS, SCAN_BITS = 2, 4
# name, leaves, routing (compressed|graph), initial ranking depth (0: one shot), candidate limit, exact re-rank,
# patience windows, FP32 flat controls, graph degree/seeds/budgets.
MODELS = [
    dict(name='c32k-lazy', leaves=32768, routing='compressed', initial=256, limit=4096, rerank=256, windows=[48, 96, 192, 768], flat=True),
    dict(name='c32k-once', leaves=32768, routing='compressed', initial=0, limit=1024, rerank=256, windows=[48, 96, 192, 768], flat=False),
    dict(name='c64k-lazy', leaves=65536, routing='compressed', initial=512, limit=8192, rerank=512, windows=[96, 192, 384, 1536], flat=True),
    dict(name='c128k-lazy', leaves=131072, routing='compressed', initial=1024, limit=16384, rerank=512, windows=[192, 384, 768, 3072], flat=False),
    dict(name='g32k-d16', leaves=32768, routing='graph', degree=16, seeds=16, budgets=[2048, 4096, 8192], windows=[48, 96, 192, 768], flat=False),
    dict(name='g64k-d16', leaves=65536, routing='graph', degree=16, seeds=16, budgets=[2048, 4096, 8192, 16384], windows=[96, 192, 384, 1536], flat=False),
    dict(name='g128k-d16', leaves=131072, routing='graph', degree=16, seeds=16, budgets=[4096, 8192, 16384], windows=[192, 384, 768, 3072], flat=False),
]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def checked(args, log):
    with log.open('w') as out:
        subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, check=True)


def write_settings(path, model):
    path.unlink(missing_ok=True)
    windows = model['windows']
    rows = [(cap, 0) for cap in FIXED_CAPS] + [(NO_CAP, w) for w in windows] + [(TAIL_CAP, windows[-1])]
    budgets = model.get('budgets', [model.get('rerank')])
    with path.open('x') as f:
        writer = csv.writer(f)
        writer.writerow(['routing', 'bounds', 'weight', 'metadata_cap', 'vector_cap', 'scan_order', 'patience'])
        for budget in budgets:
            for cap, window in rows:
                writer.writerow([model['routing'], 'none', 0, budget, cap, 'centroid', window])
        if model['flat']:
            for cap, window in [(35000, 0), (250000, 0), (NO_CAP, windows[0]), (NO_CAP, windows[-1])]:
                writer.writerow(['flat', 'none', 0, model['leaves'] + COARSE, cap, 'centroid', window])


def measure(out, model):
    settings = out / 'settings.csv'
    write_settings(settings, model)
    leaves = model['leaves']
    if model['routing'] == 'graph':
        routing = ['--graph-degree', str(model['degree']), '--graph-seeds', str(model['seeds'])]
    else:
        routing = ['--routing-bits', str(ROUTING_BITS), '--routing-scan-bits', str(SCAN_BITS),
                   '--routing-candidates', str(model['limit']), '--routing-initial', str(model['initial'])]
    args = [str(ROOT / 'build/graph_bench'), '--base', str(QWEN / 'data/base.rawview'),
            '--queries', str(QWEN / 'data/calibration_queries.fvecs'), '--truth', str(QWEN / 'data/calibration_truth.ivecs'),
            '--nq', str(CALIBRATION_QUERIES), '--int8-base', str(QWEN / 'base.i8'), '--int8-scales', str(QWEN / 'scales.f32'),
            '--int8-offsets', str(QWEN / 'offsets.f32'), '--variant', 'V0', '--coarse', str(COARSE), '--children', str(leaves // COARSE),
            '--leaves', str(leaves), '--iterations', '4', *routing, '--float-routing', '1', '--settings', str(settings),
            '--build-threads', str(THREADS), '--search-cpu', str(SEARCH_CPU), '--assignments', str(LABELS[leaves]),
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
        for model in MODELS:
            out = ROOT / 'results' / model['name']; out.mkdir(exist_ok=True)
            if not (out / 'exit.json').exists():
                measure(out, model)
        checked([sys.executable, str(ROOT / 'summarize.py'), str(ROOT)], ROOT / 'summary.log')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
