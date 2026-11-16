"""#55: can IndexIVFTieredQ reach RaBitQ's recall with less RAM than the bisection found? Two codec levers, plugin only.

Variants (env of acceptance build): `res` (RESIDUAL=1: the codec encodes rows minus their list's fp16 centroid),
`rows` (CODEC_ROWS=262144: 8x the codec's training rows), `res-rows` (both). `base` is the bisection study's plugin.

Phase 1 (screen): each variant at the bisection's matched budgets (arxiv 1.25 / 2 / 2.625 bits/dim, Qwen 1.5 / 2.0625 /
3.25), partition seed 1 (Qwen reuses the minram study's centroids), same nprobe grids, recall@100 (arxiv) and recall@10
(Qwen). Compared with base's plugin serves at the same budget and grid.
Phase 2 (bisect): per dataset, the variant with the largest mean ceiling gain over base (if at least min_gain), searched
for the smallest budget reaching each RaBitQ 1-4 bit target (bisection study's targets), to 1/8 bit, plugin serves only.
Steps run under the shared lock and resume from their exit.json. Results: results/<dataset>-<variant>-b<bits>/, study.json.
"""
import datetime, fcntl, json, math, os, shutil, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
BASE, MINRAM = EXPERIMENTS / 'TieredQ-issue55-faiss-20261116', EXPERIMENTS / 'TieredQ-issue55-minram-20261116'
BISECT = EXPERIMENTS / 'TieredQ-issue55-bisect-20261116'
SEARCH_CPU, SEARCH_SIBLING = 2, 20
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (SEARCH_CPU, SEARCH_SIBLING, 19, 35))
SERVE_CORES = f'{BUILD_CORES},{SEARCH_CPU}'
BENCH = ROOT / 'plugin-build/faiss_tieredq_acceptance'
SOURCE = ROOT / 'source/tieredq'
NPROBES = {'arxiv': '8,16,32,48,64,96,128,192,256', 'qwen': '64,128,256,512,1024,2048,4096'}
SCREEN_BITS = {'arxiv': (1.25, 2.0, 2.625), 'qwen': (1.5, 2.0625, 3.25)}
VARIANTS = {'res': {'RESIDUAL': '1'}, 'rows': {'CODEC_ROWS': '262144'}, 'res-rows': {'RESIDUAL': '1', 'CODEC_ROWS': '262144'}}
RABITQ_BITS = (1, 2, 3, 4)
BIT_STEP, RESOLUTION, MAX_STEPS, MIN_GAIN = 1 / 16, 1 / 8, 6, 0.002
ENVIRONMENT = dict(os.environ, QUERY_CORE=str(SEARCH_CPU), OMP_NUM_THREADS='32', OPENBLAS_NUM_THREADS='1',
                   OMP_WAIT_POLICY='PASSIVE', OMP_PROC_BIND='false', OMP_DYNAMIC='false')
PLUGIN = (f"set -e; cd {ROOT}; cmake -S {SOURCE}/faiss-tieredq -B plugin-build -DCMAKE_BUILD_TYPE=Release "
          f"-DCMAKE_PREFIX_PATH={BASE}/faiss-install -DFAISS_TIEREDQ_FAISS_TARGET=faiss_avx512 -DFAISS_TIEREDQ_BENCHMARKS=ON; "
          f"cmake --build plugin-build -j 32; ctest --test-dir plugin-build --output-on-failure")


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def step(name, command, cores, env=None):
    out = ROOT / 'results' / name
    if (out / 'exit.json').exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    status('running', step=name)
    started = time.time()
    with (out / 'run.log').open('w') as log:
        code = subprocess.run(['taskset', '-c', cores, *map(str, command)], stdout=log, stderr=subprocess.STDOUT,
                              env=dict(ENVIRONMENT, **(env or {}))).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
    if code:
        raise RuntimeError(f'{name} failed: {code}')
    return out


def ceiling(serve):
    return max(p['recall'] for p in json.loads(serve.read_text())['points'])


def build_and_serve(dataset, variant, bits):
    """The variant's plugin at this budget; returns its serve.json."""
    tag = f'{dataset}-{variant}-b{bits:g}'
    serve = ROOT / 'results' / tag / 'serve-plugin' / 'serve.json'
    if serve.exists():
        return serve
    indexes = ROOT / 'indexes' / tag
    indexes.mkdir(parents=True, exist_ok=True)
    data = BASE / 'data' / dataset
    env = dict(VARIANTS[variant], NPROBES=NPROBES[dataset])
    if dataset == 'qwen':
        env['CENTROIDS'] = str(MINRAM / 'centroids/qwen-s1.f32')
    step(f'{tag}/build', [BENCH, 'build', data, 1, indexes, f'{bits:g}', 'tieredq'], BUILD_CORES, env)
    step(f'{tag}/serve-plugin', [BENCH, 'serve', data, indexes, 'plugin', 'plugin.tqf', serve], SERVE_CORES, env)
    shutil.rmtree(indexes, ignore_errors=True)
    return serve


def base_serve(dataset, bits):
    serve = BISECT / 'results' / f'{dataset}-b{bits:g}-plugin' / 'serve-plugin' / 'serve.json'
    return serve if serve.exists() else None


def known_points(dataset, variant):
    points = {}
    for serve in ROOT.glob(f'results/{dataset}-{variant}-b*/serve-plugin/serve.json'):
        bits = float(serve.parent.parent.name.rsplit('-b', 1)[1])
        points[bits] = (ceiling(serve), serve)
    return dict(sorted(points.items()))


def predict(points, target):
    ordered = sorted(points.items())
    for (b0, (r0, _)), (b1, (r1, _)) in zip(ordered, ordered[1:]):
        if r0 < target <= r1:
            return math.ceil((b0 + (target - r0) / (r1 - r0) * (b1 - b0)) / BIT_STEP) * BIT_STEP
    if target <= ordered[0][1][0]:
        lower = ordered[0][0] - 0.25 if len(ordered) == 1 else ordered[0][0] - (ordered[1][0] - ordered[0][0])
        return max(BIT_STEP, math.floor(lower / BIT_STEP) * BIT_STEP)
    return ordered[-1][0] + 0.5


def bracket(points, target):
    reach = [b for b, (r, _) in points.items() if r >= target]
    miss = [b for b, (r, _) in points.items() if r < target and (not reach or b < min(reach))]
    return (max(miss) if miss else None), (min(reach) if reach else None)


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        step('plugin', ['zsh', '-c', PLUGIN], BUILD_CORES)
        report = {'screen': {}, 'bisect': {}}
        bisection = json.loads((BISECT / 'bisect.json').read_text())
        for dataset in ('arxiv', 'qwen'):
            gains = {}
            for variant in VARIANTS:
                for bits in SCREEN_BITS[dataset]:
                    serve, base = build_and_serve(dataset, variant, bits), base_serve(dataset, bits)
                    entry = dict(ceiling=ceiling(serve), ram_bytes=json.loads(serve.read_text())['ram_bytes'])
                    if base:
                        entry.update(base_ceiling=ceiling(base), base_ram_bytes=json.loads(base.read_text())['ram_bytes'])
                        gains.setdefault(variant, []).append(entry['ceiling'] - entry['base_ceiling'])
                    report['screen'][f'{dataset} {variant} {bits:g}'] = entry
                    (ROOT / 'study.json').write_text(json.dumps(report, indent=2) + '\n')
            mean_gain = {v: sum(g) / len(g) for v, g in gains.items()}
            best = max(mean_gain, key=mean_gain.get)
            report['screen'][f'{dataset} mean ceiling gain'] = mean_gain
            if mean_gain[best] < MIN_GAIN:
                continue
            for rabitq_bits in RABITQ_BITS:
                target_entry = bisection[f'{dataset} rabitq{rabitq_bits}']
                target = target_entry['target_recall']
                for _ in range(MAX_STEPS):
                    points = known_points(dataset, best)
                    low, high = bracket(points, target)
                    if high is not None and low is not None and high - low <= RESOLUTION + 1e-9:
                        break
                    bits = predict(points, target)
                    if high is not None and bits >= high:
                        bits = high - RESOLUTION if low is None else math.ceil(((low + high) / 2) / BIT_STEP) * BIT_STEP
                    if low is not None and bits <= low:
                        bits = low + RESOLUTION if high is None else math.ceil(((low + high) / 2) / BIT_STEP) * BIT_STEP
                    if bits in points:
                        break
                    build_and_serve(dataset, best, bits)
                points = known_points(dataset, best)
                low, high = bracket(points, target)
                entry = dict(variant=best, target_recall=target, rabitq_ram_bytes=target_entry['rabitq_ram_bytes'],
                             base_plugin_bits=target_entry['tieredq_bits_reaching'],
                             base_plugin_ram_bytes=target_entry.get('plugin_ram_bytes'), bits_reaching=high, bits_missing=low)
                if high is not None:
                    entry['ram_bytes'] = json.loads(points[high][1].read_text())['ram_bytes']
                    entry['ram_ratio'] = entry['ram_bytes'] / entry['rabitq_ram_bytes']
                report['bisect'][f'{dataset} rabitq{rabitq_bits}'] = entry
                (ROOT / 'study.json').write_text(json.dumps(report, indent=2) + '\n')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
