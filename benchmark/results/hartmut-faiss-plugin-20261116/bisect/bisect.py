"""#55: the RAM TieredQ IVF needs to match each RaBitQ IVF codec's recall, by searching TieredQ's bit budget.

Targets: the best recall each Faiss IndexIVFRaBitQ (1-4 bits, 8-bit queries, its most accurate setting) reaches over
the shared nprobe grid (arxiv up to 256 lists, recall@100; Qwen up to 4,096, recall@10), on partition seed 1. RaBitQ 3
and 4 bits (both datasets) and 2 bits (arxiv) come from TieredQ-issue55-minram-20261116; the rest are built here on
the same partition (Qwen reuses that study's seed-1 centroids, arxiv recomputes its k-means, which reproduces).

Search per target: TieredQ's best recall over the grid is interpolated against bits per dimension from every
TieredQIVFIndex point measured on that partition and grid, and the budget predicted to reach the target (rounded up to
1/16 bit) is built and served. This repeats until the smallest budget known to reach the target lies within 1/8 bit
of the largest known to miss it, or after MAX_STEPS. At the found budget IndexIVFTieredQ is served too. Steps run under
the shared lock and resume from their exit.json. Results: results/<dataset>-b<bits>/serve-*, and bisect.json."""
import datetime, fcntl, json, math, os, shutil, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
BASE, MINRAM = EXPERIMENTS / 'TieredQ-issue55-faiss-20261116', EXPERIMENTS / 'TieredQ-issue55-minram-20261116'
HIGH_BITS = EXPERIMENTS / 'TieredQ-issue55-highbits-20261116'
SEARCH_CPU, SEARCH_SIBLING = 2, 20
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (SEARCH_CPU, SEARCH_SIBLING, 19, 35))
SERVE_CORES = f'{BUILD_CORES},{SEARCH_CPU}'
BENCH = ROOT / 'plugin-build/faiss_tieredq_acceptance'
SOURCE = ROOT / 'source/tieredq'
NPROBES = {'arxiv': '8,16,32,48,64,96,128,192,256', 'qwen': '64,128,256,512,1024,2048,4096'}
DIMENSIONS = {'arxiv': 768, 'qwen': 1024}
RABITQ_BITS = (1, 2, 3, 4)
BIT_STEP, RESOLUTION, MAX_STEPS = 1 / 16, 1 / 8, 6
ENVIRONMENT = dict(os.environ, QUERY_CORE=str(SEARCH_CPU), OMP_NUM_THREADS='32', OPENBLAS_NUM_THREADS='1',
                   OMP_WAIT_POLICY='PASSIVE', OMP_PROC_BIND='false', OMP_DYNAMIC='false', RABITQ_QB='8')
PLUGIN = (f"set -e; cd {ROOT}; cmake -S {SOURCE}/faiss-tieredq -B plugin-build -DCMAKE_BUILD_TYPE=Release "
          f"-DCMAKE_PREFIX_PATH={BASE}/faiss-install -DFAISS_TIEREDQ_FAISS_TARGET=faiss_avx512 -DFAISS_TIEREDQ_BENCHMARKS=ON; "
          f"cmake --build plugin-build -j 32; ctest --test-dir plugin-build --output-on-failure")


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def step(name, command, cores, dataset=None):
    out = ROOT / 'results' / name
    if (out / 'exit.json').exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    status('running', step=name)
    env = dict(ENVIRONMENT, **({'NPROBES': NPROBES[dataset]} if dataset else {}))
    if dataset == 'qwen':
        env['CENTROIDS'] = str(MINRAM / 'centroids/qwen-s1.f32')
    started = time.time()
    with (out / 'run.log').open('w') as log:
        code = subprocess.run(['taskset', '-c', cores, *map(str, command)], stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
    if code:
        raise RuntimeError(f'{name} failed: {code}')
    return out


def ceiling(serve):
    return max(p['recall'] for p in json.loads(serve.read_text())['points'])


def known_points(dataset):
    """(bits, best recall, serve.json) of TieredQIVFIndex on partition seed 1 over this dataset's full grid."""
    grid = [int(n) for n in NPROBES[dataset].split(',')]
    points = {}
    for serve in [*MINRAM.glob(f'results/{dataset}-b*-s1/serve-tieredq/serve.json'),
                  *HIGH_BITS.glob(f'results/{dataset}-b*-s1/serve-tieredq/serve.json'),
                  *ROOT.glob(f'results/{dataset}-b*/serve-tieredq/serve.json')]:
        if serve.parent.parent.name.endswith('-plugin'):
            continue
        report = json.loads(serve.read_text())
        if [p['nprobe'] for p in report['points']] != grid:
            continue
        bits = float(serve.parent.parent.name.split('-b')[1].split('-s')[0])  # <dataset>-b<bits>[-s<seed>]
        points[bits] = (ceiling(serve), serve)
    return dict(sorted(points.items()))


def rabitq_target(dataset, bits):
    for serve in [*MINRAM.glob(f'results/{dataset}-b*-s1/serve-rabitq{bits}-qb8/serve.json'),
                  *ROOT.glob(f'results/{dataset}-rabitq/serve-rabitq{bits}-qb8/serve.json')]:
        return ceiling(serve), serve
    return None


def build_and_serve(dataset, bits, codecs='tieredq', serves=(('tieredq', 'tieredq.tq'),), tag=None):
    tag = tag or (f'{dataset}-b{bits:g}' if codecs == 'tieredq' else f'{dataset}-rabitq')
    indexes = ROOT / 'indexes' / tag
    indexes.mkdir(parents=True, exist_ok=True)
    data = BASE / 'data' / dataset
    step(f'{tag}/build', [BENCH, 'build', data, 1, indexes, f'{bits:g}', codecs], BUILD_CORES, dataset)
    for kind, name in serves:
        label = name.rsplit('.', 1)[0] + ('-qb8' if name.startswith('rabitq') else '')
        step(f'{tag}/serve-{label}', [BENCH, 'serve', data, indexes, kind, name, ROOT / 'results' / tag / f'serve-{label}' / 'serve.json'],
             SERVE_CORES, dataset)
    shutil.rmtree(indexes, ignore_errors=True)


def predict(points, target):
    """Smallest budget the measured (bits, recall) curve, linearly interpolated, puts at the target, on the grid."""
    ordered = sorted(points.items())
    for (b0, (r0, _)), (b1, (r1, _)) in zip(ordered, ordered[1:]):
        if r0 < target <= r1:
            return math.ceil((b0 + (target - r0) / (r1 - r0) * (b1 - b0)) / BIT_STEP) * BIT_STEP
    if ordered and target <= ordered[0][1][0]:
        return max(BIT_STEP, ordered[0][0] / 2)
    return ordered[-1][0] + 1  # beyond the largest budget measured


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
        report = {}
        for dataset in ('qwen', 'arxiv'):
            missing = [b for b in RABITQ_BITS if rabitq_target(dataset, b) is None]
            if missing:  # RaBitQ the earlier study did not build on this partition; TieredQ at 1.5 bits comes along
                build_and_serve(dataset, 1.5, ','.join(f'rabitq{b}' for b in missing),
                                [('tieredq', 'tieredq.tq')] + [('faiss', f'rabitq{b}.faiss') for b in missing])
                # the TieredQ serve of that build is a known point under its own tag
                src = ROOT / 'results' / f'{dataset}-rabitq' / 'serve-tieredq'
                dst = ROOT / 'results' / f'{dataset}-b1.5' / 'serve-tieredq'
                if src.exists() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(src, dst)
            for rabitq_bits in RABITQ_BITS:
                target, target_serve = rabitq_target(dataset, rabitq_bits)
                for _ in range(MAX_STEPS):
                    points = known_points(dataset)
                    low, high = bracket(points, target)
                    if high is not None and low is not None and high - low <= RESOLUTION + 1e-9:
                        break
                    bits = predict(points, target)
                    if high is not None and bits >= high:
                        bits = high - RESOLUTION if low is None else math.ceil(((low + high) / 2) / BIT_STEP) * BIT_STEP
                    if bits in points:
                        break
                    build_and_serve(dataset, bits)
                points = known_points(dataset)
                low, high = bracket(points, target)
                entry = dict(target_recall=target, rabitq_ram_bytes=json.loads(target_serve.read_text())['ram_bytes'],
                             tieredq_bits_reaching=high, tieredq_bits_missing=low)
                if high is not None:
                    confirm = f'{dataset}-b{high:g}-plugin'  # a build of its own: search builds are removed
                    build_and_serve(dataset, high, 'tieredq', [('plugin', 'plugin.tqf')], tag=confirm)
                    entry['tieredq_ram_bytes'] = json.loads(points[high][1].read_text())['ram_bytes']
                    plugin = ROOT / 'results' / confirm / 'serve-plugin/serve.json'
                    if plugin.exists():
                        entry['plugin_ram_bytes'] = json.loads(plugin.read_text())['ram_bytes']
                    entry['ram_ratio'] = entry['tieredq_ram_bytes'] / entry['rabitq_ram_bytes']
                report[f'{dataset} rabitq{rabitq_bits}'] = entry
                (ROOT / 'bisect.json').write_text(json.dumps(report, indent=2) + '\n')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
