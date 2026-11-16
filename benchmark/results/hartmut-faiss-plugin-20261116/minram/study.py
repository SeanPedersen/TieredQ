"""#55 step 4, the lowest RAM reaching recall 0.95: TieredQIVFIndex and IndexIVFTieredQ at the budgets between those
measured (arxiv 2 and 2.25 bits per dimension on seeds 1-3; Qwen 2.5 and 3 bits on seed 1), and Faiss's RaBitQ at 3
bits (arxiv seeds 1-3, Qwen seed 1), which the base study skipped. Same partitions as TieredQ-issue55-faiss-20261116
(the high-bit study showed they reproduce), same acceptance.cpp (source at SOURCE_COMMIT) and Faiss v1.16.0 install;
Qwen probes up to 4,096 lists, since its 2-bit curve was still rising at 2,048. RaBitQ (3 bits here, 4 bits on Qwen,
2-4 bits on arxiv) is served both with Faiss's default 4-bit queries and with 8-bit queries. Under the shared lock; resumable per
step; indexes removed per configuration."""
import datetime, fcntl, json, os, shutil, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
BASE = EXPERIMENTS / 'TieredQ-issue55-faiss-20261116'
SOURCE = ROOT / 'source/tieredq'
SEARCH_CPU, SEARCH_SIBLING = 2, 20
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (SEARCH_CPU, SEARCH_SIBLING, 19, 35))
# (dataset, TieredQ bits per dimension, seed, Faiss codecs built alongside)
CONFIGS = [('qwen', '3', 1, 'rabitq3'), ('qwen', '2.5', 1, 'rabitq4')] + \
          [('arxiv', '2', seed, 'rabitq2,rabitq3,rabitq4') for seed in (1, 2, 3)] + \
          [('arxiv', '2.25', seed, 'tieredq') for seed in (1, 2, 3)]
# RaBitQ is also served with 8-bit queries (Faiss's default is 4), its most accurate setting short of exact queries.
RABITQ_QB = '8'
BENCH = ROOT / 'plugin-build/faiss_tieredq_acceptance'
ENVIRONMENT = dict(os.environ, QUERY_CORE=str(SEARCH_CPU), OMP_NUM_THREADS='32', OPENBLAS_NUM_THREADS='1',
                   OMP_WAIT_POLICY='PASSIVE', OMP_PROC_BIND='false', OMP_DYNAMIC='false',
                   )
NPROBES = {'arxiv': '8,16,32,48,64,96,128,192,256', 'qwen': '64,128,256,512,1024,2048,4096'}
PLUGIN = (f"set -e; cd {ROOT}; cmake -S {SOURCE}/faiss-tieredq -B plugin-build -DCMAKE_BUILD_TYPE=Release "
          f"-DCMAKE_PREFIX_PATH={BASE}/faiss-install -DFAISS_TIEREDQ_FAISS_TARGET=faiss_avx512 -DFAISS_TIEREDQ_BENCHMARKS=ON; "
          f"cmake --build plugin-build -j 32; ctest --test-dir plugin-build --output-on-failure")


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def host():
    load = open('/proc/loadavg').read().split()[:3]
    top = subprocess.run(['ps', '-eo', 'pcpu,comm', '--sort=-pcpu'], capture_output=True, text=True).stdout.splitlines()[1:6]
    return dict(loadavg=load, top=[line.strip() for line in top])


def environment(step_name):
    grid = NPROBES.get(step_name.split('-', 1)[0])
    env = dict(ENVIRONMENT, NPROBES=grid) if grid else dict(ENVIRONMENT)
    if step_name.endswith(f'-qb{RABITQ_QB}'):
        env['RABITQ_QB'] = RABITQ_QB
    return env


def step(name, command, cores):
    out = ROOT / 'results' / name
    if (out / 'exit.json').exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    status('running', step=name)
    (out / 'host.json').write_text(json.dumps(host(), indent=2) + '\n')
    started = time.time()
    with (out / 'run.log').open('w') as log:
        code = subprocess.run(['taskset', '-c', cores, *map(str, command)], stdout=log, stderr=subprocess.STDOUT,
                              env=environment(name)).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
    if code:
        raise RuntimeError(f'{name} failed: {code}')
    return out


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        step('plugin', ['zsh', '-c', PLUGIN], BUILD_CORES)
        serve_cores = f'{BUILD_CORES},{SEARCH_CPU}'
        for dataset, bits, seed, codecs in CONFIGS:
            data = BASE / 'data' / dataset
            if True:
                tag = f'{dataset}-b{bits}-s{seed}'
                indexes = ROOT / 'indexes' / tag
                indexes.mkdir(parents=True, exist_ok=True)
                results = ROOT / 'results' / tag
                step(f'{tag}/build', [BENCH, 'build', data, seed, indexes, bits, codecs], BUILD_CORES)
                shutil.copy(indexes / 'build.json', results / 'build' / 'build.json')
                step(f'{tag}/parity', [BENCH, 'parity', data, indexes, results / 'parity' / 'parity.json'], serve_cores)
                faiss = [] if codecs == 'tieredq' else [('faiss', f'{name}.faiss', name) for name in codecs.split(',')]
                faiss += [('faiss', f'{name}.faiss', f'{name}-qb{RABITQ_QB}') for name in codecs.split(',') if name.startswith('rabitq')]
                for kind, name, label in [('tieredq', 'tieredq.tq', 'tieredq'), ('plugin', 'plugin.tqf', 'plugin')] + faiss:
                    step(f'{tag}/serve-{label}', [BENCH, 'serve', data, indexes, kind, name,
                                                  results / f'serve-{label}' / 'serve.json'], serve_cores)
                shutil.rmtree(indexes)
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
