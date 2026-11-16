"""#55 step 4, recall >= 0.95 on VIBE arxiv (and Qwen/MS MARCO 8.84M at 2 and 4 bits on seed 1): TieredQIVFIndex and IndexIVFTieredQ at 2.5, 3 and 4 bits per dimension on
the same three Faiss k-means partitions (seeds 1-3, 2,048 lists) as TieredQ-issue55-faiss-20261116, whose Faiss codecs
(RaBitQ 1/2/4 bits, IVF PQ fast scan at 1 and 2 bits per dimension) on those partitions are the comparison. The
benchmark is the same acceptance.cpp (source at SOURCE_COMMIT), against that study's Faiss v1.16.0 install; nprobe
runs to 256 so the curves reach their ceiling. A control rerun of 1.75 bits on seed 1 checks that the partitions reproduce. Per configuration: build (32 threads), parity, then each index served
by a fresh process (RAM, then alone on CPU 2). Under the shared lock; resumable per step; indexes removed per step."""
import datetime, fcntl, json, os, shutil, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
BASE = EXPERIMENTS / 'TieredQ-issue55-faiss-20261116'
SOURCE = ROOT / 'source/tieredq'
SEARCH_CPU, SEARCH_SIBLING = 2, 20
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (SEARCH_CPU, SEARCH_SIBLING, 19, 35))
SEEDS, BITS, QWEN_BITS = (1, 2, 3), ('2.5', '3', '4'), ('2', '4')
BENCH = ROOT / 'plugin-build/faiss_tieredq_acceptance'
ENVIRONMENT = dict(os.environ, QUERY_CORE=str(SEARCH_CPU), OMP_NUM_THREADS='32', OPENBLAS_NUM_THREADS='1',
                   OMP_WAIT_POLICY='PASSIVE', OMP_PROC_BIND='false', OMP_DYNAMIC='false',
                   )
NPROBES = {'arxiv': '8,16,32,48,64,96,128,192,256', 'qwen': '64,128,256,512,1024,2048'}
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
    return dict(ENVIRONMENT, NPROBES=grid) if grid else ENVIRONMENT


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
        # 1.75 bits on arxiv seed 1 repeats a configuration of the base study: equal results show the partitions
        # reproduce. Qwen at 2 and 4 bits runs on seed 1 only (its k-means takes about 17 minutes per build).
        configs = [('arxiv', '1.75', 1)] + [('arxiv', bits, seed) for bits in BITS for seed in SEEDS] + \
                  [('qwen', bits, 1) for bits in QWEN_BITS]
        for dataset, bits, seed in configs:
            data = BASE / 'data' / dataset
            if True:
                tag = f'{dataset}-b{bits}-s{seed}'
                indexes = ROOT / 'indexes' / tag
                indexes.mkdir(parents=True, exist_ok=True)
                results = ROOT / 'results' / tag
                step(f'{tag}/build', [BENCH, 'build', data, seed, indexes, bits, 'tieredq'], BUILD_CORES)
                shutil.copy(indexes / 'build.json', results / 'build' / 'build.json')
                step(f'{tag}/parity', [BENCH, 'parity', data, indexes, results / 'parity' / 'parity.json'], serve_cores)
                for kind, name in (('tieredq', 'tieredq.tq'), ('plugin', 'plugin.tqf')):
                    label = name.rsplit('.', 1)[0]
                    step(f'{tag}/serve-{label}', [BENCH, 'serve', data, indexes, kind, name,
                                                  results / f'serve-{label}' / 'serve.json'], serve_cores)
                shutil.rmtree(indexes)
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
