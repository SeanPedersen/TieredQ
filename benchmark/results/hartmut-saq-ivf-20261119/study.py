"""faiss#5940: SAQ (arxiv:2509.12086) IVF as a reference next to the #55 arxiv results (TieredQ is not rerun).

1. partition: the #55 acceptance binary (bisect build, source 3623bea) rebuilds arxiv partition seed 1 (Faiss k-means,
   2,048 lists) with TieredQIVFIndex only; export_assignment writes TieredQ's row assignment.
2. Per SAQ bit width: saq_ivf build on those lists and that assignment (32 threads), then saq_ivf serve in a fresh
   process (RAM, then one thread on CPU 2, recall@100 over 1,000 queries per nprobe 8-256). Indexes are removed.
Under the shared lock; resumable per step (exit.json)."""
import datetime, fcntl, json, os, shutil, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
DATA = EXPERIMENTS / 'TieredQ-issue55-faiss-20261116/data/arxiv'
ACCEPTANCE = EXPERIMENTS / 'TieredQ-issue55-bisect-20261116/plugin-build/faiss_tieredq_acceptance'
SEED = 1
BITS = ('1', '2', '3', '4')
SEARCH_CPU, SEARCH_SIBLING = 2, 20
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (SEARCH_CPU, SEARCH_SIBLING, 19, 35))
ENVIRONMENT = dict(os.environ, QUERY_CORE=str(SEARCH_CPU), OMP_NUM_THREADS='32', OPENBLAS_NUM_THREADS='1',
                   OMP_WAIT_POLICY='PASSIVE', OMP_PROC_BIND='false', OMP_DYNAMIC='false')


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def host():
    load = open('/proc/loadavg').read().split()[:3]
    top = subprocess.run(['ps', '-eo', 'pcpu,comm', '--sort=-pcpu'], capture_output=True, text=True).stdout.splitlines()[1:6]
    return dict(loadavg=load, top=[line.strip() for line in top])


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
                              env=ENVIRONMENT).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
    if code:
        raise RuntimeError(f'{name} failed: {code}')
    return out


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        partition = ROOT / 'partition'
        partition.mkdir(exist_ok=True)
        step('partition', [ACCEPTANCE, 'build', DATA, SEED, partition, '1.75', 'tieredq'], BUILD_CORES)
        step('assignment', [ROOT / 'build/export_assignment', partition / 'tieredq.tq', partition / 'assignment.u32'],
             BUILD_CORES)
        serve_cores = f'{BUILD_CORES},{SEARCH_CPU}'
        for bits in BITS:
            index = ROOT / 'indexes' / f'saq{bits}'
            index.mkdir(parents=True, exist_ok=True)
            built = step(f'saq{bits}/build', [ROOT / 'build/saq_ivf', 'build', DATA, partition / 'centroids.f32',
                                              partition / 'assignment.u32', bits, index], BUILD_CORES)
            if (index / 'build.json').exists():
                shutil.copy(index / 'build.json', built / 'build.json')
            served = ROOT / 'results' / f'saq{bits}/serve'
            step(f'saq{bits}/serve', [ROOT / 'build/saq_ivf', 'serve', DATA, index, served / 'serve.json'], serve_cores)
            (served / 'files.json').write_text(json.dumps({p.name: p.stat().st_size for p in index.iterdir()}) + '\n')
            shutil.rmtree(index)
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
