"""Quiet-host rerun of the Python 1-bit nested default on Qwen/MS MARCO 8.84M (#52 leftover). The
hartmut-pythondefaults-20261111 1-bit p50 was 7-14% above the C++ reference, under host load. Here the other hartmut
services are paused. One build through the Python API with defaults only (build.py, 32 threads, SQ8 memmap in, tier on
NVMe), on the current source (86f003d + heap fixes + packed lists; ../TieredQ-issue53-packedlists-20261111/.venv-packed).
The same saved index is then served in ABBA order (Python, C++, C++, Python) alone on pinned CPU 2 with one thread,
which separates binding overhead from host load without a partition-seed difference. Last, batch.py measures 32-thread
batch QPS of this SSD configuration (never measured before). Each step records the host's load and IO pressure when it
starts. Under the shared lock; resumable per step; the index and tier are removed at the end."""
import datetime, fcntl, json, os, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
SOURCE = EXPERIMENTS / 'TieredQ-issue53-packedlists-20261111/source-packed/tieredq'
PYTHON = str(EXPERIMENTS / 'TieredQ-issue53-packedlists-20261111/.venv-packed/bin/python')
SEARCH_CPU, SEARCH_SIBLING, QUERIES = 2, 20, 750
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (SEARCH_CPU, SEARCH_SIBLING, 19, 35))
NPROBES = '96,128,192,256,512,1024,2048,4096'
INDEX, TIER = ROOT / 'ssd/defaults.tq', ROOT / 'ssd/defaults.rows'
DATA = [str(QWEN / 'data/test_queries.fvecs'), str(QWEN / 'data/test_truth.ivecs'), str(QUERIES), NPROBES]
SERVERS = {'python': [PYTHON, str(ROOT / 'serve.py'), str(INDEX), str(TIER)],
           'cpp': [str(ROOT / 'build/serve_nested'), str(INDEX), str(TIER)]}


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def host():
    load = open('/proc/loadavg').read().split()[:3]
    pressure = open('/proc/pressure/io').read().split('\n')[0]
    top = subprocess.run(['ps', '-eo', 'pcpu,comm', '--sort=-pcpu'], capture_output=True, text=True).stdout.splitlines()[1:6]
    return dict(loadavg=load, io_pressure=pressure, top=[line.strip() for line in top])


def step(name, command, cores, threads):
    out = ROOT / 'results' / name
    if (out / 'exit.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    status('running', step=name)
    (out / 'host.json').write_text(json.dumps(host(), indent=2) + '\n')
    started = time.time()
    with (out / 'run.log').open('w') as log:
        code = subprocess.run(['taskset', '-c', cores, *[str(part).replace('{out}', str(out)) for part in command]],
                              stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS=str(threads))).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
    if code:
        raise RuntimeError(f'{name} failed: {code}')


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (ROOT / 'ssd').mkdir(exist_ok=True)
        step('driver', ['zsh', '-c', f'cmake -S {ROOT} -B {ROOT}/build -DCMAKE_BUILD_TYPE=Release -DTIEREDQ_SOURCE_DIR={SOURCE} '
                                     f'&& cmake --build {ROOT}/build -j 32 --target serve_nested'], BUILD_CORES, 32)
        step('build', [PYTHON, ROOT / 'build.py', QWEN / 'base.i8', QWEN / 'scales.f32', QWEN / 'offsets.f32', '1024',
                       INDEX, TIER, '{out}/build.json'], BUILD_CORES, 32)
        for position, server in enumerate(('python', 'cpp', 'cpp', 'python')):
            step(f'serve-{position + 1}-{server}', [*SERVERS[server], *DATA, '{out}/run.csv'], str(SEARCH_CPU), 1)
        step('batch', [PYTHON, ROOT / 'batch.py', INDEX, TIER, *DATA, '{out}/run.csv'], BUILD_CORES, 32)
        sizes = {path.name: path.stat().st_size for path in (INDEX, TIER)}
        (ROOT / 'results/build/files.json').write_text(json.dumps(sizes) + '\n')
        for path in (INDEX, TIER):
            path.unlink()
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
