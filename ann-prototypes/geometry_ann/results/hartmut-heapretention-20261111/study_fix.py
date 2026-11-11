"""Heap retention, second round: the fix (trim after build's buffers are freed; refinement encoding scratch allocated
by the calling thread) against 86f003d, in cold processes (as VIBE measures) and warm ones (after a warm-up build), on
the same flat and nested arxiv builds as study.py. Then heaptrack over the fixed warm builds, ended right after the
last snapshot, so its leaked allocations are the heap still live after build: what storage_bytes might count.
The fixed package is built from source-fix/ into .venv-fix (under the lock). Resumable per model (exit.json)."""
import datetime, fcntl, json, os, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
DATASET = EXPERIMENTS / 'TieredQ-vibe-arxiv-20261106/data/arxiv-nomic-768-normalized.hdf5'
PYTHON = {'main': str(EXPERIMENTS / 'TieredQ-issue41-nestedram-20261111/.venv/bin/python'),  # TieredQ 86f003d
          'fix': str(ROOT / '.venv-fix/bin/python')}
ROWS = 1344643
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (2, 20, 19, 35))
LISTS = {'flat': 2048, 'nested': None}
MODELS = ([(f'{kind}-{build}-cold', kind, build, dict(WARMUP='0'), False) for build in ('main', 'fix') for kind in LISTS] +
          [(f'{kind}-fix-warm', kind, 'fix', {}, False) for kind in LISTS] +
          [(f'{kind}-fix-heaptrack', kind, 'fix', dict(EXIT_AFTER='1'), True) for kind in LISTS])


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def build_fix_venv():
    if (ROOT / '.venv-fix/bin/python').exists():
        return
    with (ROOT / 'build-fix.log').open('w') as log:
        subprocess.run(['uv', 'venv', '-q', '--python', '3.12', str(ROOT / '.venv-fix')], check=True, stdout=log, stderr=log)
        subprocess.run(['uv', 'pip', 'install', '-q', '--python', PYTHON['fix'], str(ROOT / 'source-fix/tieredq'), 'h5py', 'psutil'],
                       check=True, stdout=log, stderr=log, env=dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL='32'))


def run(name, kind, build, extra_env, heaptrack):
    out = ROOT / 'results' / name
    if (out / 'exit.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    probe = [PYTHON[build], str(ROOT / 'heap_probe.py'), str(DATASET), str(out / 'probe.json'), kind, str(ROWS),
             *([str(LISTS[kind])] if LISTS[kind] else [])]
    tracer = ['heaptrack', '-o', str(out / 'heaptrack')] if heaptrack else []
    environment = dict(os.environ, BUILD_CORES=BUILD_CORES, TIER_DIR=str(ROOT / 'ssd'), OMP_NUM_THREADS='32',
                       OPENBLAS_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE', **extra_env)
    started = time.time()
    with (out / 'run.log').open('w') as log:
        code = subprocess.run(['taskset', '-c', BUILD_CORES, *tracer, *probe], stdout=log, stderr=subprocess.STDOUT,
                              env=environment).returncode
    for tier in (ROOT / 'ssd').glob('probe-tier-*.rows'):
        tier.unlink()
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
    if code:
        raise RuntimeError(f'{name} failed: {code}')


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status('building_fix_venv'); build_fix_venv()
        (ROOT / 'ssd').mkdir(exist_ok=True)
        for position, model in enumerate(MODELS):
            status('measuring', model=model[0], position=f'{position + 1}/{len(MODELS)}')
            run(*model)
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job-fix.exitcode').write_text(f'{exit_code}\n')
