"""Heap retention after build (#43 flat, #41 nested). heap_probe.py on the full VIBE arxiv base, once per glibc setting:
- default: as shipped.
- fixed-thresholds: GLIBC_TUNABLES pins the trim and mmap thresholds at glibc's defaults, which turns off their dynamic
  growth after large frees. If the thread arenas then disappear, their 2.3 MiB is free top chunk that malloc_trim
  cannot return (it trims only the main arena's top), not live data.
- one-arena: MALLOC_ARENA_MAX=1, every thread allocates from the main arena, which malloc_trim fully trims.
Flat at the #43 configuration (1.75 bits/dim, 2,048 lists) and the nested default (+ write_tier). Diagnostic only:
the settings are process-wide and not a library change. Resumable per model (exit.json), under the shared lock."""
import datetime, fcntl, json, os, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
DATASET = EXPERIMENTS / 'TieredQ-vibe-arxiv-20261106/data/arxiv-nomic-768-normalized.hdf5'
PYTHON = str(EXPERIMENTS / 'TieredQ-issue41-nestedram-20261111/.venv/bin/python')  # TieredQ 86f003d
ROWS = 1344643
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (2, 20, 19, 35))
GLIBC = {'default': {}, 'fixed-thresholds': dict(GLIBC_TUNABLES='glibc.malloc.trim_threshold=131072:glibc.malloc.mmap_threshold=131072'),
         'one-arena': dict(MALLOC_ARENA_MAX='1')}
MODELS = [(f'{kind}-{setting}', kind, lists, setting) for kind, lists in (('flat', 2048), ('nested', None))
          for setting in GLIBC]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def run(name, kind, lists, setting):
    out = ROOT / 'results' / name
    if (out / 'exit.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    command = ['taskset', '-c', BUILD_CORES, PYTHON, str(ROOT / 'heap_probe.py'), str(DATASET), str(out / 'probe.json'),
               kind, str(ROWS), *([str(lists)] if lists else [])]
    environment = dict(os.environ, BUILD_CORES=BUILD_CORES, TIER_DIR=str(ROOT / 'ssd'), OMP_NUM_THREADS='32',
                       OPENBLAS_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE', **GLIBC[setting])
    started = time.time()
    with (out / 'run.log').open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment).returncode
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
        (ROOT / 'ssd').mkdir(exist_ok=True)
        for position, model in enumerate(MODELS):
            status('measuring', model=model[0], position=f'{position + 1}/{len(MODELS)}')
            run(*model)
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
