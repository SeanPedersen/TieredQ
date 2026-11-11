"""#19 §2 drift on VIBE arxiv: drift.py for holdout and random training samples, and with only the codec or only the lists on the holdout
sample, three seeds each, one fresh process per
model under the shared benchmark lock (32 build threads; recall only, so host load does not matter). Resumable per model."""
import datetime, fcntl, json, os, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
DATASET = EXPERIMENTS / 'TieredQ-vibe-arxiv-20261106/data/arxiv-nomic-768-normalized.hdf5'
PYTHON = str(EXPERIMENTS / 'TieredQ-issue53-packedlists-20261111/.venv-packed/bin/python')
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (2, 20, 19, 35))
MODELS = [(mode, seed) for seed in (0, 1, 2) for mode in ('holdout', 'random', 'codec-holdout', 'lists-holdout')]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for mode, seed in MODELS:
            out = ROOT / 'results' / f'{mode}-s{seed}'
            if (out / 'exit.json').exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            status('measuring', model=out.name)
            started = time.time()
            with (out / 'run.log').open('w') as log:
                code = subprocess.run(['taskset', '-c', BUILD_CORES, PYTHON, str(ROOT / 'drift.py'), str(DATASET),
                                       str(out / 'run.json'), mode, str(seed)], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                      env=dict(os.environ, BUILD_CORES=BUILD_CORES, OMP_NUM_THREADS='32',
                                               OPENBLAS_NUM_THREADS='1')).returncode
            (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
            if code:
                raise RuntimeError(f'{out.name} failed: {code}')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
