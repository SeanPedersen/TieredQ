"""#60 fair build times: each method's full build from raw rows in RAM, 32 threads, under the shared lock.

1. venv: TieredQ from SOURCE_COMMIT (source/tieredq) and faiss-cpu into .venv-full.
2. TieredQ as shipped (full_build.py tieredq) at 1.25, 2, 2.5, 3 and 4 bits per dimension.
3. SAQ: preprocessing with Faiss (full_build.py saq-prep, 'fast' and 'upstream' k-means), then saq_ivf build on the
   'fast' lists with that PCA (PCA_DIR) at 1-4 bits, then saq_ivf serve (recall check, single core). Indexes removed.
Resumable per step (exit.json)."""
import datetime, fcntl, json, os, shutil, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
DATA = EXPERIMENTS / 'TieredQ-issue55-faiss-20261116/data/arxiv'
PYTHON = ROOT / '.venv-full/bin/python'
TIEREDQ_BITS = ('1.25', '2', '2.5', '3', '4')
SAQ_BITS = ('1', '2', '3', '4')
SEARCH_CPU, SEARCH_SIBLING = 2, 20
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (SEARCH_CPU, SEARCH_SIBLING, 19, 35))
ENVIRONMENT = dict(os.environ, QUERY_CORE=str(SEARCH_CPU), OMP_NUM_THREADS='32', OPENBLAS_NUM_THREADS='1',
                   OMP_WAIT_POLICY='PASSIVE', OMP_PROC_BIND='false', OMP_DYNAMIC='false')
VENV = (f"set -e; cd {ROOT}; [ -x .venv-full/bin/python ] || uv venv -q --python 3.12 .venv-full; "
        f"CMAKE_BUILD_PARALLEL_LEVEL=32 uv pip install -q --python .venv-full/bin/python numpy faiss-cpu source/tieredq")


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def step(name, command, cores, **env):
    out = ROOT / 'full' / name
    if (out / 'exit.json').exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    status('running', step=name)
    started = time.time()
    with (out / 'run.log').open('w') as log:
        code = subprocess.run(['taskset', '-c', cores, *map(str, command)], stdout=log, stderr=subprocess.STDOUT,
                              env=dict(ENVIRONMENT, **env)).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
    if code:
        raise RuntimeError(f'{name} failed: {code}')
    return out


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        step('venv', ['zsh', '-c', VENV], BUILD_CORES)
        for bits in TIEREDQ_BITS:
            step(f'tieredq-b{bits}', [PYTHON, ROOT / 'full_build.py', 'tieredq', DATA, bits, ROOT / 'full' / f'tieredq-b{bits}'],
                 BUILD_CORES)
        for kmeans in ('fast', 'upstream'):
            step(f'saq-prep-{kmeans}', [PYTHON, ROOT / 'full_build.py', 'saq-prep', DATA, kmeans,
                                        ROOT / 'full' / f'saq-prep-{kmeans}'], BUILD_CORES)
        prep = ROOT / 'full/saq-prep-fast'
        serve_cores = f'{BUILD_CORES},{SEARCH_CPU}'
        for bits in SAQ_BITS:
            index = ROOT / 'indexes' / f'full-saq{bits}'
            index.mkdir(parents=True, exist_ok=True)
            built = step(f'saq{bits}-build', [ROOT / 'build/saq_ivf', 'build', DATA, prep / 'centroids.f32',
                                              prep / 'assignment.u32', bits, index], BUILD_CORES, PCA_DIR=str(prep))
            if (index / 'build.json').exists():
                shutil.copy(index / 'build.json', built / 'build.json')
            served = ROOT / 'full' / f'saq{bits}-serve'
            step(f'saq{bits}-serve', [ROOT / 'build/saq_ivf', 'serve', DATA, index, served / 'serve.json'], serve_cores)
            shutil.rmtree(index)
        for name in ('centroids.f32', 'assignment.u32'):
            for kmeans in ('fast', 'upstream'):
                (ROOT / 'full' / f'saq-prep-{kmeans}' / name).unlink(missing_ok=True)
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'full.exitcode').write_text(f'{exit_code}\n')
