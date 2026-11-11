"""#53 packed lists A/B. A is 86f003d plus the heap-retention fix (padded lists, payload <= 5); B adds packed lists
(payload 6). Both build the same flat IVF on VIBE arxiv (1.75 bits/dim, 2,048 lists) and on Qwen/MS MARCO 8.84M (1 bit/dim,
automatic 64K lists with the router), so partitions and codes are identical and only the layout differs. Each saved
index is then served in ABBA order (A, B, B, A) by fresh processes: single queries alone on CPU 2 and 32-thread batches,
k = 10 and 100. compare.py checks identical ids and scores and summarizes QPS and storage. The C++ and Python tests of
the packed source run first, with the new fixture. Under the shared lock;
resumable per step (exit.json). Builds the packed package from source-packed/ into .venv-packed first, and generates
the padded-layout test fixture with A's predecessor (86f003d)."""
import datetime, fcntl, json, os, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
PYTHON = {'A': str(EXPERIMENTS / 'TieredQ-issue43-heapretention-20261111/.venv-fix/bin/python'),
          'B': str(ROOT / '.venv-packed/bin/python'),
          'main': str(EXPERIMENTS / 'TieredQ-issue41-nestedram-20261111/.venv/bin/python')}
SEARCH_CPU, SEARCH_SIBLING = 2, 20
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (SEARCH_CPU, SEARCH_SIBLING, 19, 35))
DATASETS = ('arxiv', 'qwen')
ORDER = ('A', 'B', 'B', 'A')
# The packed source with the new fixture, through the shared test workspace (C++ tests, then the Python suite).
TESTS = (f"set -e; cp {ROOT}/fixture/* {ROOT}/source-packed/tieredq/tests/fixtures/; cd {EXPERIMENTS}/tieredq-tests; "
         f"for part in src include tests bindings python README.md CMakeLists.txt pyproject.toml cmake; do "
         f"rsync -a {ROOT}/source-packed/tieredq/$part .; done; cmake --build build -j 32; ctest --test-dir build --output-on-failure; "
         f"uv run --group dev python -m pytest tests/python -q")
ENVIRONMENT = dict(os.environ, BUILD_CORES=BUILD_CORES, QUERY_CORE=str(SEARCH_CPU), OMP_NUM_THREADS='32',
                   OPENBLAS_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE', OMP_PROC_BIND='false', OMP_DYNAMIC='false')


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def step(name, command, cores):
    out = ROOT / 'results' / name
    if (out / 'exit.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    status('running', step=name)
    started = time.time()
    with (out / 'run.log').open('w') as log:
        code = subprocess.run(['taskset', '-c', cores, *command], stdout=log, stderr=subprocess.STDOUT, env=ENVIRONMENT).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
    if code:
        raise RuntimeError(f'{name} failed: {code}')


def build_packed_venv():
    if (ROOT / '.venv-packed/bin/python').exists():
        return
    with (ROOT / 'build-packed.log').open('w') as log:
        subprocess.run(['uv', 'venv', '-q', '--python', '3.12', str(ROOT / '.venv-packed')], check=True, stdout=log, stderr=log)
        subprocess.run(['uv', 'pip', 'install', '-q', '--python', PYTHON['B'], str(ROOT / 'source-packed/tieredq'), 'h5py'],
                       check=True, stdout=log, stderr=log, env=dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL='32'))


status('waiting_for_heap_fix_study')  # A's venv comes from that study
while not (EXPERIMENTS / 'TieredQ-issue43-heapretention-20261111/job-fix.exitcode').exists():
    time.sleep(30)
status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (ROOT / 'fixture').mkdir(exist_ok=True)
        step('fixture', [PYTHON['main'], str(ROOT / 'make_padded_fixture.py'), str(ROOT / 'fixture')], BUILD_CORES)
        step('tests', ['zsh', '-c', TESTS], BUILD_CORES)
        status('building_packed_venv'); build_packed_venv()
        (ROOT / 'indexes').mkdir(exist_ok=True)
        for dataset in DATASETS:
            for variant in ('A', 'B'):
                step(f'{dataset}-build-{variant}', [PYTHON[variant], str(ROOT / 'ab_index.py'), 'build', dataset,
                                                   str(ROOT / 'indexes' / f'{dataset}-{variant}.tq')], BUILD_CORES)
            for position, variant in enumerate(ORDER):
                name = f'{dataset}-serve-{position + 1}{variant}'
                step(name, [PYTHON[variant], str(ROOT / 'ab_index.py'), 'serve', dataset, str(ROOT / 'indexes' / f'{dataset}-{variant}.tq'),
                            str(ROOT / 'results' / name / 'serve.json')], f'{BUILD_CORES},{SEARCH_CPU}')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
