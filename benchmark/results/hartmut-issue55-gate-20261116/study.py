"""#55 non-regression gate for the step-1 library change. A is aae87c7 (before), B is fba7ef5 (fixed-capacity tiles,
public TieredQ::map_scores). Both are snapshotted under source-A/ and source-B/ and installed into .venv-A / .venv-B.

1. tests: B's C++ tests (ctest) and Python suite on hartmut's AVX-512, through the shared tieredq-tests workspace.
2. kernel: kernel_bench (whole-run block scans on both sides, tiles on B) in ABBA order alone on CPU 2.
3. Per dataset (arxiv, then qwen):
   - flat builds in ABBA order (A1, B2, B3, A4): build time, storage_bytes, peak RSS, and the saved files' sha256;
   - nested builds A then B (prefix index and SSD tier), with their sha256;
   - A1's flat index and A's nested index served in ABBA order (A, B, B, A) by fresh processes: single-query QPS and
     p50 / p95 alone on CPU 2, one 32-thread batch, anonymous RSS, and ids / scores for the identity check;
   - when a B build's files differ from A's, B's own index is served once more for a result-level check.
summarize.py writes the report. Under the shared lock; resumable per step (exit.json); indexes are removed at the end."""
import datetime, fcntl, hashlib, json, os, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
PYTHON = {v: str(ROOT / f'.venv-{v}/bin/python') for v in 'AB'}
SEARCH_CPU, SEARCH_SIBLING = 2, 20
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (SEARCH_CPU, SEARCH_SIBLING, 19, 35))
DATASETS = ('arxiv', 'qwen')
ABBA = ('A', 'B', 'B', 'A')
KERNELS = (('768', '175', '1048576', '100'), ('1024', '100', '1048576', '100'))
TESTS = (f"set -e; cd {EXPERIMENTS}/tieredq-tests; "
         f"for part in src include tests examples bindings python README.md CMakeLists.txt pyproject.toml cmake; do "
         f"rsync -a --delete {ROOT}/source-B/tieredq/$part .; done; cmake --build build -j 32; "
         f"ctest --test-dir build --output-on-failure; uv run --group dev python -m pytest tests/python -q")
ENVIRONMENT = dict(os.environ, BUILD_CORES=BUILD_CORES, QUERY_CORE=str(SEARCH_CPU), OMP_NUM_THREADS='32',
                   OPENBLAS_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE', OMP_PROC_BIND='false', OMP_DYNAMIC='false')


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def host():
    load = open('/proc/loadavg').read().split()[:3]
    pressure = open('/proc/pressure/io').read().split('\n')[0]
    top = subprocess.run(['ps', '-eo', 'pcpu,comm', '--sort=-pcpu'], capture_output=True, text=True).stdout.splitlines()[1:6]
    return dict(loadavg=load, io_pressure=pressure, top=[line.strip() for line in top])


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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record_hashes(out, paths):
    target = out / 'files.json'
    if not target.exists():
        target.write_text(json.dumps({p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in paths}, indent=2) + '\n')
    return {name: value['sha256'] for name, value in json.loads(target.read_text()).items()}


def build_venv(variant):
    if Path(PYTHON[variant]).exists():
        return
    with (ROOT / f'build-venv-{variant}.log').open('w') as log:
        subprocess.run(['uv', 'venv', '-q', '--python', '3.12', str(ROOT / f'.venv-{variant}')], check=True, stdout=log, stderr=log)
        subprocess.run(['uv', 'pip', 'install', '-q', '--python', PYTHON[variant], str(ROOT / f'source-{variant}/tieredq'), 'h5py'],
                       check=True, stdout=log, stderr=log, env=dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL='32'))


def build_kernel_bench(variant):
    tiles = 'ON' if variant == 'B' else 'OFF'
    return ['zsh', '-c', f'cmake -S {ROOT}/kernel -B {ROOT}/kernel/build-{variant} -DCMAKE_BUILD_TYPE=Release '
                         f'-DTIEREDQ_SOURCE_DIR={ROOT}/source-{variant}/tieredq -DTIEREDQ_TILES={tiles} '
                         f'&& cmake --build {ROOT}/kernel/build-{variant} -j 32']


def gate(dataset, indexes):
    flat = {}
    for position, variant in enumerate(ABBA):
        tag = f'{position + 1}{variant}'
        index = indexes / f'{dataset}-flat-{tag}.tq'
        out = step(f'{dataset}-flat-build-{tag}', [PYTHON[variant], ROOT / 'gate.py', 'build', 'flat', dataset, index,
                                                    ROOT / 'results' / f'{dataset}-flat-build-{tag}' / 'build.json'], BUILD_CORES)
        flat[tag] = record_hashes(out, [index])[index.name]
    nested = {}
    for variant in 'AB':
        index, tier = indexes / f'{dataset}-nested-{variant}.tq', indexes / f'{dataset}-nested-{variant}.rows'
        out = step(f'{dataset}-nested-build-{variant}', [PYTHON[variant], ROOT / 'gate.py', 'build', 'nested', dataset, index, tier,
                                                          ROOT / 'results' / f'{dataset}-nested-build-{variant}' / 'build.json'], BUILD_CORES)
        nested[variant] = tuple(record_hashes(out, [index, tier]).values())
    serve_cores = f'{BUILD_CORES},{SEARCH_CPU}'
    for position, variant in enumerate(ABBA):
        tag = f'{position + 1}{variant}'
        step(f'{dataset}-flat-serve-{tag}', [PYTHON[variant], ROOT / 'gate.py', 'serve', 'flat', dataset, indexes / f'{dataset}-flat-1A.tq',
                                             ROOT / 'results' / f'{dataset}-flat-serve-{tag}' / 'serve.json'], serve_cores)
        step(f'{dataset}-nested-serve-{tag}', [PYTHON[variant], ROOT / 'gate.py', 'serve', 'nested', dataset,
                                               indexes / f'{dataset}-nested-A.tq', indexes / f'{dataset}-nested-A.rows',
                                               ROOT / 'results' / f'{dataset}-nested-serve-{tag}' / 'serve.json'], serve_cores)
    if flat['2B'] != flat['1A']:
        step(f'{dataset}-flat-serve-own-B', [PYTHON['B'], ROOT / 'gate.py', 'serve', 'flat', dataset, indexes / f'{dataset}-flat-2B.tq',
                                             ROOT / 'results' / f'{dataset}-flat-serve-own-B' / 'serve.json'], serve_cores)
    if nested['B'] != nested['A']:
        step(f'{dataset}-nested-serve-own-B', [PYTHON['B'], ROOT / 'gate.py', 'serve', 'nested', dataset, indexes / f'{dataset}-nested-B.tq',
                                               indexes / f'{dataset}-nested-B.rows',
                                               ROOT / 'results' / f'{dataset}-nested-serve-own-B' / 'serve.json'], serve_cores)
    for path in indexes.glob(f'{dataset}-*'):
        path.unlink()


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        step('tests', ['zsh', '-c', TESTS], BUILD_CORES)
        for variant in 'AB':
            status('building_venv', variant=variant); build_venv(variant)
            step(f'kernel-build-{variant}', build_kernel_bench(variant), BUILD_CORES)
        for position, variant in enumerate(ABBA):
            for dimensions, bits, rows, queries in KERNELS:
                step(f'kernel-{dimensions}-{bits}-{position + 1}{variant}',
                     [ROOT / f'kernel/build-{variant}/kernel_bench', dimensions, bits, rows, queries], f'{BUILD_CORES},{SEARCH_CPU}')
        indexes = ROOT / 'indexes'; indexes.mkdir(exist_ok=True)
        for dataset in DATASETS:
            gate(dataset, indexes)
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
