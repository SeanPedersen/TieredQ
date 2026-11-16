"""#55 step 4: IndexIVFTieredQ acceptance on hartmut. Source: TieredQ at SOURCE_COMMIT under source/tieredq (the plugin
in its faiss-tieredq/), against Faiss v1.16.0 built from source here (AVX-512 kernels, OpenBLAS).

1. faiss: builds and installs Faiss v1.16.0 (C++ only, FAISS_OPT_LEVEL avx512).
2. plugin: builds the plugin and its acceptance benchmark against faiss_avx512, runs the C++ tests, installs the Python
   module into .venv and runs its tests.
3. data: prepare.py writes data/arxiv and data/qwen.
4. Per dataset and partition seed (arxiv 1, 2, 3 with 2,048 lists; qwen 1, 2 with 64K lists):
   build (32 threads): Faiss k-means lists, TieredQIVFIndex on them, IndexIVFTieredQ on TieredQ's codec and
   assignment, IndexIVFRaBitQ 1/2/4 bits and IndexIVFPQFastScan at 1 and 2 bits per dimension on the same lists,
   assignment and training rows; parity (TieredQ exact ranking against the plugin's flat quantizer); then every index
   served by a fresh process: RAM, then alone on CPU 2 single-query QPS / p50 / recall per nprobe. The seed's
   indexes are removed afterwards.
Under the shared lock; resumable per step (exit.json)."""
import datetime, fcntl, json, os, shutil, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
SOURCE = ROOT / 'source/tieredq'
SEARCH_CPU, SEARCH_SIBLING = 2, 20
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (SEARCH_CPU, SEARCH_SIBLING, 19, 35))
FAISS_TAG = 'v1.16.0'
SEEDS = {'arxiv': (1, 2, 3), 'qwen': (1, 2)}
INDEXES = [('tieredq', 'tieredq.tq'), ('plugin', 'plugin.tqf')] + [
    ('faiss', f'{name}.faiss') for name in ('rabitq1', 'rabitq2', 'rabitq4')]
PQ = {'arxiv': ('pq4fs-m192', 'pq4fs-m384'), 'qwen': ('pq4fs-m256', 'pq4fs-m512')}
BENCH = ROOT / 'plugin-build/faiss_tieredq_acceptance'
ENVIRONMENT = dict(os.environ, QUERY_CORE=str(SEARCH_CPU), OMP_NUM_THREADS='32', OPENBLAS_NUM_THREADS='1',
                   OMP_WAIT_POLICY='PASSIVE', OMP_PROC_BIND='false', OMP_DYNAMIC='false')
FAISS = (f"set -e; cd {ROOT}; [ -d faiss-src ] || git clone -q --depth 1 --branch {FAISS_TAG} "
         f"https://github.com/facebookresearch/faiss.git faiss-src; cmake -S faiss-src -B faiss-build -DCMAKE_BUILD_TYPE=Release "
         f"-DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=OFF -DBUILD_TESTING=OFF -DFAISS_OPT_LEVEL=avx512 "
         f"-DCMAKE_INSTALL_PREFIX={ROOT}/faiss-install; cmake --build faiss-build -j 32; cmake --install faiss-build")
PLUGIN = (f"set -e; cd {ROOT}; cmake -S {SOURCE}/faiss-tieredq -B plugin-build -DCMAKE_BUILD_TYPE=Release "
          f"-DCMAKE_PREFIX_PATH={ROOT}/faiss-install -DFAISS_TIEREDQ_FAISS_TARGET=faiss_avx512 -DFAISS_TIEREDQ_BENCHMARKS=ON; "
          f"cmake --build plugin-build -j 32; ctest --test-dir plugin-build --output-on-failure; "
          f"[ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv; "
          f"CMAKE_PREFIX_PATH={ROOT}/faiss-install CMAKE_BUILD_PARALLEL_LEVEL=32 uv pip install -q --python .venv/bin/python "
          f"--reinstall {SOURCE}/faiss-tieredq pytest h5py; cd {SOURCE}/faiss-tieredq; {ROOT}/.venv/bin/python -m pytest tests/python -q")


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
        step('faiss', ['zsh', '-c', FAISS], BUILD_CORES)
        step('plugin', ['zsh', '-c', PLUGIN], BUILD_CORES)
        step('data', [ROOT / '.venv/bin/python', ROOT / 'prepare.py', ROOT / 'data'], BUILD_CORES)
        serve_cores = f'{BUILD_CORES},{SEARCH_CPU}'
        for dataset, seeds in SEEDS.items():
            data = ROOT / 'data' / dataset
            for seed in seeds:
                tag = f'{dataset}-s{seed}'
                indexes = ROOT / 'indexes' / tag
                indexes.mkdir(parents=True, exist_ok=True)
                results = ROOT / 'results' / tag
                step(f'{tag}/build', [BENCH, 'build', data, seed, indexes], BUILD_CORES)
                shutil.copy(indexes / 'build.json', results / 'build' / 'build.json')
                step(f'{tag}/parity', [BENCH, 'parity', data, indexes, results / 'parity' / 'parity.json'], serve_cores)
                for kind, name in INDEXES + [('faiss', f'{name}.faiss') for name in PQ[dataset]]:
                    label = name.rsplit('.', 1)[0]
                    step(f'{tag}/serve-{label}', [BENCH, 'serve', data, indexes, kind, name,
                                                  results / f'serve-{label}' / 'serve.json'], serve_cores)
                sizes = {path.name: path.stat().st_size for path in indexes.iterdir()}
                (results / 'files.json').write_text(json.dumps(sizes, indent=2) + '\n')
                shutil.rmtree(indexes)
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
