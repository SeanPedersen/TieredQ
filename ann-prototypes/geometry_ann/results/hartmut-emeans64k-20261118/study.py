"""Issue #45 final bench: e-means coarse training (TieredQCoarseTraining.emeans_epochs) against the automatic fast
Lloyd recipe in #52's shipping low-RAM build: TieredQNestedIVFIndex.build defaults (1-bit prefix, 64K lists, router,
8-bit fine suffix tier on the NVMe SSD, 40 candidates) on Qwen/MS MARCO 8.84M x 1024 from the memory-mapped SQ8
store, 32 build threads; the 750 held-out queries alone on pinned CPU 2 with one OpenMP thread. Two partition seeds
(the seed alone moves recall@10 by up to 0.0045 here).
- lloyd: the automatic fast recipe (4 iterations on 40 rows per list, 256 x 16 restricted assignment).
- emeans-s64-e2.5: e-means for 2.5 passes over 64 rows per list (the same row passes), same restricted assignment.
- emeans-s96-e2.5: 96 rows per list (a 26 GB FP32 sample; 256 per list would be all 8.84M rows, 36 GB).
First, ../TieredQ-issue45-emeans-ivfflat-20261118's 1M TieredQIVFIndex check on seed 1 (ivf_lists_check.py).
Runs under the shared benchmark lock; resumable per step (exit.json); each model's index and tier files are removed
once it is measured."""
import datetime, fcntl, json, os, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
ONE_MILLION = EXPERIMENTS / 'TieredQ-issue45-emeans-ivfflat-20261118'
THREADS, SEARCH_CPU, QUERIES = 32, 2, 750
NPROBES = '96,128,192,256,512,1024'
SEEDS = (0, 7)
PYTHON = ['uv', 'run', '--project', str(ROOT / 'source/tieredq'), '--no-dev', 'python']
FAST = dict(iterations=4, assignment_supers=256, assignment_probes=16)
MODELS = [dict(name='lloyd', training={}),
          dict(name='emeans-s64-e2.5', training=dict(FAST, sample_rows_per_list=64, emeans_epochs=2.5)),
          dict(name='emeans-s96-e2.5', training=dict(FAST, sample_rows_per_list=96, emeans_epochs=2.5))]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def run(out, command, environment, cwd=None):
    if (out / 'exit.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (out / 'run.log').open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=cwd,
                              env=dict(os.environ, **environment)).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
    if code:
        raise RuntimeError(f'{out.name} failed: {code}')


def one_million_seed1():
    source = ONE_MILLION / 'source'
    run(ROOT / 'results/ivf1m-sync', ['uv', 'sync', '--reinstall-package', 'tieredq'], {}, cwd=source)
    queries = f"{QWEN / 'data/calibration_queries.fvecs'},{QWEN / 'data/test_queries.fvecs'}"
    for lists, probes in ((8192, '16,32,64,128,256'), (2048, '8,16,32,64,128')):
        run(ROOT / 'results' / f'ivf1m-L{lists}-seed1',
            ['taskset', '-c', '0-7', 'uv', 'run', 'python', 'ivf_lists_check.py', '../data/msmarco1m.npy',
             '../results/msmarco1m-queries/query_truth.npz', str(ROOT / 'results/ivf1m-seed1.jsonl'), '--queries',
             queries, '--lists', str(lists), '--probes', probes, '--seed', '1',
             '--only', 'lloyd-auto,emeans-s64-e2.5,emeans-s256-e2.5'],
            {'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'}, cwd=source)


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status('building')
        with (ROOT / 'build.log').open('w') as log:  # builds the package (Release) into the source tree's venv
            subprocess.run([*PYTHON, '-c', 'import tieredq'], stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL='16'))
        status('measuring', model='ivf1m-seed1')
        one_million_seed1()
        (ROOT / 'ssd').mkdir(exist_ok=True)
        for seed in SEEDS:
            for model in MODELS:
                name = f"{model['name']}-seed{seed}"
                results = ROOT / 'results' / name
                index, tier = ROOT / 'ssd' / f'{name}.tq', ROOT / 'ssd' / f'{name}.rows'
                if (results / 'serve/exit.json').exists():
                    continue
                status('measuring', model=name, step='build')
                run(results / 'build', [*PYTHON, str(ROOT / 'build.py'), str(QWEN / 'base.i8'), str(QWEN / 'scales.f32'),
                                        str(QWEN / 'offsets.f32'), '1024', str(index), str(tier),
                                        str(results / 'build/build.json'), str(seed), json.dumps(model['training'])],
                    {'OMP_NUM_THREADS': str(THREADS)})
                status('measuring', model=name, step='serve')
                run(results / 'serve', ['taskset', '-c', str(SEARCH_CPU), *PYTHON, str(ROOT / 'serve.py'), str(index),
                                        str(tier), str(QWEN / 'data/test_queries.fvecs'),
                                        str(QWEN / 'data/test_truth.ivecs'), str(QUERIES), NPROBES,
                                        str(results / 'serve/run.csv')],
                    {'OMP_NUM_THREADS': '1'})
                sizes = {path.name: path.stat().st_size for path in (index, tier)}
                (results / 'build/files.json').write_text(json.dumps(sizes) + '\n')
                for path in (index, tier):
                    path.unlink()
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
