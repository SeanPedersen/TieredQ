"""End-to-end proof (#52): the low-RAM configuration built and served through the Python API, on Qwen/MS MARCO
8.84M x 1024 from the memory-mapped SQ8 store, 32 build threads, the 750 held-out queries alone on pinned CPU 2 with
one OpenMP thread, tier on the NVMe SSD with direct reads.
- defaults: TieredQNestedIVFIndex.build(codes=, scales=, offsets=) with no other option (1-bit prefix, 40
  candidates); reference: the C++ onebit-64k run in hartmut-defaults-20261111.
- twobit: prefix_bits=2048 and candidates=20, the earlier winner; reference: hartmut-ssdtier-20261111.
Acceptance: recall@10 within 0.002 at matched nprobe, p50 within about 5%, build about 400 s or less, anonymous RSS at
the reference's. Runs under the shared benchmark lock; resumable per step (exit.json); each model's index and tier
files are removed once it is measured."""
import datetime, fcntl, json, os, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
THREADS, SEARCH_CPU, QUERIES = 32, 2, 750
NPROBES = '96,128,192,256,512,1024,2048,4096'
PYTHON = ['uv', 'run', '--project', str(ROOT / 'source/tieredq'), '--no-dev', 'python']
MODELS = [dict(name='defaults', build={}, candidates=[]),
          dict(name='twobit', build={'prefix_bits': 2048}, candidates=['20'])]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def run(out, command, environment):
    if (out / 'exit.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (out / 'run.log').open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, **environment)).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
    if code:
        raise RuntimeError(f'{out.name} failed: {code}')


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status('building')
        with (ROOT / 'build.log').open('w') as log:  # builds the package (Release) into the source tree's venv
            subprocess.run([*PYTHON, '-c', 'import tieredq'], stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL='16'))
        (ROOT / 'ssd').mkdir(exist_ok=True)
        for model in MODELS:
            results = ROOT / 'results' / model['name']
            index, tier = ROOT / 'ssd' / f"{model['name']}.tq", ROOT / 'ssd' / f"{model['name']}.rows"
            if (results / 'serve/exit.json').exists():
                continue
            status('measuring', model=model['name'], step='build')
            run(results / 'build', [*PYTHON, str(ROOT / 'build.py'), str(QWEN / 'base.i8'), str(QWEN / 'scales.f32'),
                                    str(QWEN / 'offsets.f32'), '1024', str(index), str(tier),
                                    str(results / 'build/build.json'), json.dumps(model['build'])],
                {'OMP_NUM_THREADS': str(THREADS)})
            status('measuring', model=model['name'], step='serve')
            run(results / 'serve', ['taskset', '-c', str(SEARCH_CPU), *PYTHON, str(ROOT / 'serve.py'), str(index), str(tier),
                                    str(QWEN / 'data/test_queries.fvecs'), str(QWEN / 'data/test_truth.ivecs'),
                                    str(QUERIES), NPROBES, str(results / 'serve/run.csv'), *model['candidates']],
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
