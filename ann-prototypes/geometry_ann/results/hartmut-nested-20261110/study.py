"""TieredQ refinement tiers in place of the SQ8 re-rank (#49), for the RAM axis of #52, on Qwen/MS MARCO 8.84M x 1024.
Loads the router study's from-scratch 64K index (the shipping configuration), adds the router (expansion 64, at least
32 supers), and measures, on the 750 held-out queries alone on one pinned core: the exact SQ8 re-rank of the top 40/100
(1,024 B per vector, memory-mapped store) and TieredQNestedIVFIndex suffix tiers refining to 4/5/6/8 bits/dim total,
held in RAM, rescoring the top 40/100, at 5 nprobes. Serving RSS (anonymous and file-backed) is recorded per
configuration. Runs under the shared benchmark lock; resumable per model (exit.json)."""
import datetime, fcntl, json, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
INDEXES = EXPERIMENTS / 'TieredQ-issue44-router-20261110/indexes'
ROWS, THREADS, SEARCH_CPU = 8840823, 32, 2
QUERIES = f"heldout:{QWEN / 'data/test_queries.fvecs'}:{QWEN / 'data/test_truth.ivecs'}:750"
NPROBES = [96, 128, 256, 512, 1024]
MODELS = [dict(name='tiers-64k', index=INDEXES / 'scratch-64k.tq', totals='4,5,6,8', fine='0', candidates='40,100')]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def checked(args, log):
    with log.open('w') as out:
        subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, check=True)


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status('building')
        checked(['cmake', '-S', str(ROOT / 'source'), '-B', str(ROOT / 'build'), '-DCMAKE_BUILD_TYPE=Release',
                 f"-DTIEREDQ_SOURCE_DIR={ROOT / 'source/tieredq'}"], ROOT / 'configure.log')
        checked(['cmake', '--build', str(ROOT / 'build'), '-j', '16', '--target', 'nested_qwen'], ROOT / 'build.log')
        (ROOT / 'results').mkdir(exist_ok=True)
        for model in MODELS:
            out = ROOT / 'results' / model['name']; out.mkdir(exist_ok=True)
            if (out / 'exit.json').exists():
                continue
            status('measuring', model=model['name'])
            args = [str(ROOT / 'build/nested_qwen'), f"codes={QWEN / 'base.i8'}", f"scales={QWEN / 'scales.f32'}",
                    f"offsets={QWEN / 'offsets.f32'}", f'rows={ROWS}', f"index={model['index']}", f'threads={THREADS}',
                    f'search_cpu={SEARCH_CPU}', 'nprobes=' + ','.join(map(str, NPROBES)), f"candidates={model['candidates']}",
                    f"totals={model['totals']}", f"fine={model['fine']}", f'queries={QUERIES}',
                    f"out={out / 'run'}"]
            started = time.time()
            with (out / 'run.log').open('w') as log:
                code = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT).returncode
            (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
            if code:
                raise RuntimeError(f"{model['name']} failed: {code}")
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
