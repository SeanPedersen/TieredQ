"""Metric fix (#49, #52): the truth is exact squared L2 on unnormalized vectors, but the scratch indexes used an
inner-product codec, so the 2-bit scan and the TieredQ refinement tiers ranked by the wrong metric (6- and 8-bit tiers
plateaued at the same recall, 0.003 below the L2 SQ8 re-rank). Builds the 64K index from scratch with an L2 codec, same
recipe (4 Lloyd iterations on 40 rows per list, restricted assignment 256 x 16, router at expansion 64), saves it, then
measures the SQ8 re-rank against 5- and 6-bit TieredQ tiers in RAM at 5 nprobes, with serving RSS (nested_qwen), on the
750 held-out queries alone on one pinned core. Runs under the shared benchmark lock; resumable per step (exit.json)."""
import datetime, fcntl, json, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
ROWS, THREADS, SEARCH_CPU = 8840823, 32, 2
QUERIES = f"heldout:{QWEN / 'data/test_queries.fvecs'}:{QWEN / 'data/test_truth.ivecs'}:750"
STORE = [f"codes={QWEN / 'base.i8'}", f"scales={QWEN / 'scales.f32'}", f"offsets={QWEN / 'offsets.f32'}", f'rows={ROWS}']
NPROBES = '96,128,256,512,1024'
INDEX = ROOT / 'indexes/scratch-64k-l2.tq'
STEPS = [('build-64k-l2', ['tieredq_ivf_qwen', *STORE, 'lists=65536', 'metric=l2', 'train_iterations=4', 'rows_per_list=40',
                           'assignment_supers=256', 'assignment_probes=16', f'index={INDEX}', f'threads={THREADS}',
                           f'search_cpu={SEARCH_CPU}', 'code_bits=2048', 'max_scan_bits=4', 'nprobes=128', 'candidates=40',
                           'exact=0', 'expansions=64', 'lists_per_super=128', 'rerank=256', 'router_bits=2',
                           'min_super_probes=32', f'queries={QUERIES}']),
         ('tiers-64k-l2', ['nested_qwen', *STORE, f'index={INDEX}', f'threads={THREADS}', f'search_cpu={SEARCH_CPU}',
                           f'nprobes={NPROBES}', 'candidates=40', 'totals=5,6', 'fine=0', f'queries={QUERIES}'])]


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
        checked(['cmake', '--build', str(ROOT / 'build'), '-j', '16'], ROOT / 'build.log')
        (ROOT / 'results').mkdir(exist_ok=True); (ROOT / 'indexes').mkdir(exist_ok=True)
        for name, (binary, *options) in STEPS:
            out = ROOT / 'results' / name; out.mkdir(exist_ok=True)
            if (out / 'exit.json').exists():
                continue
            status('measuring', step=name)
            started = time.time()
            with (out / 'run.log').open('w') as log:
                code = subprocess.run([str(ROOT / 'build' / binary), *options, f"out={out / 'run'}"], stdout=log,
                                      stderr=subprocess.STDOUT).returncode
            (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
            if code:
                raise RuntimeError(f'{name} failed: {code}')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
