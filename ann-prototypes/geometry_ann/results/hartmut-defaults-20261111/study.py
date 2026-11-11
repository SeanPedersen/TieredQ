"""Default thresholds and the RAM lever (#52), TieredQ 9b158c6 through its explicit calls, 750 held-out queries alone on
pinned core 2, nested 8-bit fine suffix tier at 20 candidates.
(1) threshold-1m: the first 1M Qwen rows (truth: exact L2 over them) at 4K/8K/16K/32K lists, each with the current list
training (20 Lloyd iterations, exact assignment, 64 rows per list), the fast one with exact assignment (4, 40) and the
fast one with restricted assignment (4, 256 x 16, 40) plus the router: decides from which list count fast training
and the router become defaults, and checks the rows/135 list rule at 1M (8K lists).
(2) onebit-64k: a 1-bit prefix at 64K lists on all 8.84M rows, 8-bit fine tier on the NVMe SSD (direct reads, 32 in
flight) at 20/30/40 candidates: decides whether about 2 GB of RAM serves 0.95-0.99 against the 2-bit winner.
Runs under the shared benchmark lock; resumable per model (exit.json)."""
import datetime, fcntl, json, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
ROWS, THREADS, SEARCH_CPU = 8840823, 32, 2
STORE = [f"codes={QWEN / 'base.i8'}", f"scales={QWEN / 'scales.f32'}", f"offsets={QWEN / 'offsets.f32'}", f'rows={ROWS}',
         f"queries={QWEN / 'data/test_queries.fvecs'}", 'query_count=750', f'threads={THREADS}', f'search_cpu={SEARCH_CPU}']
THRESHOLD_VARIANTS = ';'.join(
    f'{name}-{lists // 1024}k:{lists}:{spec}' for lists in (4096, 8192, 16384, 32768)
    for name, spec in (('current', '20:0:16:64:0'), ('fast-exact', '4:0:16:40:0'), ('fast', '4:256:16:40:1')))
MODELS = [dict(name='threshold-1m', args=['use_rows=1000000', f"truth={ROOT / 'truth-1m.ivecs'}", 'prefix_bits=2048',
                                          'candidates=20', f'variants={THRESHOLD_VARIANTS}']),
          dict(name='onebit-64k', args=[f"truth={QWEN / 'data/test_truth.ivecs'}", 'prefix_bits=1024', 'candidates=20,30,40',
                                        'nprobes=96,128,192,256,512,1024,2048,4096', 'sweep_exact=0', f"ssd_dir={ROOT / 'ssd'}",
                                        'queue_depth=32', 'variants=fast-64k:65536:4:256:16:40:1'])]


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
        checked(['cmake', '--build', str(ROOT / 'build'), '-j', '16', '--target', 'defaults_qwen'], ROOT / 'build.log')
        (ROOT / 'results').mkdir(exist_ok=True); (ROOT / 'ssd').mkdir(exist_ok=True)
        for model in MODELS:
            out = ROOT / 'results' / model['name']; out.mkdir(exist_ok=True)
            if (out / 'exit.json').exists():
                continue
            status('measuring', model=model['name'])
            started = time.time()
            with (out / 'run.log').open('w') as log:
                code = subprocess.run([str(ROOT / 'build/defaults_qwen'), *STORE, *model['args'], f"out={out / 'run'}"],
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
            if code:
                raise RuntimeError(f"{model['name']} failed: {code}")
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
