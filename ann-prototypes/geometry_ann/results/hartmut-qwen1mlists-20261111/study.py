"""Does 2K (or 1K) lists beat the automatic 4K on Qwen/MS MARCO 1M x 1024 (#43, #52)? hartmut-defaults-20261111
picked 4K at 1M as the best of 4K/8K/16K/32K, the smallest count it tried; arxiv 1.34M x 768 prefers 2K at k = 10 and
k = 100 (hartmut-arxivfollowup-20261111). Same driver (defaults_qwen.cpp, TieredQ b0148dd), first 1M rows (truth:
exact L2 over them), the shipping prefix (1 bit/dim, 4-bit scan cap) with the 8-bit fine tier in RAM and 40
candidates, the exact recipe for lists below 8K (20 Lloyd iterations, 64 rows per list), exact ranking, nprobe at
lists / {512..16}. Two partition seeds (42, 7). Runs under the shared benchmark lock; resumable per seed."""
import datetime, fcntl, json, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
TIEREDQ = EXPERIMENTS / 'TieredQ-issue41-arxivdefaults-20261111/source/tieredq'  # b0148dd snapshot
TRUTH = EXPERIMENTS / 'TieredQ-issue52-defaults-20261111/truth-1m.ivecs'
ROWS, THREADS, SEARCH_CPU = 8840823, 32, 2
VARIANTS = ';'.join(f'current-{lists // 1024}k:{lists}:20:0:16:64:0' for lists in (1024, 2048, 4096))
ARGS = [f"codes={QWEN / 'base.i8'}", f"scales={QWEN / 'scales.f32'}", f"offsets={QWEN / 'offsets.f32'}", f'rows={ROWS}',
        f"queries={QWEN / 'data/test_queries.fvecs'}", 'query_count=750', f'threads={THREADS}', f'search_cpu={SEARCH_CPU}',
        'use_rows=1000000', f'truth={TRUTH}', 'prefix_bits=1024', 'candidates=40', f'variants={VARIANTS}']
SEEDS = (42, 7)


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
        checked(['cmake', '-S', str(ROOT), '-B', str(ROOT / 'build'), '-DCMAKE_BUILD_TYPE=Release',
                 f'-DTIEREDQ_SOURCE_DIR={TIEREDQ}'], ROOT / 'configure.log')
        checked(['cmake', '--build', str(ROOT / 'build'), '-j', '32', '--target', 'defaults_qwen'], ROOT / 'build.log')
        for seed in SEEDS:
            out = ROOT / 'results' / f'seed{seed}'
            if (out / 'exit.json').exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            status('measuring', seed=seed)
            started = time.time()
            checked([str(ROOT / 'build/defaults_qwen'), *ARGS, f'seed={seed}', f"out={out / 'run'}"], out / 'run.log')
            (out / 'exit.json').write_text(json.dumps(dict(exit_code=0, wall_s=time.time() - started)) + '\n')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
