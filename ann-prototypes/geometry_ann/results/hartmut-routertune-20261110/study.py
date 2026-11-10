"""Router tuning (#47): the router study's scratch indexes (64K and 128K lists, loaded, not rebuilt) with more probed
supers. At expansions 16/32 the router lost 0.013-0.025 recall@10 to exact routing at nprobe 128-256 for 0.1-0.3 ms
per expansion step. Here: expansions 64/128/256, at least 32 probed supers, nprobe 64-512 (64K) and 128-1024 (128K),
top-40 SQ8 re-rank, 750 held-out queries alone on one pinned core. Uses the router study's build of the driver.
Runs under the shared benchmark lock; resumable per model (exit.json)."""
import datetime, fcntl, json, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
ROUTER = EXPERIMENTS / 'TieredQ-issue44-router-20261110'
ROWS, THREADS, SEARCH_CPU = 8840823, 32, 2
QUERIES = f"heldout:{QWEN / 'data/test_queries.fvecs'}:{QWEN / 'data/test_truth.ivecs'}:750"
MODELS = [dict(name='tune-64k', lists=65536, nprobes='64,96,128,192,256,512'),
          dict(name='tune-128k', lists=131072, nprobes='128,192,256,384,512,1024')]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (ROOT / 'results').mkdir(exist_ok=True)
        for model in MODELS:
            out = ROOT / 'results' / model['name']; out.mkdir(exist_ok=True)
            if (out / 'exit.json').exists():
                continue
            status('measuring', model=model['name'])
            index = ROUTER / 'indexes' / f"scratch-{model['lists'] // 1024}k.tq"
            args = [str(ROUTER / 'build/tieredq_ivf_qwen'), f"codes={QWEN / 'base.i8'}", f"scales={QWEN / 'scales.f32'}",
                    f"offsets={QWEN / 'offsets.f32'}", f'rows={ROWS}', f"lists={model['lists']}", f'index={index}',
                    f'threads={THREADS}', f'search_cpu={SEARCH_CPU}', 'code_bits=2048', 'max_scan_bits=4',
                    f"nprobes={model['nprobes']}", 'candidates=40', 'exact=0', 'expansions=64,128,256',
                    'lists_per_super=128', 'rerank=256', 'router_bits=2', 'min_super_probes=32',
                    f'queries={QUERIES}', f"out={out / 'run'}"]
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
