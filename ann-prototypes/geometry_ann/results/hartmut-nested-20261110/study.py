"""Nested TieredQ refinement tiers against the SQ8 re-rank (#49) on Qwen/MS MARCO 8.84M x 1024 (SQ8 store). Loads the
64K 2-bit TieredQIVFIndex saved by the router study (Lloyd lists, same codec seed and sample), adds the two-level router
(min 16 probed supers), and measures the exact SQ8 re-rank (1,024 B per vector) next to TieredQNestedIVFIndex suffix
tiers refining to 4/5/6/8 bits/dim total, with and without fine suffix tiers, in RAM, at candidates 40/100/200 over
the full nprobe sweep. Then the 6-bit tier on the NVMe SSD (TieredQFileRows; page-cached and direct reads, queue depths
1/2/4/8). Every query alone on one pinned core; 250 calibration and 750 held-out queries. Runs under the shared
benchmark lock; resumable per model (exit.json)."""
import datetime, fcntl, json, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
INDEXES = EXPERIMENTS / 'TieredQ-issue44-router-20261110/indexes'
ROWS, THREADS, SEARCH_CPU = 8840823, 32, 2
QUERIES = ';'.join([f"calibration:{QWEN / 'data/calibration_queries.fvecs'}:{QWEN / 'data/calibration_truth.ivecs'}:250",
                    f"heldout:{QWEN / 'data/test_queries.fvecs'}:{QWEN / 'data/test_truth.ivecs'}:750"])
NPROBES = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
MODELS = [dict(name='tiers-64k', index=INDEXES / 'router-64k.tq', totals='4,5,6,8', fine='0,1', candidates='40,100,200',
               ssd_total=6, ssd_nprobes='256,1024', queue_depths='1,2,4,8')]


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
                    f"totals={model['totals']}", f"fine={model['fine']}", f'ssd_dir={out}', f"ssd_total={model['ssd_total']}",
                    f"ssd_nprobes={model['ssd_nprobes']}", f"queue_depths={model['queue_depths']}", f'queries={QUERIES}',
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
