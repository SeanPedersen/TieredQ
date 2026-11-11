"""Shipping configuration of #52 through the library (#47, #49): from-scratch 64K index, router at the library
defaults (rerank 64 after the re-rank depth test, faster head packing), on the 750 held-out queries alone on one
pinned core: the in-RAM SQ8 re-rank of the top 40, the 6-bit fine tier in RAM, and the 6-bit tier with
TieredQNestedIVFIndex::attach_exact over SQ8 rows on the NVMe SSD (direct I/O, 32 reads in flight, 16 finals). Built
from the TieredQ commit in source/COMMIT plus the working tree in source/tieredq. Runs under the shared benchmark lock;
resumable per model (exit.json)."""
import datetime, fcntl, json, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
INDEXES = EXPERIMENTS / 'TieredQ-issue44-router-20261110/indexes'
ROWS, THREADS, SEARCH_CPU = 8840823, 32, 2
QUERIES = f"heldout:{QWEN / 'data/test_queries.fvecs'}:{QWEN / 'data/test_truth.ivecs'}:750"
NPROBES = [64, 96, 128, 192, 256, 512, 1024]
MODELS = [dict(name='shipping-64k', index=INDEXES / 'scratch-64k.tq', totals='6', fine='1', candidates='40',
               extra=[f"sq8_rows={ROOT / 'ssd/sq8.rows'}", 'finals=16', 'final_depth=32']),
          # 16 finals lost 0.0002-0.0005 recall at rerank 64; more finals for exact SQ8 parity.
          dict(name='shipping-64k-finals', index=INDEXES / 'scratch-64k.tq', totals='6', fine='1', candidates='40',
               extra=[f"sq8_rows={ROOT / 'ssd/sq8.rows'}", 'finals=24,32', 'final_depth=32']),
          # 16-32 finals gave the same recall, 0.0002-0.0005 below SQ8: float vs double final sums (truth is fp32).
          dict(name='shipping-64k-precision', index=INDEXES / 'scratch-64k.tq', totals='6', fine='1', candidates='40',
               extra=[f"sq8_rows={ROOT / 'ssd/sq8.rows'}", 'finals=16', 'final_depth=32'])]


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
        checked(['cmake', '--build', str(ROOT / 'build'), '-j', '16', '--target', 'shipping_qwen'], ROOT / 'build.log')
        (ROOT / 'results').mkdir(exist_ok=True); (ROOT / 'ssd').mkdir(exist_ok=True)
        for model in MODELS:
            out = ROOT / 'results' / model['name']; out.mkdir(exist_ok=True)
            if (out / 'exit.json').exists():
                continue
            status('measuring', model=model['name'])
            args = [str(ROOT / 'build/shipping_qwen'), f"codes={QWEN / 'base.i8'}", f"scales={QWEN / 'scales.f32'}",
                    f"offsets={QWEN / 'offsets.f32'}", f'rows={ROWS}', f"index={model['index']}", f'threads={THREADS}',
                    f'search_cpu={SEARCH_CPU}', 'nprobes=' + ','.join(map(str, NPROBES)), f"candidates={model['candidates']}",
                    f"totals={model['totals']}", f"fine={model['fine']}", f'queries={QUERIES}',
                    f"out={out / 'run'}"] + model.get('extra', [])
            started = time.time()
            with (out / 'run.log').open('w') as log:
                code = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT).returncode
            (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
            if code:
                raise RuntimeError(f"{model['name']} failed: {code}")
        (ROOT / 'ssd/sq8.rows').unlink(missing_ok=True)  # a copy of the SQ8 store, written by the driver
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
