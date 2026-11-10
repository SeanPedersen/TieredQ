"""TieredQ's own IVF as shipped on Qwen/MS MARCO 8.84M x 1024 (SQ8 store), built with the streaming
TieredQIVFIndex::build(RowLoader) from the int8 codes, on the same Lloyd partitions as the Faiss and geometry_ann runs
(set_lists with each list's mean, from the router-duel study), so no coarse training is repeated. One build per list
count (64K, 128K): a 2-bit prefix IVF, measured alone, with an exact SQ8 re-rank of its top 40/100, and as
TieredQNestedIVFIndex with a suffix tier to 4 bits/dim rescoring the top 40/100. Inner product on the normalized inputs;
4-bit scan cap; every query alone on one pinned core. Full nprobe sweeps on the 250 calibration and 750 held-out
queries, next to the Faiss IVF-SQ8 curves of the finelists study. Runs under the shared benchmark lock; resumable per
model (exit.json)."""
import datetime, fcntl, json, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
ROWS, THREADS, SEARCH_CPU = 8840823, 32, 2
QUERIES = ';'.join([f"calibration:{QWEN / 'data/calibration_queries.fvecs'}:{QWEN / 'data/calibration_truth.ivecs'}:250",
                    f"heldout:{QWEN / 'data/test_queries.fvecs'}:{QWEN / 'data/test_truth.ivecs'}:750"])
NPROBES = {65536: [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536],
           131072: [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072]}
DUEL = EXPERIMENTS / 'TieredQ-issue44-routerduel-20261110/results'
MODELS = [dict(name=f'lloyd-{lists // 1024}k', lists=lists, prefix_bits=2048, centroids=DUEL / f'{lists // 1024}k/centroids.f32')
          for lists in (65536, 131072)]


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
        checked(['cmake', '--build', str(ROOT / 'build'), '-j', '16', '--target', 'tieredq_ivf_qwen'], ROOT / 'build.log')
        (ROOT / 'results').mkdir(exist_ok=True)
        for model in MODELS:
            out = ROOT / 'results' / model['name']; out.mkdir(exist_ok=True)
            if (out / 'exit.json').exists():
                continue
            status('measuring', model=model['name'])
            args = [str(ROOT / 'build/tieredq_ivf_qwen'), f"codes={QWEN / 'base.i8'}", f"scales={QWEN / 'scales.f32'}",
                    f"offsets={QWEN / 'offsets.f32'}", f'rows={ROWS}', f"lists={model['lists']}", f"centroids={model['centroids']}",
                    f'threads={THREADS}', f'search_cpu={SEARCH_CPU}', 'code_bits=4096', f"prefix_bits={model['prefix_bits']}",
                    'nprobes=' + ','.join(map(str, NPROBES[model['lists']])), 'candidates=40,100', f'queries={QUERIES}',
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
