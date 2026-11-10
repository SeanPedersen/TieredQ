"""TieredQ IVF with the two-level coarse router (#47) on Qwen/MS MARCO 8.84M x 1024 (SQ8 store): the 2-bit
TieredQIVFIndex of the tieredqivf study (same codec seed and Lloyd lists via set_lists, so the same index) with an exact
SQ8 re-rank of its top 40/100, measured with the exact coarse ranking and with the router at expansions 8/16/32/64
(probed supers hold that many member centroids per probed list, at least 16 supers). 128 lists per super, 2-bit router
codes, exact re-rank of the router's first 256 lists. Single queries alone on one pinned core; a 32-thread batch pass
(#51) at three nprobes per list count. The built indexes are saved under indexes/ and reused by reruns. Full nprobe
sweeps on the 250 calibration and 750 held-out queries. Exact assignment runs as FP32 BLAS sgemm (#48).
The scratch-* models build from nothing at Faiss's IVF training budget (4 Lloyd iterations on 40 sample rows per
list), with restricted assignment (256 supers, 16 probes) in training and build, so their build time is end to end
and comparable to Faiss's; they measure the exact ranking and the router at expansions 16/32, top 40. Runs under the shared benchmark lock; resumable per model
(exit.json)."""
import datetime, fcntl, json, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
ROWS, THREADS, SEARCH_CPU = 8840823, 32, 2
QUERIES = ';'.join([f"calibration:{QWEN / 'data/calibration_queries.fvecs'}:{QWEN / 'data/calibration_truth.ivecs'}:250",
                    f"heldout:{QWEN / 'data/test_queries.fvecs'}:{QWEN / 'data/test_truth.ivecs'}:750"])
NPROBES = {65536: [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536],
           131072: [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072]}
BATCH_NPROBES = {65536: [64, 256, 1024], 131072: [128, 512, 2048]}
DUEL = EXPERIMENTS / 'TieredQ-issue44-routerduel-20261110/results'
GIVEN = dict(candidates='40,100', expansions='8,16,32,64')
SCRATCH = dict(candidates='40', expansions='16,32', train_iterations=4, rows_per_list=40, assignment_supers=256,
               assignment_probes=16)
# Scratch builds first: they answer the build-time question (#48) and need no given partition.
MODELS = ([dict(name=f'scratch-{lists // 1024}k', lists=lists, **SCRATCH) for lists in (65536, 131072)] +
          [dict(name=f'router-{lists // 1024}k', lists=lists, centroids=DUEL / f'{lists // 1024}k/centroids.f32', **GIVEN)
           for lists in (65536, 131072)])


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
        (ROOT / 'results').mkdir(exist_ok=True); (ROOT / 'indexes').mkdir(exist_ok=True)
        for model in MODELS:
            out = ROOT / 'results' / model['name']; out.mkdir(exist_ok=True)
            if (out / 'exit.json').exists():
                continue
            status('measuring', model=model['name'])
            lists = model['lists']
            lists_option = ([f"centroids={model['centroids']}"] if 'centroids' in model else
                            [f"{key}={model[key]}" for key in ('train_iterations', 'rows_per_list', 'assignment_supers',
                                                               'assignment_probes')])
            args = [str(ROOT / 'build/tieredq_ivf_qwen'), f"codes={QWEN / 'base.i8'}", f"scales={QWEN / 'scales.f32'}",
                    f"offsets={QWEN / 'offsets.f32'}", f'rows={ROWS}', f'lists={lists}', *lists_option,
                    f"index={ROOT / 'indexes' / (model['name'] + '.tq')}", f'threads={THREADS}', f'search_cpu={SEARCH_CPU}',
                    'code_bits=2048', 'max_scan_bits=4', 'nprobes=' + ','.join(map(str, NPROBES[lists])),
                    f"candidates={model['candidates']}", f"expansions={model['expansions']}", 'lists_per_super=128',
                    'rerank=256', 'router_bits=2', 'min_super_probes=16',
                    f'batch_threads={THREADS}', 'batch_nprobes=' + ','.join(map(str, BATCH_NPROBES[lists])),
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
