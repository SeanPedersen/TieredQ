"""Router head-to-head over the leaf centroids at 32K/64K/128K lists (Lloyd partitions from the emeans and finelists
studies): exact fp32 scan, full TieredQ 2-bit scan, TieredQ IVF (library coarse ranking), our two-level router (fp32
super ranking + SQ8/SQ4 member codes), and a degree-16 exact-kNN/RNG graph walked with fp32, SQ8 or SQ4 distances.
Quality is the share of each held-out query's true top-10 whose list is in the router's first L lists; routing time is
one pinned core. Centroids are the means of each list's SQ8-decoded rows. Runs under the shared benchmark lock."""
import datetime, fcntl, json, subprocess
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
STAGE_A = EXPERIMENTS / 'TieredQ-issue44-finelists-20261110/results'
LABELS = {'32k': (32768, EXPERIMENTS / 'TieredQ-issue44-emeans-20261110/results/lloyd-32k/labels.u32'),
          '64k': (65536, STAGE_A / 'lloyd-64k/labels.u32'), '128k': (131072, STAGE_A / 'lloyd-128k/labels.u32')}
THREADS, SEARCH_CPU, QUERIES, BITS = 32, 2, 750, 2


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
        checked(['cmake', '-S', str(ROOT / 'source'), '-B', str(ROOT / 'build'), '-DCMAKE_BUILD_TYPE=Release'], ROOT / 'configure.log')
        checked(['cmake', '--build', str(ROOT / 'build'), '-j', '16', '--target', 'duel'], ROOT / 'build.log')
        (ROOT / 'results').mkdir(exist_ok=True)
        for name, (k, labels) in LABELS.items():
            out = ROOT / 'results' / name; out.mkdir(exist_ok=True)
            if (out / 'exit.json').exists():
                continue
            centroids = out / 'centroids.f32'
            if not centroids.exists():
                status('centroids', size=name)
                checked(['uv', 'run', '--no-project', '--python', '3.12', '--with', 'numpy', str(ROOT / 'source/centroid_means.py'),
                         '--codes', str(QWEN / 'base.i8'), '--scales', str(QWEN / 'scales.f32'), '--offsets', str(QWEN / 'offsets.f32'),
                         '--labels', str(labels), '--output', str(centroids), '--dim', '1024', '--k', str(k)], out / 'centroids.log')
            status('duel', size=name)
            with (out / 'duel.txt').open('w') as stdout:
                code = subprocess.run([str(ROOT / 'build/duel'), str(centroids), str(labels), str(QWEN / 'data/test_queries.fvecs'),
                                       str(QWEN / 'data/test_truth.ivecs'), str(BITS), str(THREADS), str(QUERIES), str(SEARCH_CPU)],
                                      stdout=stdout, stderr=subprocess.STDOUT, env={'OPENBLAS_NUM_THREADS': str(THREADS), 'PATH': '/usr/bin:/bin'}).returncode
            (out / 'exit.json').write_text(json.dumps(dict(exit_code=code)) + '\n')
            if code:
                raise RuntimeError(f'{name} failed: {code}')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
