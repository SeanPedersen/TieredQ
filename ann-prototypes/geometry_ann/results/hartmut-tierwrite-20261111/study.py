"""Tier-write time A/B (#52 follow-up): `write_tier` took 117 s at 6298356 (../hartmut-quiethost-20261111) against
94 s at 9b158c6 (../hartmut-pythondefaults-20261111), on the 1-bit nested default over Qwen/MS MARCO 8.84M. Is that
our code or the host? A = the 9b158c6 tree of that earlier run, B = 6298356 (../TieredQ-issue53-packedlists-20261111
package). Each step runs ../hartmut-quiethost-20261111/build.py (defaults only, SQ8 memmap in, 32 threads, tier on
NVMe) in ABBA order, so drift in host state lands on both. Each step records the host at its start and the child's
user and system CPU seconds; CPU per wall second separates compute from waiting on the SSD. The index and tier are
removed after each step (sizes kept in files.json). Under the shared lock; resumable per step."""
import datetime, fcntl, json, os, resource, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
BUILD_SCRIPT = EXPERIMENTS / 'TieredQ-issue52-quiethost-20261111/build.py'
PYTHONS = {'A': EXPERIMENTS / 'TieredQ-issue52-pythondefaults-20261111/source/tieredq/.venv/bin/python',
           'B': EXPERIMENTS / 'TieredQ-issue53-packedlists-20261111/.venv-packed/bin/python'}
ORDER = 'ABBA'
BENCHMARK_CPUS = (2, 20, 19, 35)
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in BENCHMARK_CPUS)
BUILD_THREADS = 32
DIMENSIONS = 1024


def write_json(path, value):
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


def status(stage, **extra):
    write_json(ROOT / 'status.json', dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra))


def host():
    load = open('/proc/loadavg').read().split()[:3]
    pressure = open('/proc/pressure/io').read().split('\n')[0]
    top = subprocess.run(['top', '-bn1', '-o', '%CPU'], capture_output=True, text=True).stdout.splitlines()[7:12]
    return dict(loadavg=load, io_pressure=pressure, top=[line.strip() for line in top])


def step(position, variant):
    out = ROOT / 'results' / f'{position}-{variant}'
    if (out / 'exit.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    status('running', step=out.name)
    write_json(out / 'host.json', host())
    index, tier = ROOT / 'ssd/index.tq', ROOT / 'ssd/tier.rows'
    command = ['taskset', '-c', BUILD_CORES, str(PYTHONS[variant]), str(BUILD_SCRIPT), str(QWEN / 'base.i8'),
               str(QWEN / 'scales.f32'), str(QWEN / 'offsets.f32'), str(DIMENSIONS), str(index), str(tier),
               str(out / 'build.json')]
    before, started = resource.getrusage(resource.RUSAGE_CHILDREN), time.time()
    with (out / 'run.log').open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                              env=dict(os.environ, OMP_NUM_THREADS=str(BUILD_THREADS))).returncode
    after, wall = resource.getrusage(resource.RUSAGE_CHILDREN), time.time() - started
    if code == 0:
        write_json(out / 'files.json', {path.name: path.stat().st_size for path in (index, tier)})
    for path in (index, tier):
        path.unlink(missing_ok=True)
    write_json(out / 'exit.json', dict(exit_code=code, wall_s=wall, user_s=after.ru_utime - before.ru_utime,
                                       system_s=after.ru_stime - before.ru_stime))
    if code:
        raise RuntimeError(f'{out.name} failed: {code}')


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (ROOT / 'ssd').mkdir(exist_ok=True)
        for position, variant in enumerate(ORDER, start=1):
            step(position, variant)
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
