"""Bisects the end-to-end recall gap (#52): Python's defaults builds were 0.002-0.005 recall@10 below the C++ study
drivers at every nprobe, for 1- and 2-bit prefixes alike. On the first 1M Qwen rows at 8K lists (fast training plus
router, 2-bit prefix, 8-bit tier in RAM, 20 candidates), defaults_qwen from hartmut-defaults-20261111 compares the
explicit-call build at seeds 42 and 0 (seed variance) with TieredQNestedIVFIndex::build at the same seeds (the
library's own codec sample, list sample and lists), and checks that one-row search_batch (the Python path) returns
search()'s ids. Runs under the shared benchmark lock; resumable per model (exit.json)."""
import datetime, fcntl, json, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
QWEN = EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107'
TRUTH = EXPERIMENTS / 'TieredQ-issue52-defaults-20261111/truth-1m.ivecs'
COMMON = [f"codes={QWEN / 'base.i8'}", f"scales={QWEN / 'scales.f32'}", f"offsets={QWEN / 'offsets.f32'}", 'rows=8840823',
          'use_rows=1000000', f"queries={QWEN / 'data/test_queries.fvecs'}", 'query_count=750', f'truth={TRUTH}',
          'threads=32', 'search_cpu=2', 'prefix_bits=2048', 'candidates=20', 'sweep_exact=0', 'check_batch=1',
          'variants=fast-8k:8192:4:256:16:40:1']
MODELS = [(f'{path}-{seed}', [f'seed={seed}', f'builder={int(path == "builder")}'])
          for path in ('explicit', 'builder') for seed in (42, 0)]
# Full scale: the explicit 2-bit build at seed 0 against the seed-42 reference (hartmut-ssdtier-20261111, 0.9513 at
# nprobe 128): whether the Python builds' 0.005 deficit (seed 0) is the partition seed.
FULL = [f"codes={QWEN / 'base.i8'}", f"scales={QWEN / 'scales.f32'}", f"offsets={QWEN / 'offsets.f32'}", 'rows=8840823',
        f"queries={QWEN / 'data/test_queries.fvecs'}", 'query_count=750', f"truth={QWEN / 'data/test_truth.ivecs'}",
        'threads=32', 'search_cpu=2', 'prefix_bits=2048', 'candidates=20', 'sweep_exact=0',
        'nprobes=96,128,192,256,512,1024', 'variants=fast-64k:65536:4:256:16:40:1']
MODELS_FULL = [('full-explicit-0', ['seed=0']), ('full-explicit-42', ['seed=42']),
               ('full-builder-0', ['seed=0', 'builder=1']), ('full-internal-0', ['seed=0', 'internal_lists=1']),
               ('full-explicit-7', ['seed=7'])]
# After the fix (list and codec samples on their own seed streams): the builder at full scale, as Python builds.
MODELS_FIXED = [('full-builder-0-fixed', ['seed=0', 'builder=1']), ('full-builder-42-fixed', ['seed=42', 'builder=1']),
                ('full-builder-7-fixed', ['seed=7', 'builder=1'])]
# The steps only Python takes, on the 1M builder-0 model: a serialize/deserialize round trip, and the tier written by
# write_tier() and read from its file.
SSD = f"ssd_dir={ROOT / 'ssd'}"
MODELS_STEPS = [('builder-0-roundtrip', ['seed=0', 'builder=1', 'round_trip=1']),
                ('builder-0-tierfile', ['seed=0', 'builder=1', 'tier_file=1', SSD]),
                ('builder-0-both', ['seed=0', 'builder=1', 'round_trip=1', 'tier_file=1', SSD])]


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
        (ROOT / 'ssd').mkdir(exist_ok=True)
        steps = [(n, COMMON + a) for n, a in MODELS] + [(n, FULL + a) for n, a in MODELS_FULL]
        steps += [(n, COMMON + a) for n, a in MODELS_STEPS] + [(n, FULL + a) for n, a in MODELS_FIXED]
        for name, args in steps:
            out = ROOT / 'results' / name; out.mkdir(parents=True, exist_ok=True)
            if (out / 'exit.json').exists():
                continue
            status('measuring', model=name)
            started = time.time()
            with (out / 'run.log').open('w') as log:
                code = subprocess.run([str(ROOT / 'build/defaults_qwen'), *args, f"out={out / 'run'}"],
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
            if code:
                raise RuntimeError(f'{name} failed: {code}')
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
