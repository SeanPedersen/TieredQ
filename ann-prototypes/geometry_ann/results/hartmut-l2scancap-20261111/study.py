"""L2 automatic scan width (#44 finding, hartmut-centroidq / compressed): should L2 get a 4-bit automatic cap
(max(4, ceil(bits/dim)), the counterpart of IP/cosine's max(6, ceil) without additive books, which L2 dropped in
9e7bb30 for -0.72 pt R@10 on SIFT1M)? Above 4 bits per coordinate L2 scores on the generic kernel either way, so the
decision covers <= 4 bits/dim. Compares the automatic width (8) with max_scan_bits 4 on held-out recall@10 and
single-query latency, two codec / partition seeds each:
- flat TieredQIndex on SIFT1M (1, 2, 3, 4 bits/dim) and on the first 1M Qwen/MS MARCO rows (2, 3, 4);
- TieredQIVFIndex on Qwen 1M at 2 bits/dim (automatic 2K lists), where IVF's own rule already applies the 4-bit cap
  to every metric: does that rule cost L2 the recall the flat 2-bit models lose? (cap 0 here means the automatic width
  of TieredQ::train, which is what IVF passes through for L2 above its rule, so cap 8 is given explicitly.)
The planned Qwen flat 4-bit and IVF 3/4-bit models were dropped once the flat 2-3 bit results had decided the default.
One fresh process per model under the shared benchmark lock; 32 build threads, single queries on CPU 2."""
import datetime, fcntl, json, os, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
PYTHON = str(EXPERIMENTS / 'TieredQ-issue41-nestedram-20261111/.venv/bin/python')  # TieredQ 86f003d
SEARCH_CPU, SEARCH_SIBLING = 2, 20
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (SEARCH_CPU, SEARCH_SIBLING, 19, 35))
QWEN_ROWS = 1_000_000
IVF_NPROBES = [16, 32, 64, 128, 256]
SEEDS = (0, 1)
CAPS = (0, 4)


def model(dataset, kind, bits, cap, seed):
    return dict(name=f'{dataset}-{kind}-b{bits:g}-cap{cap}-s{seed}', dataset=dataset, kind=kind, bits=bits, cap=cap, seed=seed)


MODELS = ([model('sift', 'flat', b, c, s) for b in (1, 2, 3, 4) for s in SEEDS for c in CAPS] +
          [model('qwen1m', 'flat', b, c, s) for b in (2, 3) for s in SEEDS for c in CAPS] +
          [model('qwen1m', 'ivf', 2, c, s) for s in SEEDS for c in (8, 4)])


def prepare_datasets():
    links = {'sift': {'base.fvecs': EXPERIMENTS / 'TieredQ-issue44-simd-20261107/data/base.fvecs',
                      'test_queries.fvecs': EXPERIMENTS / 'TieredQ-issue44-simd-20261107/data/test_queries.fvecs',
                      'truth.ivecs': EXPERIMENTS / 'TieredQ-issue44-simd-20261107/data/test_truth.ivecs'},
             'qwen1m': {'base.i8': EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107/base.i8',
                        'scales.f32': EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107/scales.f32',
                        'offsets.f32': EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107/offsets.f32',
                        'test_queries.fvecs': EXPERIMENTS / 'TieredQ-issue44-msmarco8m-20261107/data/test_queries.fvecs',
                        'truth.ivecs': EXPERIMENTS / 'TieredQ-issue52-defaults-20261111/truth-1m.ivecs'}}
    for dataset, files in links.items():
        directory = ROOT / 'data' / dataset
        directory.mkdir(parents=True, exist_ok=True)
        for name, target in files.items():
            if not (directory / name).exists():
                (directory / name).symlink_to(target)


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def run(m):
    out = ROOT / 'results' / m['name']
    if (out / 'exit.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    command = ['taskset', '-c', f'{BUILD_CORES},{SEARCH_CPU}', PYTHON, str(ROOT / 'l2_scan_cap.py'), str(ROOT / 'data' / m['dataset']),
               str(out / 'run.json'), m['kind'], str(m['bits']), str(m['cap']), str(m['seed']),
               *([json.dumps(IVF_NPROBES)] if m['kind'] == 'ivf' else [])]
    environment = dict(os.environ, BUILD_CORES=BUILD_CORES, QUERY_CORE=str(SEARCH_CPU), ROWS=str(QWEN_ROWS),
                       OMP_NUM_THREADS='32', OPENBLAS_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE', OMP_PROC_BIND='false',
                       OMP_DYNAMIC='false')
    started = time.time()
    with (out / 'run.log').open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment).returncode
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
    if code:
        raise RuntimeError(f"{m['name']} failed: {code}")


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepare_datasets()
        for position, m in enumerate(MODELS):
            status('measuring', model=m['name'], position=f'{position + 1}/{len(MODELS)}')
            run(m)
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
