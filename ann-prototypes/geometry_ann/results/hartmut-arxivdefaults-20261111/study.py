"""b0148dd defaults on VIBE arxiv-nomic-768-normalized (1.34M x 768, k = 100), VIBE-like harness (vibe_arxiv.py).
Decides:
- #43: do the automatic list count (4K here) and the 4-bit scan cap hold at 768-D / k = 100 against the #43 best
  configurations (1024/2048 lists, 6-bit cap), and would the router (now from 8K lists) help at 4K?
- #41: QPS/GB of the nested 1-bit default with its 8-bit tier on SSD (100/200/400 candidates = reads per query)
  against flat TieredQ and rabitq-ivf (4 bits, 2048 lists) on the same host and day.
- #42: does THP 'always' still inflate the default build's USS (one THP=host build against its THP-off twin)?
- #34: list padding at the automatic list count (recorded with every TieredQ build).
Each model is a fresh process (its own USS), resumable via exit.json, under the shared benchmark lock; 32 build
threads, queries alone on CPU 2 (sibling 20 idle). Tier files are removed after each model."""
import datetime, fcntl, json, os, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
DATASET = EXPERIMENTS / 'TieredQ-vibe-arxiv-20261106/data/arxiv-nomic-768-normalized.hdf5'
RABITQ_PYTHON = [str(EXPERIMENTS / 'TieredQ-vibe-arxiv-20261106/rq-env/bin/python')]
# The source tree's venv (uv sync, plus h5py and psutil); `uv run --with` recompiled the package for every model.
TIEREDQ_PYTHON = [str(ROOT / 'source/tieredq/.venv/bin/python')]
SEARCH_CPU, SEARCH_SIBLING = 2, 20
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (SEARCH_CPU, SEARCH_SIBLING, 19, 35))
NPROBES = {4096: [16, 24, 32, 48, 64, 96, 128, 160, 192], 2048: [8, 12, 16, 24, 32, 48, 64, 80, 96],
           1024: [4, 6, 8, 12, 16, 24, 32, 40, 48]}
NESTED = [[nprobe, candidates] for candidates in (100, 200, 400) for nprobe in (32, 64, 96, 128, 192)]


def flat(name, bits, lists=None, thp='off', **extra):
    return dict(name=name, algorithm='tieredq-ivf', thp=thp, build=dict(bits_per_dim=bits, lists=lists, **extra),
                queries=NPROBES[lists or 4096])


MODELS = [
    flat('flat175-auto', 1.75),
    flat('flat175-2048', 1.75, 2048),
    flat('flat175-1024', 1.75, 1024),
    flat('flat175-4096-s1', 1.75, 4096, sample_seed=1),
    flat('flat175-2048-s1', 1.75, 2048, sample_seed=1),
    flat('flat175-auto-cap6', 1.75, max_scan_bits=6),
    flat('flat175-2048-cap6', 1.75, 2048, max_scan_bits=6),
    flat('flat175-4096-router', 1.75, 4096, router=True),
    flat('flat125-auto', 1.25),
    flat('flat125-1024', 1.25, 1024),
    dict(name='nested-default', algorithm='tieredq-nested-ivf', thp='off', build={}, queries=NESTED),
    dict(name='rabitq-4b-2048', algorithm='rabitq-ivf', thp='off', build=dict(num_clusters=2048, total_bits=4),
         queries=[8, 12, 16, 24, 32, 48, 64]),
    flat('flat175-auto-thphost', 1.75, thp='host'),
]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def run(model):
    out = ROOT / 'results' / model['name']
    if (out / 'exit.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.jsonl').unlink(missing_ok=True)
    python = RABITQ_PYTHON if model['algorithm'] == 'rabitq-ivf' else TIEREDQ_PYTHON
    command = ['taskset', '-c', f'{BUILD_CORES},{SEARCH_CPU}', *python, str(ROOT / 'vibe_arxiv.py'), str(DATASET),
               str(out / 'results.jsonl'), model['algorithm'], json.dumps(model['build']), json.dumps(model['queries'])]
    environment = dict(os.environ, BUILD_CORES=BUILD_CORES, QUERY_CORE=str(SEARCH_CPU), THP=model['thp'],
                       TIER_DIR=str(ROOT / 'ssd'), OMP_NUM_THREADS='32', OPENBLAS_NUM_THREADS='1',
                       OMP_WAIT_POLICY='PASSIVE', OMP_PROC_BIND='false', OMP_DYNAMIC='false')
    started = time.time()
    with (out / 'run.log').open('w') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment).returncode
    for tier in (ROOT / 'ssd').glob('tier-*.rows'):
        tier.unlink()
    (out / 'exit.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time() - started)) + '\n')
    if code:
        raise RuntimeError(f"{model['name']} failed: {code}")


status('queued_for_benchmark_lock')
exit_code = 1
try:
    with (EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status('building_package')
        with (ROOT / 'build.log').open('w') as log:  # builds the package (Release) into the source tree's venv
            for command in (['uv', 'sync', '--project', str(ROOT / 'source/tieredq'), '--no-dev'],
                            ['uv', 'pip', 'install', '--python', TIEREDQ_PYTHON[0], 'h5py', 'psutil']):
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               env=dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL='32'))
        (ROOT / 'ssd').mkdir(exist_ok=True)
        for position, model in enumerate(MODELS):
            status('measuring', model=model['name'], position=f'{position + 1}/{len(MODELS)}')
            run(model)
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
