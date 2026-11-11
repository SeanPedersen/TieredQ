"""Nested RAM after write_tier's heap release (#41): the nested default (1-bit prefix, 8-bit fine tier on the NVMe
SSD, direct reads) on VIBE arxiv at k = 100, 200 candidates (candidates_for(100)), at the automatic list count of
86f003d (2,048 here). Decides whether the 0.19 GiB estimate holds and so whether nested passes rabitq-ivf's 1,342
QPS/GB at Recall@100 >= 0.98. Two partitions: the builder's own seed and a seeded sample (sample_seed 1). The
`-fix` models rerun both on the heap-retention fix (../TieredQ-issue43-heapretention-20261111/.venv-fix).
Each model is a fresh process under the shared benchmark lock; 32 build threads, queries alone on CPU 2."""
import datetime, fcntl, json, os, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
DATASET = EXPERIMENTS / 'TieredQ-vibe-arxiv-20261106/data/arxiv-nomic-768-normalized.hdf5'
PYTHON = {'main': str(ROOT / '.venv/bin/python'),  # TieredQ 86f003d, built for this study
          'fix': str(EXPERIMENTS / 'TieredQ-issue43-heapretention-20261111/.venv-fix/bin/python')}
SEARCH_CPU, SEARCH_SIBLING = 2, 20
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (SEARCH_CPU, SEARCH_SIBLING, 19, 35))
CANDIDATES = 200
NPROBES = (16, 24, 32, 48, 64, 96, 128)


def nested(name, python='main', **build):
    return dict(name=name, algorithm='tieredq-nested-ivf', k=100, build=build, python=python,
                queries=[[nprobe, CANDIDATES] for nprobe in NPROBES])


MODELS = [nested('nested-default'), nested('nested-default-s1', sample_seed=1),
          nested('nested-fix', 'fix'), nested('nested-fix-s1', 'fix', sample_seed=1)]


def status(stage, **extra):
    value = json.dumps(dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2) + '\n'
    temporary = ROOT / 'status.tmp'; temporary.write_text(value); temporary.replace(ROOT / 'status.json')


def run(model):
    out = ROOT / 'results' / model['name']
    if (out / 'exit.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.jsonl').unlink(missing_ok=True)
    command = ['taskset', '-c', f'{BUILD_CORES},{SEARCH_CPU}', PYTHON[model['python']], str(ROOT / 'vibe_arxiv.py'), str(DATASET),
               str(out / 'results.jsonl'), model['algorithm'], json.dumps(model['build']), json.dumps(model['queries'])]
    environment = dict(os.environ, BUILD_CORES=BUILD_CORES, QUERY_CORE=str(SEARCH_CPU), THP='off', K=str(model['k']),
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
        (ROOT / 'ssd').mkdir(exist_ok=True)
        for position, model in enumerate(MODELS):
            status('measuring', model=model['name'], position=f'{position + 1}/{len(MODELS)}')
            run(model)
        status('complete'); exit_code = 0
except Exception as exc:
    status('failed', error=str(exc)); raise
finally:
    (ROOT / 'job.exitcode').write_text(f'{exit_code}\n')
