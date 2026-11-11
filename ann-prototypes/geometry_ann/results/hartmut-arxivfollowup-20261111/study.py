"""Follow-up to hartmut-arxivdefaults-20261111 on VIBE arxiv (vibe_arxiv.py with K and memory attribution). Decides:
- #43 / automatic_lists: is the 4K-list loss at k = 100 about k? The same 1K/2K/4K flat builds at recall@10.
- #42 and flat's USS above storage_bytes: does THP 'always' still inflate the issue's own configuration (2 bits/dim, 2,048 lists, THP host vs off), and
  which mappings hold flat's ~45 MiB of USS above storage_bytes (per-mapping smaps growth)?
- #41 / candidates_for(k): nested at k = 100 with 150 candidates, and where its 0.128 GiB above storage_bytes sits
  (USS per step, serialized refinement size).
- #27: nested (1-bit + 8-bit tier on SSD, default 40 candidates) against flat at recall@10 on a second dataset.
Each model is a fresh process under the shared benchmark lock; 32 build threads, queries alone on CPU 2."""
import datetime, fcntl, json, os, subprocess, time
from pathlib import Path

ROOT = Path.cwd()
EXPERIMENTS = ROOT.parent
DATASET = EXPERIMENTS / 'TieredQ-vibe-arxiv-20261106/data/arxiv-nomic-768-normalized.hdf5'
RABITQ_PYTHON = [str(EXPERIMENTS / 'TieredQ-vibe-arxiv-20261106/rq-env/bin/python')]
# hartmut-arxivdefaults-20261111's b0148dd venv (uv sync, plus h5py and psutil), reused unchanged.
TIEREDQ_PYTHON = [str(EXPERIMENTS / 'TieredQ-issue41-arxivdefaults-20261111/source/tieredq/.venv/bin/python')]
SEARCH_CPU, SEARCH_SIBLING = 2, 20
BUILD_CORES = ','.join(str(c) for c in range(36) if c not in (SEARCH_CPU, SEARCH_SIBLING, 19, 35))
NPROBES = {4096: [16, 24, 32, 48, 64, 96, 128, 160, 192], 2048: [8, 12, 16, 24, 32, 48, 64, 80, 96],
           1024: [4, 6, 8, 12, 16, 24, 32, 40, 48]}


def flat(name, bits, lists=None, thp='off', k=100, **extra):
    return dict(name=name, algorithm='tieredq-ivf', thp=thp, k=k, build=dict(bits_per_dim=bits, lists=lists, **extra),
                queries=NPROBES[lists or 4096])


def nested(name, queries, k):
    return dict(name=name, algorithm='tieredq-nested-ivf', thp='off', k=k, build={}, queries=queries)


MODELS = [
    flat('flat175-1024-k10', 1.75, 1024, k=10),
    flat('flat175-2048-k10', 1.75, 2048, k=10),
    flat('flat175-auto-k10', 1.75, k=10),
    nested('nested-default-k10', [[nprobe, 40] for nprobe in (16, 32, 64, 128, 256)], k=10),
    nested('nested-default-k100', [[64, 150], [96, 150], [128, 150], [192, 150], [64, 200]], k=100),
    flat('flat200-2048', 2.0, 2048),
    flat('flat200-2048-thphost', 2.0, 2048, thp='host'),
    # One-time process state (module, BLAS buffers, OpenMP pool) built before the baseline: per-index growth only.
    dict(flat('flat175-2048-warm', 1.75, 2048), env=dict(WARMUP_ROWS='65536')),
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
    environment = dict(os.environ, BUILD_CORES=BUILD_CORES, QUERY_CORE=str(SEARCH_CPU), THP=model['thp'], K=str(model['k']),
                       TIER_DIR=str(ROOT / 'ssd'), OMP_NUM_THREADS='32', OPENBLAS_NUM_THREADS='1',
                       OMP_WAIT_POLICY='PASSIVE', OMP_PROC_BIND='false', OMP_DYNAMIC='false', **model.get('env', {}))
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
