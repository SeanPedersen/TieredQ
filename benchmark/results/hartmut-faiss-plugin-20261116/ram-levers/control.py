"""#55 RAM study control: is the Qwen screen's ~0.003 recall spread codec or coarse routing?

Serves build the HNSW graph over the 65,536 fp16 centroids with 32 threads, which may not give the same graph twice.
Here base, res and rows are rebuilt at 2.0625 bits/dim (partition seed 1, minram centroids) and each plugin is served
with OMP_NUM_THREADS=1 throughout, so the graph is built serially; base is served twice with that setting and once
with 32 threads (the screen's setting) to measure routing noise. Results: results/control-<variant>/serve-<label>/.
"""
import json, os, shutil, sys
sys.argv = [sys.argv[0]]
from pathlib import Path

# study.py's definitions only (its run starts at the status line); status goes to control-status.json.
source = (Path.cwd() / 'study.py').read_text().split("\nstatus('queued_for_benchmark_lock')")[0]
study = type(sys)('study'); exec(compile(source, 'study.py', 'exec'), study.__dict__)


def control_status(stage, **extra):
    import datetime
    value = dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    (study.ROOT / 'control-status.json').write_text(json.dumps(value, indent=2) + '\n')


study.status = control_status

BITS = 2.0625
SERVES = {'base': ('serial-a', 'serial-b', 'parallel'), 'res': ('serial-a',), 'rows': ('serial-a',)}
ENV = {'base': {}, 'res': study.VARIANTS['res'], 'rows': study.VARIANTS['rows']}

study.status('queued_for_benchmark_lock')
exit_code = 1
try:
    import fcntl
    with (study.EXPERIMENTS / '.tieredq-issue44-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report = {}
        data = study.BASE / 'data' / 'qwen'
        for variant, labels in SERVES.items():
            tag = f'control-{variant}'
            indexes = study.ROOT / 'indexes' / tag
            indexes.mkdir(parents=True, exist_ok=True)
            env = dict(ENV[variant], NPROBES=study.NPROBES['qwen'], CENTROIDS=str(study.MINRAM / 'centroids/qwen-s1.f32'))
            study.step(f'{tag}/build', [study.BENCH, 'build', data, 1, indexes, f'{BITS:g}', 'tieredq'], study.BUILD_CORES, env)
            for label in labels:
                serve = study.ROOT / 'results' / tag / f'serve-{label}' / 'serve.json'
                threads = {'OMP_NUM_THREADS': '1'} if label.startswith('serial') else {}
                study.step(f'{tag}/serve-{label}', [study.BENCH, 'serve', data, indexes, 'plugin', 'plugin.tqf', serve],
                           study.SERVE_CORES, dict(env, **threads))
                points = json.loads(serve.read_text())['points']
                report[f'{variant} {label}'] = {str(p['nprobe']): p['recall'] for p in points}
                (study.ROOT / 'control.json').write_text(json.dumps(report, indent=2) + '\n')
            shutil.rmtree(indexes, ignore_errors=True)
        study.status('control complete'); exit_code = 0
except Exception as exc:
    study.status('control failed', error=str(exc)); raise
finally:
    (study.ROOT / 'control.exitcode').write_text(f'{exit_code}\n')
