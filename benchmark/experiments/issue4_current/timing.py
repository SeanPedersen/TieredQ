"""One reproducible interleaved timing session after matching/recall lanes exit."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import statistics
from collections import defaultdict
from matrix import DIMS,primary_cells,primary_matches,primary_equal

B = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(B))


def wait_for_other_jobs(out):
    """Do not start a timed cell while another benchmark, compiler or test suite is active."""
    patterns=('measure_memory.py','match_memory.py','equal_recall.py','scan_benchmark','/run.py',
              ' -m pytest ','cibuildwheel')
    while True:
        processes=subprocess.check_output(['ps','-eo','pid,comm,args'],text=True).splitlines()[1:]
        busy=[]
        for line in processes:
            fields=line.split(None,2)
            if len(fields)<3 or int(fields[0])==os.getpid():
                continue
            if fields[1] in ('cc1plus','rustc','cargo','pytest') or any(word in fields[2] for word in patterns):
                busy.append(line)
        if not busy:
            return
        with (out/'quiet-wait.log').open('a') as log:
            log.write(dt_stamp()+'\n'+'\n'.join(busy)+'\n')
        print('Waiting for other build/test/benchmark jobs before timing',flush=True)
        time.sleep(30)


def dt_stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('workspace', type=Path)
    a = p.parse_args()
    w = a.workspace.resolve()
    assert (w/'match-exit-code').read_text().strip() == '0'
    assert (w/'glove1m/match-exit-code').read_text().strip() == '0'
    out = w/'timing'
    out.mkdir(exist_ok=True)
    # A partial timing session is never silently combined with a new session.
    assert not (out/'started-utc.txt').exists(), 'Use a new timing output for a rerun'
    assert list((w/'preparation-build').glob('codec_preparation*.so'))
    assert (w/'preparation-qdrant/target/release/libqdrant_bench.so').is_file()
    wait_for_other_jobs(out)
    plan = set()
    dimensions = DIMS
    for bits in (1, 2, 4):
        for ds in dimensions:
            for codec in ('rabitq', 'qdrant'):
                plan.add((ds, codec, bits, 0))
                rows = primary_matches(w,bits,codec)
                selected = [r for r in rows if r['dataset'] == ds]
                assert len(selected) == 3
                for row in selected:
                    if row['code_bits']:
                        code = int(row['code_bits'])
                        plan.add((ds, 'tieredq', -(-code//dimensions[ds]), code))
        for row in primary_equal(w,bits):
            code, ds = int(row['code_bits']), row['dataset']
            plan.add((ds, 'tieredq', -(-code//dimensions[ds]), code))
    # A memory-confirmation probe can reach .95 at less storage than the bisection result.
    # Time the smallest fully evaluated storage point as well, across all three lanes.
    candidates=defaultdict(dict)
    for bits in (1,2,4):
        for row in primary_cells(w,bits):
            if row['codec']=='tieredq':
                candidates[row['dataset'],int(row['requested_code_bits'])][int(row['seed'])]=row
    minimum95={}
    for ds in dimensions:
        reached=[]
        for (dataset,code),seeds in candidates.items():
            if dataset==ds and set(seeds)=={0,1,2} and statistics.mean(float(r['recall10']) for r in seeds.values())>=.95:
                reached.append((max(float(r['disk_bytes']) for r in seeds.values()),code,list(seeds.values())))
        if reached:
            disk,code,rows=min(reached,key=lambda point:(point[0],point[1]))
            plan.add((ds,'tieredq',-(-code//dimensions[ds]),code))
            minimum95[ds]=dict(code_bits=code,disk_bytes=disk,memory_rows=rows)
    (w/'recall95-selected.json').write_text(json.dumps(minimum95,indent=2))
    plan = sorted(plan)
    random.Random(4).shuffle(plan)
    (out/'plan.json').write_text(json.dumps(plan, indent=2))
    (out/'host-processes.txt').write_text(subprocess.check_output(
        ['ps', '-eo', 'pid,psr,pcpu,rss,etime,args', '--sort=-pcpu'], text=True))
    start = time.monotonic()
    (out/'started-utc.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
    env = dict(os.environ, PYTHONPATH=f'{B}/build-native:{B}', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OMP_NUM_THREADS='8', BENCH_SEED='0')
    with (out/'run.log').open('a', buffering=1) as log:
        for i, (ds, codec, bits, code) in enumerate(plan):
            wait_for_other_jobs(out)
            dest = out/f'cell-{ds}-{codec}-{bits}'
            dest = Path(str(dest)+(f'-c{code}' if code else '')+'.json')
            cmd = ['taskset', '-c', '0-7', sys.executable, '-u', str(B/'experiments/issue4_current/measure_cell.py'), '--codec', codec,
                   '--datasets', ds, '--bits', str(bits), '--seeds', '0', '--threads', '1', '8',
                   '--batches', '1', '32', '--queries', '1000', '--data', str(w/'data'),
                   '--output', str(out), '--cell', str(dest)]
            if code:
                cmd += ['--code-bits', str(code)]
            log.write(json.dumps(dict(cell=i+1, total=len(plan), command=cmd))+'\n')
            subprocess.run(cmd, env=env, cwd=B, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = json.loads(dest.read_text())
            assert len(rows) == 4 and all(r['queries'] == 1000 and r['seed'] == 0 for r in rows)
            print(f'{i+1}/{len(plan)} {ds} {codec} {bits} {code}: {time.monotonic()-start:.0f}s', flush=True)
    (out/'finished-utc.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out/'exit-code').write_text('0\n')


if __name__ == '__main__':
    main()
