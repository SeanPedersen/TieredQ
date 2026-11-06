"""Confirm equal-recall budgets on three seeds and measure selected budgets in fresh processes.

Bisection is exploratory: reports describe the smallest evaluated storage, not a proof of a global minimum.
All probes and seed recalls are retained. The target .95 is evaluated alongside reference levels.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys

B = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(B), str(B/'experiments/issue4_1m')]
import importlib.util
spec = importlib.util.spec_from_file_location('issue4_bisection', B/'experiments/issue4_1m/equal_recall.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
recall_at, smallest_budget, RESOLUTION = legacy.recall_at, legacy.smallest_budget, legacy.RESOLUTION
from match_memory import Cells
from data import prepare
import codec_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('bits', type=int)
    parser.add_argument('datasets', nargs='+')
    args = parser.parse_args()
    w, bits = args.workspace.resolve(), args.bits
    out = w/f'equal-recall/b{bits}'
    out.mkdir(parents=True, exist_ok=True)
    cells = Cells(w/f'match-b{bits}', dict(os.environ), w/'data', True)
    codec_native.set_threads(8)
    for dataset in args.datasets:
        base, queries, truth = prepare(w/'data', dataset)
        d = base.shape[1]
        targets = {}
        for codec in ('rabitq','qdrant'):
            rows = [cells.measure(dataset,codec,bits,seed) for seed in (0,1,2)]
            targets[codec] = statistics.mean(float(r['recall10']) for r in rows)
        if bits == 4:
            targets['recall95'] = .95
        for codec, target in targets.items():
            destination = out/f'{dataset}-{codec}.json'
            if destination.exists():
                continue
            cache = {}
            probe_path = out/f'{dataset}-{codec}-probes.jsonl'
            def evaluate(code, seed=0):
                key = code, seed
                if key not in cache:
                    value = recall_at(base, queries, truth, dataset=='sift',code,seed)
                    cache[key] = value
                    with probe_path.open('a') as f:
                        f.write(json.dumps(dict(code_bits=code,seed=seed,recall10=value))+'\n')
                return cache[key]
            # Verify the upper endpoint; never call an unreached endpoint a match.
            upper = min(8*d, 2*bits*d+RESOLUTION)
            while evaluate(upper) < target and upper < 8*d:
                upper = min(8*d, upper+d)
            code, _, _ = smallest_budget(evaluate,target,upper)
            while True:
                recalls = [evaluate(code,seed) for seed in (0,1,2)]
                if statistics.mean(recalls) >= target or code >= 8*d:
                    break
                code = min(8*d, code+RESOLUTION)
            # Fresh-process build/reload/serve memory at the final evaluated budget, all seeds.
            rows = [cells.measure(dataset,'tieredq',-(-code//d),seed,code) for seed in (0,1,2)]
            measured_recalls = [float(r['recall10']) for r in rows]
            assert all(abs(a-b) < 1e-10 for a,b in zip(recalls, measured_recalls)), (recalls,measured_recalls)
            result = dict(dataset=dataset,reference_codec=codec,reference_bits=bits,
                          reference_recall10=target,code_bits=code,recall10_seed0=recalls[0],
                          recall10_mean=statistics.mean(recalls),recall10_std=statistics.stdev(recalls),
                          reached=statistics.mean(recalls)>=target,memory_rows=rows)
            destination.write_text(json.dumps(result,indent=2))
            print(result,flush=True)
        del base, queries, truth

if __name__ == '__main__':
    main()
