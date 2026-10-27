"""Fair-storage comparison: give TieredQ the same total bytes/vector as SAQ.

SAQ's effective storage includes its PCA basis and per-vector factors, so its
nominal 1/2/4-bit requests consume more bytes than TieredQ's nominal budgets.
For each SAQ cell this selects the integer TieredQ code budget whose measured
serving storage lands in the 99.0-99.5% band below the SAQ cap, then evaluates
three seeds with the audited TieredQ harness.
"""
import csv
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits

REFRESH = Path('/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542')
sys.path.insert(0, str(REFRESH / 'three-codec'))
import evaluation
import runtime
import selection

HERE = Path(__file__).resolve().parent
OUT = HERE / 'tieredq-matched'
DATASETS = ('gemini', 'glove', 'openai', 'sift')
BITS = (1, 2, 4)
SEEDS = (0, 1, 2)


def write(rows):
    with (OUT / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_anchors(dataset, dimensions):
    rows = [
        row for row in csv.DictReader((REFRESH / 'output/results.csv').open())
        if row['dataset'] == dataset and row['codec'] == 'tieredq'
        and row['seed'] == '0' and row['threads'] == '8' and row['batch'] == '32'
    ]
    anchors = [
        dict(code_bits=round(float(row['target_bits']) * dimensions),
             bytes=float(row['effective_bytes_per_vector']))
        for row in rows
    ]
    anchors.sort(key=lambda anchor: anchor['code_bits'])
    return anchors


def main():
    OUT.mkdir(exist_ok=True)
    saq = list(csv.DictReader((HERE / 'results.csv').open()))
    rows = []
    for dataset in DATASETS:
        group = [row for row in saq if row['dataset'] == dataset]
        if not group:
            print('NO SAQ', dataset, flush=True)
            continue
        base, queries, truth = runtime.load(dataset)
        dimensions = base.shape[1]
        anchors = reference_anchors(dataset, dimensions)
        assert len(anchors) >= 2
        for bits in BITS:
            cell = [row for row in group if int(row['target_bits']) == bits]
            if not cell:
                print('NO SAQ', dataset, bits, flush=True)
                continue
            seed0 = next(row for row in cell if row['seed'] == '0')
            cap = float(seed0['effective_bytes_per_vector'])
            directory = OUT / f'{dataset}-saq{bits}'
            directory.mkdir(exist_ok=True)
            selected = selection.select_budget(
                cap, anchors,
                lambda budget: runtime.build(base, budget, 0, directory, dataset == 'sift'),
                lambda attempt: print('STORAGE', dataset, bits, attempt, flush=True),
                8 * dimensions)
            runtime.save(directory / 'selection.json', dict(
                dataset=dataset, anchor_bits=bits, saq_cap_bytes=cap,
                selected_code_bits=selected.code_bits,
                selected_bytes_per_vector=selected.built.bytes_per_vector,
                in_target_band=selected.in_target_band, attempts=selected.attempts,
                policy='Match SAQ total bytes/vector; 99.0-99.5% band; seed-0 storage only'))
            for seed in SEEDS:
                built = selected.built if seed == 0 else runtime.build(
                    base, selected.code_bits, seed, directory, dataset == 'sift')
                common = dict(
                    dataset=dataset, d=dimensions, base_n=len(base), queries=len(queries),
                    metric='l2' if dataset == 'sift' else 'cosine',
                    codec='tieredq', config='fast-native-exact-code-budget-matched-to-saq',
                    target_bits=bits, code_bits=selected.code_bits, anchor_bits='',
                    saq_cap_bytes=cap,
                    effective_bits_per_dim=built.bytes_per_vector * 8 / dimensions,
                    effective_bytes_per_vector=built.bytes_per_vector,
                    build_s=built.build_s, encode_vps=built.encode_vps,
                    rerank=False, seed=seed)
                rows += evaluation.evaluate(built.index, queries, truth, common, directory)
                write(rows)
                if seed == 0:
                    selected.built = None
                del built
        del base, queries, truth
    runtime.save(OUT / 'meta.json', dict(
        status='complete', rows=len(rows),
        policy='TieredQ code budget selected to match each SAQ effective bytes/vector within 99.0-99.5%'))
    print('MATCHED COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
