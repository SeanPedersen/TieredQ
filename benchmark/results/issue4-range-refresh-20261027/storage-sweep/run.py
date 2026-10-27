"""Measure the five remaining OpenAI/Gemini points below actual RaBitQ storage caps."""
from datetime import datetime, timezone
import json
from threadpoolctl import threadpool_limits

from runtime import OUTPUT, ROOT, load_dataset, reference_rows, verify_environment, verify_control, build, write_json
from selection import select_budget, TARGET_LOW, TARGET_HIGH, MAX_ATTEMPTS
from evaluation import evaluate, write_rows

POINTS = {'openai': (1, 4), 'gemini': (1, 2, 4)}
MAIN_BUDGETS = (1, 2, 4)


def baseline(rows, dataset, codec, bits):
    return next(r for r in rows if (r['dataset'], r['codec'], r['target_bits'], r['seed'], r['threads'], r['batch']) ==
                (dataset, codec, str(bits), '0', '8', '32'))


def measure_point(dataset, anchor_bits, arrays, references):
    directory = OUTPUT / f'{dataset}-rabitq{anchor_bits}'
    directory.mkdir(exist_ok=True)
    base, queries, truth = arrays
    cap = float(baseline(references, dataset, 'rabitq', anchor_bits)['effective_bytes_per_vector'])
    anchors = [dict(code_bits=base.shape[1] * bits,
                    bytes=float(baseline(references, dataset, 'tieredq', bits)['effective_bytes_per_vector']))
               for bits in MAIN_BUDGETS]
    print(f'SELECT {dataset} RaBitQ {anchor_bits}bit cap={cap:.8f} B/vector', flush=True)
    def record(attempt):
        print(f'STORAGE {dataset} cap={anchor_bits}: {attempt}', flush=True)
    selected = select_budget(cap, anchors, lambda budget: build(base, budget, 0, directory), record)
    selection = dict(dataset=dataset, anchor_bits=anchor_bits, cap_bytes=cap,
                     policy='Storage only; affine initial estimate from first two main budgets, then integer bisection',
                     target_low=TARGET_LOW, target_high=TARGET_HIGH, max_attempts=MAX_ATTEMPTS,
                     attempts=selected.attempts, selected_code_bits=selected.code_bits,
                     in_target_band=selected.in_target_band,
                     selected_bytes_per_vector=selected.built.bytes_per_vector)
    write_json(directory / 'selection.json', selection)
    rows = evaluate(dataset, base, queries, truth, anchor_bits, selected, directory)
    assert len(rows) == 12
    assert float(rows[0]['effective_bytes_per_vector']) <= cap
    write_json(directory / 'audit.json', dict(status='passed', rows=len(rows), timed_rows=4,
               seed0_within_cap=True, storage_selected_before_recall=True))
    return rows


def main():
    OUTPUT.mkdir(exist_ok=True)
    metadata = verify_environment()
    metadata.update(status='running', points=POINTS, target_low=TARGET_LOW, target_high=TARGET_HIGH)
    write_json(OUTPUT / 'meta.json', metadata)
    references = reference_rows()
    all_rows = []
    controls = []
    inputs = {}
    with threadpool_limits(limits=1):
        for dataset, caps in POINTS.items():
            base, queries, truth, hashes = load_dataset(dataset)
            inputs[dataset] = hashes
            print(f'INPUT VERIFIED {dataset} base={base.shape}, queries={queries.shape}', flush=True)
            controls.append(verify_control(dataset, base, queries, OUTPUT, references))
            write_json(OUTPUT / 'controls.json', controls)
            print(f'CONTROL VERIFIED {dataset}: identical archived IDs and storage', flush=True)
            for anchor_bits in caps:
                all_rows.extend(measure_point(dataset, anchor_bits, (base, queries, truth), references))
                write_rows(OUTPUT / 'results.csv', all_rows)
    metadata.update(status='complete', input_sha256=inputs, controls=controls,
                    rows=len(all_rows), timed_rows=sum(r['seed'] == 0 for r in all_rows),
                    completed=datetime.now(timezone.utc).isoformat())
    write_json(OUTPUT / 'meta.json', metadata)
    print(f'COMPLETE: {len(all_rows)} rows, five new storage-selected points', flush=True)


if __name__ == '__main__':
    main()
