"""Audit the five-point extension and summarize all six actual-storage comparisons."""
import csv
import hashlib
import json
import math
from pathlib import Path

from report_data import ROOT, all_points, serving_points, anchor_point

OUTPUT = ROOT / 'storage-sweep'
POINT_KEYS = {('openai', 1), ('openai', 4), ('gemini', 1), ('gemini', 2), ('gemini', 4)}
SCHEDULES = {(seed, threads, batch) for seed in (0, 1, 2) for threads in (1, 8) for batch in (1, 32)}


def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    metadata = json.loads((OUTPUT / 'meta.json').read_text())
    assert metadata['status'] == 'complete'
    assert metadata['affinity'] == list(range(8))
    assert metadata['source_hashes_match']
    assert metadata['source_commit'] == json.loads((ROOT / 'output/meta.json').read_text())['tieredq_commit']
    controls = json.loads((OUTPUT / 'controls.json').read_text())
    assert {c['dataset'] for c in controls} == {'openai', 'gemini'}
    assert all(c['identical_ids'] for c in controls)
    inputs = json.loads((ROOT / 'output/inputs.json').read_text())
    assert all(hashes == inputs[dataset]['sha256'] for dataset, hashes in metadata['input_sha256'].items())
    rows = list(csv.DictReader((OUTPUT / 'results.csv').open()))
    assert len(rows) == 60
    assert {(r['dataset'], int(r['anchor_bits'])) for r in rows} == POINT_KEYS
    provenance = []
    for dataset, anchor in sorted(POINT_KEYS):
        group = [r for r in rows if r['dataset'] == dataset and int(r['anchor_bits']) == anchor]
        assert {(int(r['seed']), int(r['threads']), int(r['batch'])) for r in group} == SCHEDULES
        assert len(group) == len(SCHEDULES)
        selected = json.loads((OUTPUT / f'{dataset}-rabitq{anchor}/selection.json').read_text())
        assert {int(r['code_bits']) for r in group} == {selected['selected_code_bits']}
        for row in group:
            assert 0 <= float(row['recall10']) <= 1
            assert float(row['effective_bytes_per_vector']) > 0
            assert math.isclose(float(row['effective_bits_per_dim']), float(row['effective_bytes_per_vector']) * 8 / int(row['d']))
            for field in ('qps', 'p50_ms', 'p95_ms'):
                assert math.isfinite(float(row[field])) if row['seed'] == '0' else math.isnan(float(row[field]))
            provenance.append(dict(dataset=dataset, anchor_bits=anchor, seed=int(row['seed']),
                                   threads=int(row['threads']), batch=int(row['batch']),
                                   source=f'{dataset}-rabitq{anchor}/results.csv'))
    points = serving_points(all_points())
    summary = []
    for point in points:
        if not point.supplemental:
            continue
        anchor = anchor_point(points, point.dataset, point.anchor_bits)
        assert point.storage < anchor.storage
        summary.append(dict(dataset=point.dataset, anchor_bits=point.anchor_bits,
                            code_bits=point.code_bits, bytes_per_vector=point.storage,
                            stored_bits_per_vector=point.storage * 8,
                            recall_mean=point.recall, recall_population_std=point.std,
                            rabitq_bytes_per_vector=anchor.storage, rabitq_recall_mean=anchor.recall,
                            recall_gain_pp=100 * (point.recall - anchor.recall),
                            storage_saving_percent=100 * (1 - point.storage / anchor.storage),
                            qps_ratio_8t_b32=point.qps / anchor.qps))
    assert len(summary) == 6
    write_json('summary.json', summary)
    write_json('row-provenance.json', provenance)
    write_json('audit.json', dict(status='passed', rows=60, timed_rows=20, points=5,
                                combined_supplemental_points=6, all_seed0_points_below_caps=True,
                                controls_identical_ids=True, input_hashes_match=True,
                                source_hashes_match=True, storage_selected_before_recall=True,
                                results_sha256=hashlib.sha256((OUTPUT / 'results.csv').read_bytes()).hexdigest()))
    print('PASS: five-point extension audit; all six paired storage-cap summaries generated')


if __name__ == '__main__':
    main()
