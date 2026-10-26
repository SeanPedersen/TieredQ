"""Audit the single-seed core matrix and compare unchanged historical measurements."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT.parent / 'hartmut-full-single-run-20261023/output-full-single-run'
DIMENSIONS = {'sift': (128, 1000000), 'glove': (200, 100000), 'openai': (1536, 100000)}
KEYS = ('dataset', 'codec', 'target_bits', 'threads', 'batch')
INVARIANTS = ( 'effective_bytes_per_vector', 'effective_bits_per_dim', 'build_s', 'encode_vps', 'config')


def load_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    rows = load_rows(ROOT / 'results.csv')
    expected = set(itertools.product(DIMENSIONS, ('tieredq', 'rabitq'), ('1','2','4','8'), ('1','8'), ('1','32')))
    actual = [tuple(row[key] for key in KEYS) for row in rows]
    assert len(actual) == len(set(actual)) == 96 and set(actual) == expected
    for row in rows:
        assert row['seed'] == '0' and row['rerank'] == 'False'
        dimension, count = DIMENSIONS[row['dataset']]
        assert int(row['d']) == dimension and int(row['base_n']) == count and int(row['queries']) == 1000
        assert row['metric'] == ('l2' if row['dataset'] == 'sift' else 'cosine')
        for key in ('build_s','encode_vps','qps','p50_ms','p95_ms','effective_bytes_per_vector'):
            assert math.isfinite(float(row[key])) and float(row[key]) > 0, (actual, key)
        assert 0 <= float(row['recall10']) <= 1 and float(row['p95_ms']) >= float(row['p50_ms'])
        assert math.isclose(float(row['effective_bits_per_dim']), float(row['effective_bytes_per_vector'])*8/dimension)
        expected_config = 'flat-fht-native-bounds-hacc' if row['codec']=='rabitq' else 'fast-native-query-metric-lift' if row['dataset']=='sift' else 'fast-native'
        assert row['config'] == expected_config
    recall_variations = []
    for cell in itertools.product(DIMENSIONS, ('tieredq','rabitq'), ('1','2','4','8')):
        group = [r for r in rows if tuple(r[k] for k in KEYS[:3]) == cell]
        for key in INVARIANTS:
            assert len({r[key] for r in group}) == 1, (cell, key)
        recalls = [float(r['recall10']) for r in group]
        for batch in ('1','32'):
            assert len({r['recall10'] for r in group if r['batch']==batch})==1
        if max(recalls) != min(recalls):
            recall_variations.append({'dataset':cell[0], 'codec':cell[1], 'bits':int(cell[2]), 'minimum':min(recalls), 'maximum':max(recalls), 'spread':max(recalls)-min(recalls)})
    meta = json.loads((ROOT/'meta.json').read_text())
    assert meta['status']=='complete' and not meta['validation_only']
    assert meta['tieredq_commit']=='3c9dbf590209dabf8ecccd3385fd56254779c70d'
    assert meta['cpu_affinity']==list(range(8))
    assert meta['rabitq_search']==dict(container='direct-flat', implicit_row_ids=True, high_accuracy=True, extra_bit_bounds=True)
    assert meta['rabitq_commit']=='ced3763857e7e4e48553951bf66fabc8517bc4f0'
    assert 'profiling_wall_s' not in meta
    assert len(meta['compiler_commands'])==5
    for command in meta['compiler_commands']:
        args=command['arguments']
        assert [a for a in args if a.startswith('-O')][-1]=='-O3'
        assert '-march=native' in args and '-ffp-contract=off' in args
    data=json.loads((ROOT/'data-verification.json').read_text())
    old_data=json.loads((HISTORY/'data-manifest.json').read_text())
    for dataset in DIMENSIONS:
        assert data[dataset]['manifest']==old_data[dataset], dataset
    diagnostic=json.loads((ROOT/'batch-recall-diagnostic.json').read_text())
    assert diagnostic['same_serial_probes_ids_equal_across_native_batches']
    assert diagnostic['same_batched_probes_ids_equal_across_native_batches']
    assert diagnostic['query_top10_sets_changed']==1
    assert len(recall_variations)==1 and recall_variations[0]['dataset']=='sift' and recall_variations[0]['bits']==1
    assert math.isclose(recall_variations[0]['spread'],0.0001,abs_tol=1e-12)
    for name,digest in data['split_indices'].items():
        assert hashlib.sha256((ROOT/'split-indices'/name).read_bytes()).hexdigest()==digest
    provenance=json.loads((ROOT/'provenance.json').read_text())
    assert hashlib.sha256((ROOT/'committed-source.tar').read_bytes()).hexdigest()==provenance['committed_source_tar_sha256']
    source=json.loads((ROOT/'source-verification.json').read_text())
    assert source['native_source_hashes_match_metadata'] and source['harness_hashes_match_metadata']
    assert (ROOT/'exit-code').read_text().strip()=='0'
    historical={tuple(r[k] for k in KEYS):r for r in load_rows(HISTORY/'results.csv')}
    differences=[]
    for r in rows:
        old=historical[tuple(r[k] for k in KEYS)]
        differences.append({k:r[k] for k in KEYS} | {'recall10_delta':float(r['recall10'])-float(old['recall10']), 'bytes_per_vector_old':float(old['effective_bytes_per_vector']), 'bytes_per_vector_current':float(r['effective_bytes_per_vector']), 'qps_current_over_old':float(r['qps'])/float(old['qps']), 'build_current_over_old':float(r['build_s'])/float(old['build_s'])})
    (ROOT/'historical-differences.json').write_text(json.dumps(differences,indent=2)+'\n')
    audit={'rows':96,'unique_rows':96,'configurations':24,'seed':0,'full_datasets':True,'queries':1000,'finite_positive_timings':True,'scheduling_invariants':list(INVARIANTS),'recall_consistent_across_threads':True,'batch_recall_variations':recall_variations,'dataset_manifests_identical_to_issue4':True,'compiler_commands_audited':5,'cpu_affinity':meta['cpu_affinity'],'matrix_wall_s':meta['matrix_wall_s'],'profiling_run':False,'statistical_confidence_claim':False}
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
