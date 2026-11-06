"""Prepare current-main issue #4 inputs and provenance without modifying caches."""
import csv
import hashlib
import json
from pathlib import Path
import sys

B = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(B))
from measure_memory import FIELDS
from match_memory import Cells

W = Path(sys.argv[1]).resolve()
R = B / 'results'
SOURCES = {
    'sift': [R/'issue9-ship-20261030/match-sift/cells.csv'],
    'glove': [R/'issue9-ship-20261030/match-a/cells.csv'],
    'openai': [R/'issue9-ship-20261030/match-b/cells.csv'],
    'gemini': [R/'issue9-ship-20261030/match-b/cells.csv'],
    'openai1m': [R/f'issue4-1m-20261030/match-b{b}/cells.csv' for b in (1,2,4)],
    'msmarco1m': [R/f'issue22-msmarco1m-flat-20261103/match-b{b}/cells.csv' for b in (1,2,4)],
}
# A completed current bundle can seed a fresh rerun without the historical result trees.
if len(sys.argv)>2:
    reference_bundle=Path(sys.argv[2]).resolve()
    SOURCES={dataset:[reference_bundle/f'match-b{bits}/cells.csv' for bits in (1,2,4)] for dataset in SOURCES}
DATA = [Path.home()/'experiments'/p/'data' for p in (
    'TieredQ-issue4-equal-memory-after-20261028', 'TieredQ-issue4-1m-20261030',
    'TieredQ-issue22-msmarco-20261031')]
DATA += [Path.home()/'experiments/TieredQ-issue4-precheck-20261105/cache', W/'recovered-cache']
(W/'data').mkdir(exist_ok=True)
provenance = {}
for root in DATA:
    for src in sorted(root.iterdir()):
        if not src.exists() or src.suffix not in ('.npz', '.json', '.npy'):
            continue
        dst = W/'data'/src.name
        if not dst.exists():
            dst.symlink_to(src)
        assert dst.resolve() == src.resolve()
        if src.suffix == '.npz':
            digest = hashlib.file_digest(src.open('rb'), 'sha256').hexdigest()
            provenance[str(src)] = {'sha256': digest, 'bytes': src.stat().st_size}
(W/'input-files.json').write_text(json.dumps(provenance, indent=2))
required = ['sift-1000000-1000-v2.npz', 'glove-100000-1000-v2.npz', 'openai-100000-1000-v2.npz',
            'gemini-prepared.npz', 'openai1m-999000-1000-v2.npz', 'msmarco1m-1000000-1000-v1.npz']
assert all((W/'data'/name).exists() for name in required), 'Missing prepared caches; do not launch or download'
reference_sources = {}
for bits in (1,2,4):
    out = W/f'match-b{bits}'
    out.mkdir(exist_ok=True)
    assert not (out/'cells.csv').exists(), 'Refuse to overwrite existing measurements'
    rows = {}
    for ds, sources in SOURCES.items():
        for source in sources:
            reference_sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
            for row in csv.DictReader(source.open()):
                if row['dataset'] == ds and row['codec'] in ('rabitq','qdrant') and int(row['target_bits']) == bits:
                    row = {field: row.get(field, '') for field in FIELDS} | {'requested_code_bits':'0'}
                    rows[(ds,row['codec'],int(row['seed']))] = row
    assert len(rows) == 6*2*3, (bits,len(rows))
    with (out/'cells.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS+['requested_code_bits'])
        writer.writeheader()
        writer.writerows(rows.values())
    loaded = Cells(out, {}, W/'data', True)
    assert len(loaded.rows) == len(rows)
    for row in rows.values():
        cached = loaded.measure(row['dataset'], row['codec'], bits, int(row['seed']))
        assert cached['recall10'] == row['recall10']
    print(out, len(rows), 'reference cells; current-header reload verified', flush=True)
(W/'reference-sources.json').write_text(json.dumps(reference_sources, indent=2))
