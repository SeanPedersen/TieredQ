"""Fresh-process reference build peaks, including inputs; verify cached seed-0 recall/storage."""
import argparse
import json
import os
from pathlib import Path
import sys

B = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(B))
from match_memory import Cells

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('workspace', type=Path)
p.add_argument('bits', type=int)
p.add_argument('datasets', nargs='+')
a = p.parse_args()
out = a.workspace/f'reference-build-b{a.bits}'
out.mkdir(exist_ok=True)
fresh = Cells(out, dict(os.environ), a.workspace/'data', True)
cached = Cells(a.workspace/f'match-b{a.bits}', {}, a.workspace/'data', True)
for ds in a.datasets:
    for codec in ('rabitq', 'qdrant'):
        old = cached.measure(ds, codec, a.bits, 0)
        new = fresh.measure(ds, codec, a.bits, 0)
        for key in ('disk_bytes', 'recall10'):
            assert float(old[key]) == float(new[key]), (ds, codec, key, old[key], new[key])
        print(ds, codec, a.bits, 'cached seed-0 recall and disk verified', flush=True)
(out/'verified.json').write_text(json.dumps(dict(bits=a.bits, datasets=a.datasets)))
