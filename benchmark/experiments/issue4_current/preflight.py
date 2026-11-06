"""Require every prepared input and forbid network fallback before starting workers."""
import hashlib
import difflib
import json
from pathlib import Path
import sys
import numpy as np

w=Path(sys.argv[1]).resolve()
configs={'sift-1000000-1000-v2.npz':(1000000,128), 'glove-100000-1000-v2.npz':(100000,200),
         'openai-100000-1000-v2.npz':(100000,1536), 'gemini-prepared.npz':(99000,768),
         'openai1m-999000-1000-v2.npz':(999000,1536), 'msmarco1m-1000000-1000-v1.npz':(1000000,1024)}
if (w/'data/glove1m-1000000-1000-v1.npz').exists():
    configs['glove1m-1000000-1000-v1.npz']=(1000000,200)
records={}
for name,(n,d) in configs.items():
    path=w/'data'/name
    assert path.is_file(),path
    with np.load(path) as arrays:
        shapes={key:list(arrays[key].shape) for key in arrays.files}
        assert shapes['base']==[n,d],(name,shapes)
        # MS MARCO stores both query sets in the same prepared cache.
        if name.startswith('msmarco'):
            assert any(shape==[1000,d] for shape in shapes.values()),shapes
        else:
            assert shapes['queries']==[1000,d] and shapes['truth']==[1000,10],(name,shapes)
    records[name]=dict(source=str(path.resolve()),bytes=path.stat().st_size,shapes=shapes,
                       sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest())
(w/'input-files.json').write_text(json.dumps(records,indent=2))
# Guard every archived data.download call. Prepared caches must always short-circuit this function.
p=w/'source/benchmark/data.py'
old=p.read_text()
needle='    partial = target.with_suffix(".partial")\n'
guard='    raise RuntimeError("issue4 current run forbids downloads; prepared cache missing: " + str(target))\n'
if guard not in old:
    assert needle in old
    new=old.replace(needle,guard+needle)
    p.write_text(new)
    (w/'network-guard.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
        fromfile='a/benchmark/data.py',tofile='b/benchmark/data.py')))
print(f'{len(records)} prepared input shapes and SHA256 hashes verified; downloads disabled',flush=True)
