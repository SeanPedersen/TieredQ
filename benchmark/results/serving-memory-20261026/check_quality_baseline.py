"""Verify the old representation's actual storage at the selected larger code budget."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

bench=Path.cwd();sys.path.insert(0,str(bench))
from data import prepare
import codec_native
selection=Path(sys.argv[1]) if len(sys.argv)>1 else bench.parent.parent/'sequential/evidence/quality.json'
report=json.loads(selection.read_text())
selected=report['selected'] if 'selected' in report else report['records'][0]
result_name='dual-budget-baseline' if len(sys.argv)>1 else 'quality-baseline'
bits=selected['code_bits']
base,queries,truth=prepare(bench/'data','openai')
with threadpool_limits(limits=1):
    codec_native.set_threads(8)
    tick=time.perf_counter()
    index=codec_native.Index(base,(bits+1535)//1536,0,True,False,True,bits)
    result={'code_bits':bits,'build_s':time.perf_counter()-tick,'encode_s':index.encode_seconds}
    result['allocation']=index.allocations(str(bench.parent/f'evidence/{result_name}.bin'))
    ids=np.empty((1000,10),dtype=np.uint32);index.search(queries,ids,8)
    result['ids_sha256']=hashlib.sha256(ids.tobytes()).hexdigest()
    assert result['ids_sha256']==selected['ids_sha256']
    result['bytes_per_vector']=max(result['allocation']['resident_bytes'],result['allocation']['serialized_bytes'])/len(base)
    (bench.parent/f'evidence/{result_name}.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)
