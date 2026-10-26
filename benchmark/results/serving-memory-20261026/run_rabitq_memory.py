"""Measure separate full-data RaBitQ disk and allocated serving RAM at the comparison anchors."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

bench=Path.cwd();sys.path.insert(0,str(bench))
from data import prepare
from measure import measure
from native_build import build_native
meta={'affinity':sorted(os.sched_getaffinity(0)),'compiler_commands':build_native().compiler_commands}
import codec_native
output=bench.parent/'evidence'
base,queries,truth=prepare(bench/'data','openai')
meta['data_manifest']=json.loads((bench/'data/openai-100000-1000-v2.json').read_text())
meta['records']=[]
os.environ['BENCH_SEED']='0'
with threadpool_limits(limits=1):
    for bits in (2,4,8):
        codec_native.set_threads(8)
        tick=time.perf_counter()
        index=codec_native.Index(base,bits,0,False,False)
        build_s=time.perf_counter()-tick
        metrics=index.memory_metrics(str(output/f'rabitq-{bits}.bin'))
        assert metrics['ram_bytes']==sum(metrics['ram_components'].values())
        record={'bits':bits,'build_s':build_s,'encode_s':index.encode_seconds,'memory':metrics,'timings':[]}
        if bits==2:
            for threads,batch in ((8,32),(8,1),(1,1)):
                timing=measure(index,queries,truth,threads,batch)
                record['timings'].append({'threads':threads,'batch':batch}|timing)
        meta['records'].append(record)
        print(record,flush=True)
        del index
(output/'rabitq-memory.json').write_text(json.dumps(meta,indent=2))
