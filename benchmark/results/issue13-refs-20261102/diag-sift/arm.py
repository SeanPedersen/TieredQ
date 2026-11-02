import sys,os
B=os.path.expanduser('~/experiments/TieredQ-issue13-refs-20261102/diag/source/benchmark'); sys.path[:0]=[B,B+'/build-native',B+'/experiments/issue4_1m']
from pathlib import Path
import codec_native
from data import prepare
from equal_recall import recall_at
codec_native.set_threads(8)
arm,ds=sys.argv[1],sys.argv[2]
data=Path(os.path.expanduser('~/experiments/TieredQ-issue4-equal-memory-after-20261028/data'))
base,queries,truth=prepare(data,ds)
for code in map(int,sys.argv[3:]):
    r=[recall_at(base,queries,truth,ds=='sift',code,s) for s in (0,1,2)]
    print(arm,ds,code,'%.4f'%(sum(r)/3),' '.join('%.4f'%x for x in r),flush=True)
