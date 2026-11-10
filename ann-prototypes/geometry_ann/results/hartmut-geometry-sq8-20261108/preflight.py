# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py>=3.10", "numpy>=1.26"]
# ///
"""Exercise INT8 geometry settings, complete-search equivalence and diagnostic output."""
import hashlib, csv, subprocess, sys, tempfile
from pathlib import Path
import h5py, numpy as np
source,build=map(lambda s:Path(s).resolve(),sys.argv[1:3])
root=Path(tempfile.mkdtemp(prefix='geometry-sq8-preflight-'))
rng=np.random.default_rng(71)
base=rng.normal(size=(257,17)).astype('float32')*np.geomspace(.03,2,17).astype('float32')
queries=rng.normal(size=(12,17)).astype('float32')
truth=np.argsort(((queries[:,None,:].astype('float64')-base[None,:,:])**2).sum(axis=2),axis=1)[:,:10]
with h5py.File(root/'input.hdf5','w') as f:
 f['train']=base; f['test']=queries; f['neighbors']=truth
subprocess.run(['uv','run','--no-project','--python','3.12',str(source/'prepare_hdf5_view.py'),str(root/'input.hdf5'),str(root/'data'),'--expected-sha256',hashlib.sha256((root/'input.hdf5').read_bytes()).hexdigest(),'--int8','--calibration','6','--test','6'],check=True,stdout=subprocess.DEVNULL)
settings=root/'settings.csv'
with settings.open('w') as f:
 w=csv.writer(f);w.writerow(['routing','bounds','weight','metadata_cap','vector_cap','scan_order'])
 for route in ['flat','graph','hierarchy']:
  for mode in ['none','radius','shared','shared-residual']:
   for weight in ([0] if mode=='none' else [.5,1]):w.writerow([route,mode,weight,64,257,'routing'])
cmd=[str(build/'graph_bench'),'--base',str(root/'data/base.rawview'),'--queries',str(root/'data/calibration_queries.fvecs'),'--truth',str(root/'data/calibration_truth.ivecs'),'--nq','6','--variant','V0','--coarse','4','--children','4','--leaves','16','--shared-directions','8','--settings',str(settings),'--geometry-diagnostics',str(root/'diagnostics.csv'),'--evaluation-queries',str(root/'data/test_queries.fvecs'),'--evaluation-truth',str(root/'data/test_truth.ivecs'),'--evaluation-output',str(root/'heldout.csv'),'--evaluation-nq','6','--repeats','1']
for flag,name in [('base','base.i8'),('scales','scales.f32'),('offsets','offsets.f32')]:cmd+=['--int8-'+flag,str(root/'data'/name)]
with (root/'calibration.csv').open('w') as out:subprocess.run(cmd,check=True,stdout=out)
rows=list(csv.DictReader((root/'calibration.csv').open()))
assert len(rows)==21 and len({r['recall10'] for r in rows})==1,rows
assert all(int(r['max_vectors'])<=257 and int(r['max_metadata'])<=64 for r in rows)
diag=list(csv.DictReader((root/'diagnostics.csv').open()))
assert len(diag)==60 and all(int(r['unsafe_members'])==0 for r in diag)
assert any(int(r['positive_bounds'])>0 for r in diag if r['bounds']=='shared-residual')
print('GEOMETRY INT8 PREFLIGHT PASSED',root)
