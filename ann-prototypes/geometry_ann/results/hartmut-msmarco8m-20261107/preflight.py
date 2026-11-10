# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py>=3.10", "numpy>=1.26"]
# ///
import csv,hashlib,json,subprocess,tempfile,sys
from pathlib import Path
import h5py,numpy as np
root=Path(tempfile.mkdtemp(prefix='geometry-scale-preflight-'))
rng=np.random.default_rng(42); base=rng.normal(size=(12,3)).astype('float32'); base/=np.linalg.norm(base,axis=1,keepdims=True)
queries=rng.normal(size=(10,3)).astype('float32'); queries/=np.linalg.norm(queries,axis=1,keepdims=True)
truth=np.argsort(((queries[:,None,:].astype('float64')-base[None,:,:])**2).sum(axis=2),axis=1)[:,:10]
source=root/'input.hdf5'
with h5py.File(source,'w') as f:
 f.attrs['distance']='normalized'; f['train']=base; f['test']=queries; f['neighbors']=truth
sha=hashlib.sha256(source.read_bytes()).hexdigest()
repo=Path(sys.argv[1]).resolve(); build=Path(sys.argv[2]).resolve()
subprocess.run(['uv','run','--no-project','--python','3.12',str(repo/'prepare_hdf5_view.py'),str(source),str(root/'data'),'--calibration','5','--test','5','--expected-sha256',sha,'--int8'],check=True)
settings=root/'settings.csv'; settings.write_text('routing,bounds,weight,metadata_cap,vector_cap,scan_order\ngraph,none,0,20,12,centroid\nflat,none,0,20,12,centroid\n')
args=[str(build/'graph_bench'),'--base',str(root/'data/base.rawview'),'--queries',str(root/'data/calibration_queries.fvecs'),'--truth',str(root/'data/calibration_truth.ivecs'),'--nq','5','--variant','V0','--bounds','none','--coarse','2','--children','2','--leaves','4','--settings',str(settings),'--evaluation-queries',str(root/'data/test_queries.fvecs'),'--evaluation-truth',str(root/'data/test_truth.ivecs'),'--evaluation-output',str(root/'heldout.csv'),'--evaluation-nq','5','--repeats','3']
args += ['--int8-base',str(root/'data/base.i8'),'--int8-scales',str(root/'data/scales.f32'),'--int8-offsets',str(root/'data/offsets.f32')]
subprocess.run(args,check=True,stdout=(root/'calibration.csv').open('w'))
rows=list(csv.DictReader((root/'heldout.csv').open())); assert len(rows)==6 and all(float(r['recall10'])==1 for r in rows)
args=['uv','run','--no-project','--python','3.12',str(repo/'compare_faiss.py'),'--base',str(root/'data/base.rawview'),'--queries',str(root/'data/calibration_queries.fvecs'),'--truth',str(root/'data/calibration_truth.ivecs'),'--nq','5','--variant','ivf','--nlist','2','--sample','12','--budgets','1,2','--evaluation-queries',str(root/'data/test_queries.fvecs'),'--evaluation-truth',str(root/'data/test_truth.ivecs'),'--evaluation-output',str(root/'faiss.csv'),'--evaluation-nq','5','--evaluation-repeats','3','--targets','0.95,0.99,0.995']
args += ['--int8-base',str(root/'data/base.i8'),'--int8-scales',str(root/'data/scales.f32'),'--int8-offsets',str(root/'data/offsets.f32')]
subprocess.run(args,check=True,stdout=(root/'faiss-calibration.csv').open('w'))
rows=list(csv.DictReader((root/'faiss.csv').open())); assert len(rows)==3 and all(float(r['recall10'])==1 for r in rows)
print('PREFLIGHT PASSED',root)

args=[str(build/'sq8_oracle'),'--base',str(root/'data/base.rawview'),'--queries',str(root/'data/test_queries.fvecs'),'--truth',str(root/'data/test_truth.ivecs'),'--nq','5','--int8-base',str(root/'data/base.i8'),'--int8-scales',str(root/'data/scales.f32'),'--int8-offsets',str(root/'data/offsets.f32')]
p=subprocess.run(args,check=True,capture_output=True,text=True)
rows=list(csv.DictReader(p.stdout.splitlines())); assert float(rows[0]['recall10'])==1 and int(rows[0]['codes_bytes'])==36
print('SQ8 ORACLE PREFLIGHT PASSED',root)
