"""Controlled shape metadata exploration on immutable fixed4 INT8 partitions."""
import csv,datetime,fcntl,hashlib,json,subprocess
from pathlib import Path
ROOT=Path.cwd(); PREVIOUS=ROOT.parent/'TieredQ-issue44-msmarco8m-20261107'
SIFT=ROOT.parent/'TieredQ-issue44-sift-sq8-20261107'

def status(stage,**extra):
 (ROOT/'status.json').write_text(json.dumps(dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2)+'\n')

def settings(path,caps,graph=False):
 with path.open('x') as f:
  w=csv.writer(f);w.writerow(['routing','bounds','weight','metadata_cap','vector_cap','scan_order'])
  for cap in caps:
   w.writerow(['flat','none',0,8256,cap,'centroid'])
   for mode in ['radius','shared','shared-residual']:
    for weight in [.5,1]:w.writerow(['flat',mode,weight,8256,cap,'routing'])
   if graph:
    for meta in [2048,4096]:
     w.writerow(['graph','none',0,meta,cap,'centroid'])
     for weight in [.5,1]:w.writerow(['graph','shared-residual',weight,meta,cap,'routing'])

def run(name,data_root,directions,nq,caps,heldout=False,graph=False):
 out=ROOT/'results'/name; out.mkdir()
 grid=out/'settings.csv';settings(grid,caps,graph)
 args=['taskset','-c','2',str(ROOT/'build/graph_bench'),'--base',str(data_root/'data/base.rawview'),
 '--queries',str(data_root/'data/calibration_queries.fvecs'),'--truth',str(data_root/'data/calibration_truth.ivecs'),
 '--nq',str(nq),'--variant','V0','--coarse','64','--children','128','--leaves','8192','--iterations','4',
 '--graph-degree','8','--shared-directions',str(directions),'--shared-sample','4096','--settings',str(grid),
 '--geometry-diagnostics',str(out/'geometry-diagnostics.csv')]
 for flag,file in [('base','base.i8'),('scales','scales.f32'),('offsets','offsets.f32')]:args+=['--int8-'+flag,str(data_root/file)]
 if heldout:
  args+=['--evaluation-queries',str(data_root/'data/test_queries.fvecs'),'--evaluation-truth',str(data_root/'data/test_truth.ivecs'),
  '--evaluation-nq','750','--evaluation-output',str(out/'heldout.csv'),'--repeats','3']
 status('running',model=name,directions=directions,command=args)
 with (out/'calibration.csv').open('x') as stdout,(out/'run.log').open('x') as stderr:
  code=subprocess.run(args,stdout=stdout,stderr=stderr).returncode
 (out/'exit.json').write_text(json.dumps(dict(exit_code=code))+'\n')
 if code:raise RuntimeError(f'{name} failed: {code}')
 rows=list(csv.DictReader((out/'geometry-diagnostics.csv').open()))
 if any(int(r['unsafe_members']) for r in rows):raise RuntimeError(f'{name}: unsafe bound in diagnostic')
 return out

with (ROOT.parent/'.tieredq-issue44-benchmark.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 (ROOT/'results').mkdir(exist_ok=False)
 manifest=json.loads((PREVIOUS/'data/manifest.json').read_text())
 assert manifest['base_shape']==[8840823,1024]
 (ROOT/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'source').iterdir() if p.is_file()}
 (ROOT/'source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
 try:
  # Small control first; all following trials retain the entire 8.84M corpus.
  run('sift-sanity-k8',SIFT,8,50,[35000],graph=False)
  for k in [8,16,32]:
   run(f'qwen-k{k}',PREVIOUS,k,250,[35000,176816,442041],heldout=True,graph=True)
  status('complete')
 except Exception as exc:
  status('failed',error=str(exc));raise
