"""Run edge-support and oracle exploration after earlier controlled benchmarks release the lock."""
import csv,datetime,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path.cwd(); PREVIOUS=ROOT.parent/'TieredQ-issue44-msmarco8m-20261107'
SIFT=ROOT.parent/'TieredQ-issue44-sift-sq8-20261107'

def status(stage,**extra):
 value=json.dumps(dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2)+'\n'
 temporary=ROOT/'status.tmp';temporary.write_text(value);temporary.replace(ROOT/'status.json')

def checked(args,log):
 with log.open('w') as out:subprocess.run(args,stdout=out,stderr=subprocess.STDOUT,check=True)

def run(name,data_root,nq,caps,heldout=False):
 out=ROOT/'results'/name;out.mkdir()
 settings=out/'settings.csv'
 with settings.open('x') as f:
  w=csv.writer(f);w.writerow(['routing','bounds','weight','metadata_cap','vector_cap','scan_order'])
  for cap in caps:
   w.writerow(['flat','none',0,8256,cap,'centroid'])
   for weight in [.5,1]:w.writerow(['flat','radius',weight,8256,cap,'routing'])
   for weight in [0,.25,.5,.75,1]:w.writerow(['flat','support',weight,8256,cap,'routing'])
 args=['taskset','-c','2',str(ROOT/'build/graph_bench'),'--base',str(data_root/'data/base.rawview'),
 '--queries',str(data_root/'data/calibration_queries.fvecs'),'--truth',str(data_root/'data/calibration_truth.ivecs'),
 '--nq',str(nq),'--variant','V0','--coarse','64','--children','128','--leaves','8192','--iterations','4',
 '--graph-degree','8','--settings',str(settings),'--geometry-diagnostics',str(out/'geometry-diagnostics.csv')]
 for flag,file in [('base','base.i8'),('scales','scales.f32'),('offsets','offsets.f32')]:args+=['--int8-'+flag,str(data_root/file)]
 if heldout:
  args+=['--evaluation-queries',str(data_root/'data/test_queries.fvecs'),'--evaluation-truth',str(data_root/'data/test_truth.ivecs'),
  '--evaluation-nq','750','--evaluation-output',str(out/'heldout.csv'),'--repeats','3','--evaluation-group-bounds','1']
 status('running',model=name,command=args)
 with (out/'calibration.csv').open('x') as stdout,(out/'run.log').open('x') as stderr:
  code=subprocess.run(args,stdout=stdout,stderr=stderr).returncode
 (out/'exit.json').write_text(json.dumps(dict(exit_code=code))+'\n')
 if code:raise RuntimeError(f'{name} failed: {code}')
 rows=list(csv.DictReader((out/'geometry-diagnostics.csv').open()))
 if any(int(r['unsafe_members']) for r in rows):raise RuntimeError('unsafe bound in diagnostic')
 if any(int(r['oracle_vectors_9'])>int(r['ranked_vectors_9']) or int(r['oracle_vectors_10'])>int(r['ranked_vectors_10']) for r in rows):raise RuntimeError('oracle coverage inconsistency')

status('queued_for_benchmark_lock')
exit_code=1
try:
 with (ROOT.parent/'.tieredq-issue44-benchmark.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  status('building_and_validating')
  checked(['cmake','-S',str(ROOT/'source'),'-B',str(ROOT/'build'),'-DCMAKE_BUILD_TYPE=Release'],ROOT/'configure.log')
  checked(['cmake','--build',str(ROOT/'build'),'-j','6'],ROOT/'build.log')
  checked(['ctest','--test-dir',str(ROOT/'build'),'--output-on-failure'],ROOT/'ctest.log')
  checked(['uv','run','--no-project','--python','3.12',str(ROOT/'preflight.py'),str(ROOT/'source'),str(ROOT/'build')],ROOT/'preflight.log')
  (ROOT/'results').mkdir(exist_ok=False)
  hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'source').iterdir() if p.is_file()}
  (ROOT/'source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
  manifest=json.loads((PREVIOUS/'data/manifest.json').read_text());assert manifest['base_shape']==[8840823,1024]
  (ROOT/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  run('sift-sanity',SIFT,50,[35000])
  run('qwen-support',PREVIOUS,250,[35000,176816,442041],heldout=True)
  checked([sys.executable,str(ROOT/'summarize.py'),str(ROOT)],ROOT/'summary.log')
  status('complete');exit_code=0
except Exception as exc:
 status('failed',error=str(exc));raise
finally:
 (ROOT/'job.exitcode').write_text(str(exit_code)+'\n')
