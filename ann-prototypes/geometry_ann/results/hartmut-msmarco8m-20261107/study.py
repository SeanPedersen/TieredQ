"""Full-corpus scale study. Run from the remote experiment root after preparing data/."""
import csv
import fcntl
import datetime
import json
import subprocess
from pathlib import Path

ROOT = Path.cwd()
_lock=(ROOT.parent/'.tieredq-issue44-benchmark.lock').open('a')
fcntl.flock(_lock,fcntl.LOCK_EX)
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=False)
manifest = json.loads((ROOT/'data/manifest.json').read_text())
n, d = manifest['base_shape']
assert (n,d) == (8840823,1024), (n,d)
assert manifest.get('quantization'), 'Missing shared SQ8 dataset'
FIELDS = ['routing','bounds','weight','metadata_cap','vector_cap','scan_order']
MODELS = {
 'adaptive4': ['--variant','V2','--split','kmeans','--iterations','4'],
 'fixed4': ['--variant','V0','--children','128','--iterations','4'],
 'fixed12': ['--variant','V0','--children','128','--iterations','12'],
}

def status(stage, **extra):
    (ROOT/'status.json').write_text(json.dumps(dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2)+'\n')

def settings(path, rows):
    with path.open('x') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

def row(route,meta,cap):
    return dict(routing=route,bounds='none',weight=0,metadata_cap=meta,vector_cap=cap,scan_order='centroid')

caps = sorted({35000,50000,*(int(n*fraction) for fraction in [.01,.02,.035,.05])})
settings(OUT/'calibration-settings.csv', [row(route,meta,cap) for cap in caps
         for route,budgets in [('graph',[512,1024,2048,4096,8256]),('flat',[8256])]
         for meta in budgets])
exit_codes={}
codec_args=['--int8-base',str(ROOT/'base.i8'),'--int8-scales',str(ROOT/'scales.f32'),'--int8-offsets',str(ROOT/'offsets.f32')]
status('running',model='fp32_truth_diagnostic')
with (OUT/'fp32-oracle.csv').open('x') as out,(OUT/'fp32-oracle.log').open('x') as log:
    args=['taskset','-c','2',str(ROOT/'build/sq8_oracle'),'--base','data/base.rawview',
          '--queries','data/calibration_queries.fvecs','--truth','data/calibration_truth.ivecs','--nq','50']
    exit_codes['fp32_oracle']=subprocess.run(args,stdout=out,stderr=log).returncode
status('running',model='sq8_exhaustive_diagnostic')
with (OUT/'sq8-oracle.csv').open('x') as out,(OUT/'sq8-oracle.log').open('x') as log:
    args=['taskset','-c','2',str(ROOT/'build/sq8_oracle'),'--base','data/base.rawview',
          '--queries','data/calibration_queries.fvecs','--truth','data/calibration_truth.ivecs','--nq','50',*codec_args]
    exit_codes['sq8_oracle']=subprocess.run(args,stdout=out,stderr=log).returncode
for model,variant in MODELS.items():
    fixed=[row('graph',2048,50000 if model=='adaptive4' else 35000)]
    if model=='fixed4': fixed.append(row('flat',8256,35000))
    settings(OUT/f'{model}-sift-replay.csv',fixed)
    args=['taskset','-c','2',str(ROOT/'build/graph_bench'),
      '--base','data/base.rawview','--queries','data/calibration_queries.fvecs',
      '--truth','data/calibration_truth.ivecs','--nq','250',
      '--coarse','64','--leaves','8192','--target-size','64','--bounds','none','--graph-degree','8',
      '--settings',str(OUT/'calibration-settings.csv'),
      '--evaluation-queries','data/test_queries.fvecs','--evaluation-truth','data/test_truth.ivecs',
      '--evaluation-output',str(OUT/f'{model}-heldout.csv'),'--evaluation-nq','750','--repeats','3',
      '--evaluation-fixed-settings',str(OUT/f'{model}-sift-replay.csv'),*codec_args,*variant]
    status('running',model=model,command=args)
    with (OUT/f'{model}-calibration.csv').open('x') as out,(OUT/f'{model}.log').open('x') as log:
        exit_codes[model]=subprocess.run(args,stdout=out,stderr=log).returncode
    (OUT/f'{model}-exit.json').write_text(json.dumps(dict(exit_code=exit_codes[model]))+'\n')

args=['taskset','-c','2','uv','run','--no-project','--python','3.12','source/compare_faiss.py',
 '--base','data/base.rawview','--queries','data/calibration_queries.fvecs',
 '--truth','data/calibration_truth.ivecs','--nq','250','--nlist','8192','--sample','327680',
 '--variant','ivf','--budgets','32,64,96,128,192,256,512,1024',
 '--targets','0.95,0.99,0.995','--evaluation-queries','data/test_queries.fvecs',
 '--evaluation-truth','data/test_truth.ivecs','--evaluation-output',str(OUT/'faiss-heldout.csv'),
 '--evaluation-nq','750','--evaluation-repeats','3']
for reference, extra in [('faiss-sq8',codec_args),('faiss-fp32',[])]:
    command=args+extra
    command[command.index('--evaluation-output')+1]=str(OUT/f'{reference}-heldout.csv')
    status('running',model=reference,command=command)
    with (OUT/f'{reference}-calibration.csv').open('x') as out,(OUT/f'{reference}.log').open('x') as log:
        exit_codes[reference]=subprocess.run(command,stdout=out,stderr=log).returncode
status('complete' if not any(exit_codes.values()) else 'finished_with_failures',exit_codes=exit_codes)
raise SystemExit(int(any(exit_codes.values())))
