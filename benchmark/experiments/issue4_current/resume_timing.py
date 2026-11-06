"""Explicitly resume an interrupted timing plan, retaining the interruption provenance."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from timing import B, wait_for_other_jobs

w=Path(sys.argv[1]).resolve()
out=w/'timing'
assert not (out/'exit-code').exists()
frozen=json.loads((w/'frozen-native.json').read_text())
assert hashlib.file_digest(Path(frozen['module']).open('rb'),'sha256').hexdigest()==frozen['sha256']
plan=json.loads((out/'plan.json').read_text())
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
metadata=dict(resumed_utc=stamp,reason='Original process disappeared without exit marker; cause unknown',
              boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              source_revision=(w/'source/SOURCE_REVISION').read_text().strip(),completed_before_resume=[])
env=dict(os.environ,PYTHONPATH=f'{B}/build-native:{B}',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='8',BENCH_SEED='0')
start=time.monotonic()
with (out/'run.log').open('a',buffering=1) as log:
    log.write(json.dumps(dict(event='resume',**metadata))+'\n')
    for i,(ds,codec,bits,code) in enumerate(plan):
        dest=out/(f'cell-{ds}-{codec}-{bits}'+(f'-c{code}' if code else '')+'.json')
        profile=w/'preparation'/dest.name
        if dest.exists() and profile.exists():
            rows=json.loads(dest.read_text())
            diagnostic=json.loads(profile.read_text())
            assert len(rows)==4 and {(r['threads'],r['batch']) for r in rows}=={(1,1),(1,32),(8,1),(8,32)}
            assert all(r['queries']==1000 and r['cpu_affinity']==list(range(8)) for r in rows)
            assert diagnostic['available'] and diagnostic['search_ids_equal']
            metadata['completed_before_resume'].append(dest.name)
            continue
        (out/'resume.json').write_text(json.dumps(metadata,indent=2))
        wait_for_other_jobs(out)
        cmd=['taskset','-c','0-7',sys.executable,'-u',str(B/'experiments/issue4_current/measure_cell.py'),
             '--codec',codec,'--datasets',ds,'--bits',str(bits),'--seeds','0','--threads','1','8',
             '--batches','1','32','--queries','1000','--data',str(w/'data'),'--output',str(out),'--cell',str(dest)]
        if code:cmd+=['--code-bits',str(code)]
        log.write(json.dumps(dict(cell=i+1,total=len(plan),command=cmd,resumed=True))+'\n')
        subprocess.run(cmd,env=env,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert len(json.loads(dest.read_text()))==4 and profile.exists()
        print(f'{i+1}/{len(plan)} {ds} {codec} {bits} {code}: resumed {time.monotonic()-start:.0f}s',flush=True)
(out/'resume.json').write_text(json.dumps(metadata,indent=2))
(out/'finished-utc.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'exit-code').write_text('0\n')
