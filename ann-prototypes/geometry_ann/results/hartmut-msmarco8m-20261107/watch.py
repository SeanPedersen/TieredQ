"""Read-only progress watcher; does not inspect credentials or download URLs."""
from pathlib import Path
import datetime,json,os,subprocess,time

root=Path.cwd()
big=Path('/mnt/storage/tieredq-issue44-msmarco8m-20261107')
while True:
    state=json.loads((root/'status.json').read_text())
    output={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
            'stage':state['stage'],'model':state.get('model')}
    if state['stage']=='waiting_for_download':
        paths=list((big/'xet-download/.cache/huggingface/download').glob('*.incomplete'))
        if paths:
            output['source_GiB']=round(max(p.stat().st_size for p in paths)/(1<<30),2)
        output['source_total_GiB']=round(36217723200/(1<<30),2)
    if state['stage']=='validating_and_preparing_dataset':
        path=big/'export/base.i8'
        output['converted_codes_GiB']=round(path.stat().st_size/(1<<30),2) if path.exists() else 0
        path=root/'base.i8'
        output['nvme_codes_ready']=path.exists()
    for line in subprocess.run(['ps','-eo','pid=,pcpu=,rss=,etime=,comm=,args='],capture_output=True,text=True).stdout.splitlines():
        parts=line.split(None,5)
        if len(parts)!=6:continue
        pid,cpu,rss,elapsed,comm,args=parts
        if comm not in ['graph_bench','sq8_oracle','python','python3']:continue
        if not any(name in args for name in ['graph_bench','sq8_oracle','compare_faiss.py']):continue
        try:
            if Path(f'/proc/{pid}/cwd').resolve()!=root:continue
        except OSError:continue
        output.update(worker_elapsed=elapsed,worker_cpu_percent=float(cpu),worker_rss_GiB=round(int(rss)/(1<<20),2))
    print(json.dumps(output),flush=True)
    if state['stage'] in ['complete','finished_with_failures']:break
    if (root/'job.exitcode').exists():
        pid=int((root/'job.pid').read_text())
        if not Path(f'/proc/{pid}').exists():
            print(json.dumps({'job_exited':(root/'job.exitcode').read_text().strip()}),flush=True);break
    time.sleep(45)
