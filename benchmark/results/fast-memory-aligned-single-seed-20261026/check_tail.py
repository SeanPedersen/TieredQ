"""Retain a targeted latency recheck for the anomalous OpenAI 2048-bit tail."""
import json
import subprocess
import time
import numpy as np
from threadpoolctl import threadpool_limits
from run_calibrated import OUT,load_data,build


def main():
    base,queries,truth,_=load_data('openai')
    index,_,_=build(base,'openai',2048)
    output=np.empty((1,10),np.uint32)
    for q in queries[:3]: index.search(q[None,:],output,8)
    repeats=[];durations=[]
    before=subprocess.check_output(['ps','-eo','pid,etime,pcpu,args','--sort=-pcpu'],text=True).splitlines()[:15]
    for _ in range(3):
        current=[]
        for q in queries:
            tick=time.perf_counter();index.search(q[None,:],output,8)
            current.append((time.perf_counter()-tick)*1000)
        durations.append(current)
        repeats.append({'p50_ms':float(np.percentile(current,50)),'p95_ms':float(np.percentile(current,95)),'max_ms':max(current)})
    result={'scope':'Targeted diagnostic only: OpenAI seed 0, 2048 code bits, eight threads, batch 1, original 1000 queries, three warmup queries and three repeats. Matrix row retained unchanged; no profiling or budget selection.', 'repeats':repeats,'overall_p50_ms':float(np.percentile(durations,50)),'overall_p95_ms':float(np.percentile(durations,95)),'host_before':before,'durations_ms':durations,'limitation':'A separate repeat cannot recover the cause of the original p95 spike. Background services remain active; no isolated causal or confidence claim.'}
    (OUT/'tail-diagnostic.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='durations_ms'},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
