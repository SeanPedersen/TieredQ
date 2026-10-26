"""Audit every scheduling row and raw prediction/timing, then plot discrete measured points."""
import csv
import hashlib
import json
import re
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(sys.argv[1]).resolve()
def read(name):return json.loads((OUT/name).read_text())
def digest(x):return hashlib.sha256(x.tobytes()).hexdigest()


def main():
    rows=list(csv.DictReader((OUT/'results.csv').open()))
    expected={(c,a,t,b) for c in ('qdrant-turboquant-plus','tieredq-fast') for a in (1,2,4) for t in (1,8) for b in (1,32)}
    assert {(r['codec'],int(r['anchor_bits']),int(r['threads']),int(r['batch'])) for r in rows}==expected
    assert len(rows)==24
    manifest=read('provenance/gemini-manifest.json')
    assert manifest['source']['revision']=='e14a0f193988e9efb7ccc57b4c615418e6540f62'
    assert manifest['original_shape']==[100000,768] and manifest['corpus_n']==99000 and manifest['query_n']==1000 and manifest['finite']
    corpus=np.load(OUT/'provenance/gemini-corpus-row-ids.npy');queries=np.load(OUT/'provenance/gemini-query-row-ids.npy')
    assert len(corpus)==99000 and len(queries)==1000 and not np.intersect1d(corpus,queries).size
    assert np.array_equal(np.sort(np.concatenate([corpus,queries])),np.arange(100000))
    assert digest(corpus)==manifest['array_sha256']['corpus_ids'] and digest(queries)==manifest['array_sha256']['query_ids']
    truth=np.load(OUT/'truth.npy');assert digest(truth)==manifest['array_sha256']['truth']
    validations=read('validation.json');assert len(validations)==6
    assert all(v['serial_batch_thread_ids_equal'] and v['reload_ids_equal'] and v['storage_equal'] for v in validations)
    anchors={x['anchor_bits']:x for x in read('anchors.json')}
    selections={x['anchor_bits']:x for x in read('selections.json')}
    calibration=read('calibration.json')
    for bits in (1,2,4):
        limit=anchors[bits]['storage']['serving_bytes'];s=selections[bits]
        feasible=[x for x in calibration if x['anchor_bits']==bits and x['storage']['serving_bytes']<=limit]
        assert s==max(feasible,key=lambda x:x['storage']['serving_bytes'])
        assert 1<=s['requested_code_bits']<=8*768
    per_codec={}
    for row in rows:
        c=row['codec'];bits=int(row['anchor_bits']);t=int(row['threads']);b=int(row['batch']);label=f'{c}-{bits}'
        for k in ('qps','p50_ms','p95_ms','build_s','encode_vps','bytes_per_vector'):
            assert np.isfinite(float(row[k])) and float(row[k])>0,(label,k)
        assert int(row['base_n'])==99000 and int(row['queries'])==1000 and int(row['seed'])==0
        assert float(row['p95_ms'])>=float(row['p50_ms'])
        ids=np.load(OUT/f'ids-{label}-{t}-{b}.npy');reference=np.load(OUT/f'ids-reference-{label}.npy')
        assert np.array_equal(ids,reference),(label,'scheduling variation')
        assert ids.shape==(1000,10) and np.max(ids)<99000
        assert np.all(np.diff(np.sort(ids,axis=1),axis=1)>0)
        recall=float(np.mean([len(set(a)&set(z))/10 for a,z in zip(truth,ids,strict=True)]))
        assert abs(recall-float(row['recall10']))<1e-14
        times=read(f'timings-{label}-{t}-{b}.json')
        assert len(times)==3*((1000+b-1)//b)
        assert all(np.isfinite(x['seconds']) and x['seconds']>0 for x in times)
        for repeat in range(3):
            calls=[x for x in times if x['repeat']==repeat]
            assert [x['start'] for x in calls]==list(range(0,1000,b))
            assert sum(x['count'] for x in calls)==1000
        qps=3000/sum(x['seconds'] for x in times)
        latency=np.repeat([x['seconds']*1000 for x in times],[x['count'] for x in times])
        assert np.isclose(qps,float(row['qps']),rtol=1e-12)
        assert np.isclose(np.percentile(latency,50),float(row['p50_ms']),rtol=1e-12)
        assert np.isclose(np.percentile(latency,95),float(row['p95_ms']),rtol=1e-12)
        details=(anchors[bits] if c.startswith('qdrant') else selections[bits])['storage']
        assert float(row['bytes_per_vector'])==max(details['disk_bytes'],details['resident_bytes'])/99000
        assert float(row['bytes_per_vector'])<=float(row['anchor_bytes_per_vector'])
        per_codec.setdefault(label,[]).append(recall)
    assert all(len(set(v))==1 for v in per_codec.values())
    commands=read('provenance/compile_commands.json')
    for item in commands:
        command=item.get('command',' '.join(item.get('arguments',[])))
        assert '-O3' in command and '-march=native' in command and '-ffp-contract=off' in command
        assert '-Os' not in command
        assert re.findall(r'(?<!\S)-O(?:[0-3sgz]|fast)(?=\s|$)',command)[-1]=='-O3'
    fingerprint=read('provenance/rust-cargo-fingerprint.json')
    assert fingerprint['rustflags']==['-C','target-cpu=native','-C','opt-level=3']
    assert read('provenance/source-reconciliation.json')['status']=='PASS'
    for name,expected_hash in read('provenance/binary-sha256.json').items():
        assert hashlib.sha256((OUT/'binaries'/name).read_bytes()).hexdigest()==expected_hash
    audit=dict(status='PASS',scheduling_rows=24,unique_codec_anchor_pairs=6,finite_timings=True,
               dataset_identity=True,disjoint_split=True,storage_feasible=True,selection_uses_storage_only=True,
               native_validation=validations,serial_batch_thread_prediction_equality=True,recall_variation_across_scheduling=False,
               note='One seed and one run; no statistical confidence claims. Host services remained running.')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    primary=[r for r in rows if r['threads']=='8' and r['batch']=='32']
    for metric,title,ylabel,name in [('recall10','Recall@10','Recall@10','recall-storage.png'),('qps','8 threads, batch 32','Queries/second','qps-storage.png')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for codec in ('qdrant-turboquant-plus','tieredq-fast'):
            points=sorted([r for r in primary if r['codec']==codec],key=lambda r:int(r['anchor_bits']))
            ax.plot([float(r['bytes_per_vector']) for r in points],[float(r[metric]) for r in points],'o-',label=codec)
            for r in points:ax.annotate(f"A{r['anchor_bits']}",(float(r['bytes_per_vector']),float(r[metric])),xytext=(5,6),textcoords='offset points')
        ax.set(xlabel='Measured serving bytes/vector',ylabel=ylabel,title=f'Gemini/DBpedia · single-seed (0) · {title}')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/name,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,t in zip(axes,(1,8),strict=True):
        for codec in ('qdrant-turboquant-plus','tieredq-fast'):
            points=sorted([r for r in rows if r['codec']==codec and int(r['threads'])==t and r['batch']=='1'],key=lambda r:int(r['anchor_bits']))
            for percentile,style in [('p50_ms','o-'),('p95_ms','s--')]:
                ax.plot([float(r['bytes_per_vector']) for r in points],[float(r[percentile]) for r in points],style,label=f'{codec} {percentile[:3]}')
                for r in points:ax.annotate(f"A{r['anchor_bits']}",(float(r['bytes_per_vector']),float(r[percentile])),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set(xlabel='Measured serving bytes/vector',ylabel='Batch-1 response latency (ms)',title=f'{t} thread(s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Gemini/DBpedia · single-seed (0)');fig.tight_layout();fig.savefig(OUT/'latency-storage.png',dpi=160);plt.close(fig)
    lines=['# Gemini/DBpedia: native Qdrant TurboQuant versus TieredQ fast','',
      'Single-seed experiment (seed 0), Hartmut i9-7980XE, CPUs 0–7. All 24 scheduling rows passed the audit. No statistical-confidence claims. Queries are held-out document vectors, not independently embedded search queries.', '',
      '100,000 finite 768D Google `models/embedding-001` vectors; 99,000 corpus rows and 1,000 disjoint held-out queries. Both sides use identical FP32-normalized arrays and exhaustive FP32 cosine top-10 ground truth. Dataset revision: `e14a0f193988e9efb7ccc57b4c615418e6540f62`.', '',
      '| Pair | Codec | Requested / learned code bits | Bytes/vector | Below anchor | Recall@10 | Build s | Encode vectors/s | QPS (8t/32) | 1t p50/p95 ms | 8t p50/p95 ms |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in primary:
        single={int(x['threads']):x for x in rows if x['codec']==r['codec'] and x['anchor_bits']==r['anchor_bits'] and x['batch']=='1'}
        requested=r['requested_code_bits'] or str(int(r['anchor_bits'])*768)
        lines.append(f"| A{r['anchor_bits']} | {r['codec']} | {requested} / {r['learned_code_bits']} | {float(r['bytes_per_vector']):.3f} | {float(r['percent_below_anchor']):.3f}% | {float(r['recall10']):.4f} | {float(r['build_s']):.3f} | {float(r['encode_vps']):,.0f} | {float(r['qps']):,.1f} | {float(single[1]['p50_ms']):.3f}/{float(single[1]['p95_ms']):.3f} | {float(single[8]['p50_ms']):.3f}/{float(single[8]['p95_ms']):.3f} |")
    meta=read('meta.json')
    lines+=['','Calibration uses only measured storage: first feasible attempt within 0.5%, otherwise the largest measured feasible layout after bounded integer search. `calibration.json` retains every attempt; `selections.json` records choices. These are discrete within-memory comparisons. No interpolation or nominal-bit equality.', '',
      f"Calibration wall time: {meta['calibration_wall_s']:.2f} s. Selected construction: {meta['selected_build_s']:.2f} s. Matrix/search and result capture wall time: {meta['search_matrix_wall_s']:.2f} s. TieredQ construction times training and add/pack, excluding save/reload I/O. Qdrant construction times production calibration/encoding plus adapter compaction/persistence. These construction timers have different adapter boundaries; search timings both cover preprocessing, full scan and top-k.", '',
      'Storage is max(saved bytes, reloaded serving allocations including model and shell). `anchors.json` and `selections.json` separate disk, resident, code/cache/model and adapter allocations. No FP32 corpus storage, HNSW, filtering or original-vector reranking is retained by either index. Input arrays, transient score/top-k/query scratch, allocator bookkeeping and language/thread runtimes are outside this allocation measure; it is not process RSS.', '',
      'See [integration details](INTEGRATION.md) for production API constraints, calibration, actual bit paths, query precision, build flags and serving state. `provenance/` contains pinned dependencies, source manifests/patches, compiler commands, host state, dataset checksums and split IDs; `binaries/` contains the measured shared libraries. Full sources, datasets, build trees and model artifacts remain in the isolated remote workspace recorded in `local-state.json`.', '',
      'Three single-query warmups precede each scheduling point; all 1,000 queries run three timing repeats. Raw call times and predictions are retained. Batch-32 latency is whole-batch response latency, including the final partial batch. Scheduling recall is identical per codec/anchor: no recall values were averaged across schedules. Quantization recall differences between anchors are retained as measured.', '',
      '![Recall](recall-storage.png)','![Throughput](qps-storage.png)','![Batch-1 latency](latency-storage.png)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
