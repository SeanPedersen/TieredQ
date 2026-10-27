"""Independently check published table values and local artifact links."""
import csv,json,math,re,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[4]
report=root/'docs/tieredq-vs-rabitq-benchmark.md'
text=report.read_text()
art=root/'benchmark/results/issue4-range-refresh-20261027'
rows=list(csv.DictReader((art/'output/results.csv').open()))
supp=list(csv.DictReader((art/'matched-storage/results.csv').open()))
sweep=list(csv.DictReader((art/'storage-sweep/results.csv').open()))
assert len(rows)==288 and len(supp)==12 and len(sweep)==60
allrows=rows+supp+sweep
names={'SIFT1M':'sift','GloVe-200':'glove','OpenAI3-1536':'openai','DBpedia Gemini-768':'gemini'}
codecs={'TieredQ fast':'tieredq','RaBitQ':'rabitq'}
names_inv={v:k for k,v in names.items()}

def metrics(dataset,codec,bits,threads=8,batch=32,source=rows):
    group=[r for r in source if (r['dataset'],r['codec'],float(r['target_bits']),int(r['threads']),int(r['batch']))==(dataset,codec,bits,threads,batch)]
    assert len(group)==3
    timed=next(r for r in group if r['seed']=='0')
    p={k:float(timed[k]) for k in ['qps','effective_bytes_per_vector','build_s','encode_vps','p50_ms','p95_ms']}
    p.update(mean=statistics.mean(float(r['recall10']) for r in group),std=statistics.pstdev(float(r['recall10']) for r in group))
    return p

def measured_options(dataset,codec):
    budgets=sorted({float(r['target_bits']) for r in allrows if r['dataset']==dataset and r['codec']==codec})
    return [(b,metrics(dataset,codec,b,source=allrows)) for b in budgets]


def supplemental_pair(label):
    display,anchor=label.rsplit(' M',1)
    dataset=names[display]
    candidates=[r for r in supp+sweep if r['dataset']==dataset and int(r.get('anchor_bits') or 2)==int(anchor)]
    budget=float(candidates[0]['target_bits'])
    point=metrics(dataset,'tieredq',budget,source=allrows)
    baseline=metrics(dataset,'rabitq',int(anchor))
    return dataset,budget,point,baseline


blocks=re.findall(r'(?m)^\|.*(?:\n\|.*)+',text)
checked=0
for block in blocks:
    lines=[[x.strip() for x in line.strip('|').split('|')] for line in block.splitlines()]
    header=lines[0]
    for v in lines[2:]:
        if header[0]=='Dataset' and header[1]=='Codec' and header[2]=='Nominal bits/dim':
            d,c,b=names[v[0]],codecs[v[1]],int(v[2]);p=metrics(d,c,b);q=metrics(d,c,b,1)
            assert v[3:]==[f"{p['mean']:.4f} ± {p['std']:.4f}",f"{p['effective_bytes_per_vector']:.3f}",f"{q['qps']:,.0f}",f"{p['qps']:,.0f}"]
        elif 'TieredQ · 1 thread' in header:
            d,b=names[v[0]],int(v[1]); ps=[metrics(d,c,b,t,1) for t in (1,8) for c in ('tieredq','rabitq')]
            assert v[2:]==[f"{p['p50_ms']:.3f} / {p['p95_ms']:.3f}" for p in ps]
        elif len(header)>2 and header[2].startswith('TieredQ / RaBitQ'):
            d,b=names[v[0]],int(v[1]); assert v[2:]==[f"{metrics(d,'tieredq',b,t)['qps']/metrics(d,'rabitq',b,t)['qps']:.2f}×" for t in (1,8)]
        elif header[1]=='Cap · B/vector':
            d=names[v[0]];caps=[metrics(d,'rabitq',b)['effective_bytes_per_vector'] for b in (1,2,4)];cap=next(x for x in caps if f'{x:.3f}'==v[1])
            for c,cell in zip(('tieredq','rabitq'),v[2:]):
                eligible=[(b,p) for b,p in measured_options(d,c) if p['effective_bytes_per_vector']<=cap]
                if not eligible: assert cell=='None';continue
                b,p=max(eligible,key=lambda pair:pair[1]['mean'])
                source=next(r for r in allrows if r['dataset']==d and r['codec']==c and float(r['target_bits'])==b)
                code_bits=int(source.get('code_bits') or round(b*int(source['d'])))
                marker='†' if source.get('code_bits') else ''
                assert cell==f"{code_bits:,}{marker}; {p['effective_bytes_per_vector']:.3f} B; {p['mean']:.4f}"
        elif 'Minimum B/vector' in header:
            d,c=names[v[0]],codecs[v[1]];eligible=[(b,p) for b,p in measured_options(d,c) if p['mean']>=.95]
            if not eligible: assert v[2:]==['Not reached','—','—','—'];continue
            b,p=min(eligible,key=lambda pair:pair[1]['effective_bytes_per_vector'])
            source=next(r for r in allrows if r['dataset']==d and r['codec']==c and float(r['target_bits'])==b)
            code_bits=int(source.get('code_bits') or round(b*int(source['d'])))
            marker='†' if source.get('code_bits') else ''
            assert v[2:]==[f"{code_bits:,}{marker}",f"{p['effective_bytes_per_vector']:.3f}",f"{p['mean']:.4f}",f"{p['qps']:,.0f}"]
        elif 'TieredQ build · s' in header:
            d,b=names[v[0]],int(v[1]);t,r=metrics(d,'tieredq',b),metrics(d,'rabitq',b)
            assert v[2:]==[f"{t['build_s']:.3f}",f"{r['build_s']:.3f}",f"{t['encode_vps']:,.0f}",f"{r['encode_vps']:,.0f}"]
        elif header[-1]=='Bounds/control QPS · 1 thread, batch 1':
            a=next(x for x in json.loads((art/'output/ablation-summary.json').read_text()) if x['dataset']==names[v[0]] and x['bits']==int(v[1]) and x['threads']==1 and x['batch']==1)
            assert a['checkpoint_active'] and v[2]==f"{a['qps_ratio']:.2f}×"
        elif header[0]=='Dataset / cap' and header[1]=='Codec':
            d,b,t,r=supplemental_pair(v[0]);p=t if v[1]=='TieredQ fast' else r
            q=metrics(d,'tieredq',b,1,32,allrows) if v[1]=='TieredQ fast' else metrics(d,'rabitq',int(v[0].rsplit('M',1)[1]),1)
            assert v[2:]==[f"{p['effective_bytes_per_vector']:.3f}",f"{8*p['effective_bytes_per_vector']:,.1f}",f"{p['mean']:.4f} ± {p['std']:.4f}",f"{q['qps']:,.0f}",f"{p['qps']:,.0f}"]
        elif header[0]=='Dataset / cap' and header[1]=='TieredQ code bits/vector':
            d,b,p,r=supplemental_pair(v[0]);code_bits=round(b*(1536 if d=='openai' else 768))
            assert v[1:]==[f'{code_bits:,}',f"{100*(1-p['effective_bytes_per_vector']/r['effective_bytes_per_vector']):.2f}%",f"{100*(p['mean']-r['mean']):+.2f}",f"{p['qps']/r['qps']:.2f}×"]
        elif header[0]=='TieredQ point':
            d,b,p,r=supplemental_pair(v[0]);one=metrics(d,'tieredq',b,1,1,allrows);eight=metrics(d,'tieredq',b,8,1,allrows)
            assert v[1:]==[f"{one['p50_ms']:.3f} / {one['p95_ms']:.3f}",f"{eight['p50_ms']:.3f} / {eight['p95_ms']:.3f}",f"{p['build_s']:.3f}",f"{p['encode_vps']:,.0f}"]
        elif header[0]=='Dataset' and header[1]=='Base vectors':
            r=next(x for x in rows if x['dataset']==names[v[0]])
            assert v[1:4]==[f"{int(r['base_n']):,}",f"{int(r['d']):,}",f"{int(r['queries']):,}"]
        else: raise AssertionError(header)
        checked+=1
links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
assert all((report.parent/link).exists() for link in links),[link for link in links if not (report.parent/link).exists()]
for d in names.values():
 for b in (1,2,4):
    t,r=metrics(d,'tieredq',b),metrics(d,'rabitq',b)
    assert t['build_s']>r['build_s']
    assert (t['encode_vps']<r['encode_vps']) == (b==1)
for d in ('openai','gemini'):
 for b in (2,4):
    ratio=metrics(d,'rabitq',b)['qps']/metrics(d,'tieredq',b)['qps']
    assert 3.285<=ratio<3.725
p=metrics('openai','tieredq',float(supp[0]['target_bits']),source=supp);r=metrics('openai','rabitq',2)
assert f"{100*(p['mean']-r['mean']):.2f}"=='3.13'
assert f"{100*(1-p['effective_bytes_per_vector']/r['effective_bytes_per_vector']):.2f}"=='0.52'
assert f"{p['build_s']:.3f}"=='14.552'
assert f"{p['encode_vps']:,.0f}"=='83,780'
assert json.loads((art/'matched-storage/meta.json').read_text())['status']=='complete'
assert json.loads((art/'storage-sweep/meta.json').read_text())['status']=='complete'
for dataset in ('openai','gemini'):
 for anchor in (1,2,4):
    d,b,p,r=supplemental_pair(names_inv[dataset]+f' M{anchor}')
    assert p['effective_bytes_per_vector']<r['effective_bytes_per_vector']
def verify_reused_sources():
    source_files={key:list(csv.DictReader((art/'reuse'/name).open())) for key,name in
                  [('seed0','issue4-results.csv'),('recall','curve-results.csv'),('gemini','gemini-results.csv')]}
    count=0
    for row in rows:
        if row['codec']!='rabitq':
            continue
        if row['dataset']=='gemini':
            if row['seed']!='0':
                continue
            original=next(r for r in source_files['gemini'] if r['anchor_bits']==row['target_bits'] and
                          all(r[k]==row[k] for k in ['seed','threads','batch']))
            fields=['recall10','qps','p50_ms','p95_ms','build_s','encode_vps']
        else:
            pool=source_files['seed0' if row['seed']=='0' else 'recall']
            original=next(r for r in pool if all(r[k]==row[k] for k in ['dataset','codec','target_bits','seed']) and
                          (row['seed']!='0' or all(r[k]==row[k] for k in ['threads','batch'])))
            fields=['recall10','effective_bytes_per_vector']
            if row['seed']=='0':
                fields+=['qps','p50_ms','p95_ms','build_s','encode_vps']
        for field in fields:
            assert float(row[field])==float(original[field])
        count+=1
    return count


pairs=[supplemental_pair(names_inv[d]+f' M{anchor}') for d in ('openai','gemini') for anchor in (1,2,4)]
gains=[100*(p['mean']-r['mean']) for _,_,p,r in pairs]
savings=[100*(1-p['effective_bytes_per_vector']/r['effective_bytes_per_vector']) for _,_,p,r in pairs]
assert all(gain>0 for gain in gains)
assert f"{min(gains):.2f} to {max(gains):.2f} percentage points" in text
assert f"{min(savings):.2f}–{max(savings):.2f}% less accounted serving storage" in text
low=[p['qps']/r['qps'] for d,b,p,r in pairs if b*(1536 if d=='openai' else 768) in (833,493)]
high=[r['qps']/p['qps'] for d,b,p,r in pairs if b*(1536 if d=='openai' else 768) not in (833,493)]
assert f"{min(low):.2f}–{max(low):.2f}×" in text
assert f"{min(high):.2f}–{max(high):.2f}× faster" in text
for path in [art/'matched-storage/diagnostics-seed0.json',*sorted((art/'storage-sweep').glob('*/diagnostics-seed0.json'))]:
    diagnostics=json.loads(path.read_text())
    assert sum(tier['dimensions']*tier['bits'] for tier in diagnostics['tiers'])>0
reused=verify_reused_sources()
print(f'PASS: {checked} table rows checked independently; {len(links)} relative links resolve; '
      f'{reused} reused RaBitQ rows match source CSVs; narrative comparisons checked.')
