"""Independently check published table values and local artifact links."""
import csv,json,math,re,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[4]
report=root/'docs/tieredq-vs-rabitq-benchmark.md'
text=report.read_text()
art=root/'benchmark/results/issue4-range-refresh-20261027'
rows=list(csv.DictReader((art/'output/results.csv').open()))
supp=list(csv.DictReader((art/'matched-storage/results.csv').open()))
assert len(rows)==288 and len(supp)==12
names={'SIFT1M':'sift','GloVe-200':'glove','OpenAI3-1536':'openai','DBpedia Gemini-768':'gemini'}
codecs={'TieredQ fast':'tieredq','RaBitQ':'rabitq'}

def metrics(dataset,codec,bits,threads=8,batch=32,source=rows):
    group=[r for r in source if (r['dataset'],r['codec'],float(r['target_bits']),int(r['threads']),int(r['batch']))==(dataset,codec,bits,threads,batch)]
    assert len(group)==3
    timed=next(r for r in group if r['seed']=='0')
    p={k:float(timed[k]) for k in ['qps','effective_bytes_per_vector','build_s','encode_vps','p50_ms','p95_ms']}
    p.update(mean=statistics.mean(float(r['recall10']) for r in group),std=statistics.pstdev(float(r['recall10']) for r in group))
    return p

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
                eligible=[(b,metrics(d,c,b)) for b in (1,2,4) if metrics(d,c,b)['effective_bytes_per_vector']<=cap]
                if not eligible: assert cell=='None';continue
                b,p=max(eligible,key=lambda pair:pair[1]['mean'])
                assert cell==f"{b}b; {p['effective_bytes_per_vector']:.3f} B; {p['mean']:.4f}"
        elif 'Minimum B/vector' in header:
            d,c=names[v[0]],codecs[v[1]];eligible=[(b,metrics(d,c,b)) for b in (1,2,4) if metrics(d,c,b)['mean']>=.95]
            if not eligible: assert v[2:]==['Not reached','—','—','—'];continue
            b,p=min(eligible,key=lambda pair:pair[1]['effective_bytes_per_vector'])
            assert v[2:]==[str(b),f"{p['effective_bytes_per_vector']:.3f}",f"{p['mean']:.4f}",f"{p['qps']:,.0f}"]
        elif 'TieredQ build · s' in header:
            d,b=names[v[0]],int(v[1]);t,r=metrics(d,'tieredq',b),metrics(d,'rabitq',b)
            assert v[2:]==[f"{t['build_s']:.3f}",f"{r['build_s']:.3f}",f"{t['encode_vps']:,.0f}",f"{r['encode_vps']:,.0f}"]
        elif header[-1]=='Bounds/control QPS · 1 thread, batch 1':
            a=next(x for x in json.loads((art/'output/ablation-summary.json').read_text()) if x['dataset']==names[v[0]] and x['bits']==int(v[1]) and x['threads']==1 and x['batch']==1)
            assert a['checkpoint_active'] and v[2]==f"{a['qps_ratio']:.2f}×"
        elif header[0]=='OpenAI3-1536 configuration':
            p=metrics('openai','tieredq',float(supp[0]['target_bits']),source=supp) if v[0].startswith('TieredQ') else metrics('openai','rabitq',2)
            assert v[1:]==[f"{p['effective_bytes_per_vector']:.3f}",f"{8*p['effective_bytes_per_vector']:,.1f}",f"{p['mean']:.4f} ± {p['std']:.4f}",f"{p['qps']:,.0f}"]
        elif header[0]=='TieredQ supplemental threads':
            t=int(v[0]);p=metrics('openai','tieredq',float(supp[0]['target_bits']),t,1,supp);q=metrics('openai','tieredq',float(supp[0]['target_bits']),t,32,supp)
            assert v[1:]==[f"{p['p50_ms']:.3f} / {p['p95_ms']:.3f}",f"{q['qps']:,.0f}"]
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
print(f'PASS: {checked} table rows checked independently; {len(links)} relative links resolve; narrative comparisons checked.')
