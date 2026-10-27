"""Load measured scheduling points, keeping full-accurate and native-pruned SAQ separate."""
import csv
from pathlib import Path
import statistics
HERE=Path(__file__).resolve().parent
REFRESH=HERE.parent/'issue4-range-refresh-20261027'
NAMES={'sift':'SIFT1M','glove':'GloVe-200','openai':'OpenAI3-1536','gemini':'DBpedia Gemini-768'}
CODECS={'tieredq':'TieredQ fast','saq':'SAQ full accurate','saq-native':'SAQ native pruned'}


def load(path,supplemental=False):
    rows=list(csv.DictReader(path.open()))
    groups={}
    for r in rows:
        if r['codec']=='rabitq':continue
        key=(r['dataset'],r['codec'],float(r['target_bits']),int(r['threads']),int(r['batch']))
        groups.setdefault(key,[]).append(r)
    points=[]
    for (d,c,b,t,batch),group in groups.items():
        assert len(group)==3 and {r['seed'] for r in group}=={'0','1','2'}
        zero=next(r for r in group if r['seed']=='0')
        p=dict(dataset=d,codec=c,bits=b,threads=t,batch=batch,supplemental=supplemental,
            code_bits=int(zero.get('code_bits') or round(b*int(zero['d']))),
            recall=statistics.mean(float(r['recall10']) for r in group),std=statistics.pstdev(float(r['recall10']) for r in group))
        p.update({k:float(zero[v]) for k,v in [('storage','effective_bytes_per_vector'),('qps','qps'),('p50','p50_ms'),('p95','p95_ms'),('build','build_s'),('encode','encode_vps')]})
        points.append(p)
    return points


def all_points():
    return (load(REFRESH/'output/results.csv')+load(REFRESH/'matched-storage/results.csv',True)+
            load(REFRESH/'storage-sweep/results.csv',True)+load(HERE/'results.csv')+load(HERE/'native-pruned/results.csv'))


def point(points,d,c,b,t=8,batch=32):
    return next(p for p in points if (p['dataset'],p['codec'],p['bits'],p['threads'],p['batch'])==(d,c,b,t,batch))


def choice(p):
    return f"{p['code_bits']:,}{'†' if p['supplemental'] else ''}; {p['storage']:.3f} B; {p['recall']:.4f}"
