"""Generate issue #4 tables and plots exclusively from the current workspace.

Refuses incomplete timing sessions and incomplete three-seed recall points.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics as st
from collections import defaultdict
from matrix import NAMES,DIMS,memory_folders,primary_cells,primary_matches,primary_equal

CODECS = dict(tieredq='TieredQ', rabitq='RaBitQ', qdrant='Qdrant TQ+')


def read_csv(p):
    return list(csv.DictReader(p.open()))


def table(header, rows):
    return '\n'.join(['| '+' | '.join(header)+' |', '| '+' | '.join(['---']*len(header))+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def mean_std(values):
    values = list(map(float, values))
    return f'{st.mean(values):.4f} ± {st.stdev(values):.4f}'


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(w):
    matches, cells, timings, equal, builds, serving = {}, [], {}, [], [], []
    for bits in (1, 2, 4):
        rows = primary_cells(w,bits)
        for row in rows:
            row['lane'] = bits
            row['effective_bytes_per_vector']=float(row['disk_bytes'])/int(row['base_n'])
            row['fp32_compression']=32*int(row['d'])/float(row['disk_bits_per_vector'])
            cells.append(row)
        for codec in ('rabitq', 'qdrant'):
            for row in primary_matches(w,bits,codec):
                matches[row['dataset'], codec, bits, row['metric']] = row
        for folder in [*memory_folders(w,bits),w/f'reference-build-b{bits}']:
            for path in folder.glob('cell-*/build.json'):
                if not any(path.parent.name.startswith('cell-'+dataset+'-') for dataset in NAMES):
                    continue
                built = json.loads(path.read_text())
                built['cell'] = str(path.parent.relative_to(w))
                builds.append(built)
                serve=path.parent/'serve.json'
                if serve.exists():
                    values=json.loads(serve.read_text())
                    values={key:value for key,value in values.items() if key!='schedule_metrics'}
                    values.update(cell=built['cell'],base_n=built['base_n'],d=built['d'])
                    for key in ('ram_loaded_total_rss_bytes','ram_peak_total_rss_bytes'):
                        if key in values:
                            values[key.removesuffix('_bytes')+'_bits_per_vector']=values[key]*8/built['base_n']
                    serving.append(values)
        equal += primary_equal(w,bits)
    plan = json.loads((w/'timing/plan.json').read_text())
    for ds, codec, bits, code in plan:
        filename = f'cell-{ds}-{codec}-{bits}'+(f'-c{code}' if code else '')+'.json'
        for row in json.loads((w/'timing'/filename).read_text()):
            row['serialized_index_bytes']=round(float(row['effective_bytes_per_vector'])*int(row['base_n']))
            row['serialized_bits_per_vector']=8*float(row['effective_bytes_per_vector'])
            row['fp32_compression']=32*int(row['d'])/row['serialized_bits_per_vector']
            timings[ds, codec, code or bits, row['threads'], row['batch']] = row | {'code_bits': code}
    assert len(matches) == 6*2*3*3, len(matches)
    assert len(equal) == 6*(2*3+1), len(equal)
    assert len(timings) == 4*len(plan)
    return matches, cells, timings, equal, builds, serving


def groups(cells):
    selected = defaultdict(dict)
    for row in cells:
        budget = int(row['requested_code_bits']) if row['codec'] == 'tieredq' else int(row['target_bits'])
        selected[row['dataset'], row['codec'], budget][int(row['seed'])] = row
    return {key: list(seeds.values()) for key, seeds in selected.items() if set(seeds) == {0, 1, 2}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('workspace', type=Path)
    a = p.parse_args()
    w = a.workspace.resolve()
    assert (w/'timing/exit-code').read_text().strip() == '0'
    matches, cells, timing, equal, builds, serving = collect(w)
    grouped = groups(cells)
    write_csv(w/'all-memory.csv', cells)
    write_csv(w/'all-timing.csv', list(timing.values()))
    write_csv(w/'all-build-diagnostics.csv', builds)
    write_csv(w/'all-serving-diagnostics.csv', serving)
    write_csv(w/'equal-recall.csv', [{k:v for k,v in r.items() if k != 'memory_rows'} for r in equal])
    preparations=[]
    for path in (w/'preparation').glob('cell-*.json'):
        record=json.loads(path.read_text())
        assert record['available'] and record['search_ids_equal'],path
        common={key:value for key,value in record.items() if key not in ('schedules','index_diagnostics')}
        if record.get('index_diagnostics'):
            common['scalar_fast_scan']=record['index_diagnostics']['scalar_fast_scan']
        preparations += [common|row for row in record['schedules']]
    assert len(preparations)==len(timing),(len(preparations),len(timing))
    write_csv(w/'preparation.csv',preparations)
    by_equal = {(r['dataset'], r['reference_codec'], r['reference_bits']): r for r in equal}

    def speed(ds, codec, budget, threads=8, batch=32, field='qps'):
        return float(timing[ds, codec, budget, threads, batch][field])

    def ref_mean(ds, codec, bits):
        return st.mean(float(r['recall10']) for r in grouped[ds, codec, bits])

    def fastest(ds, codec, target, threads, batch):
        candidates = [bits for bits in (1, 2, 4) if ref_mean(ds, codec, bits) >= target-1e-12]
        return max(candidates, key=lambda bits: speed(ds, codec, bits, threads, batch)) if candidates else None

    recall_rows, memory_rows, speed_rows, equal_rows, detail_rows, fastest_rows = [], [], [], [], [], []
    deltas = defaultdict(list)
    memory_speed_wins=defaultdict(list)
    speed_ratios = defaultdict(list)
    for ds in NAMES:
        for codec in ('rabitq', 'qdrant'):
            for bits in (1, 2, 4):
                ref = grouped[ds, codec, bits]
                entry = [NAMES[ds], CODECS[codec], bits, mean_std(r['recall10'] for r in ref)]
                mem = [NAMES[ds], CODECS[codec], bits]
                for metric in ('disk', 'ram', 'peak'):
                    row = matches[ds, codec, bits, metric]
                    tq = row['tieredq']
                    if tq:
                        recall = tq['recall10']
                        delta = 100*(recall['mean']-row['reference']['recall10']['mean'])
                        deltas[metric].append(delta)
                        entry += [f"{recall['mean']:.4f} ± {recall['std']:.4f} ({delta:+.2f})"]
                        mem += [f"{tq[metric]['max']:.1f} / {row['target_bits_per_vector']:.1f} ({row['code_bits']} code bits)"]
                        for threads, batch in ((1,1), (1,32), (8,1), (8,32)):
                            tq_speed = speed(ds, 'tieredq', row['code_bits'], threads, batch)
                            ref_speed = speed(ds, codec, bits, threads, batch)
                            detail_rows.append([NAMES[ds], CODECS[codec], bits, metric, row['code_bits'], threads, batch,
                                                f'{tq_speed:.1f}', f'{ref_speed:.1f}', f'{tq_speed/ref_speed:.2f}'])
                    else:
                        entry += ['none fits']
                        mem += [f"{row['target_bits_per_vector']:.1f} / none"]
                recall_rows.append(entry)
                memory_rows.append(mem)
                row = matches[ds, codec, bits, 'ram']
                code = row['code_bits']
                if code:
                    ratio = speed(ds, 'tieredq', code)/speed(ds, codec, bits)
                    memory_speed_wins[codec,bits].append((ds,ratio))
                    speed_rows.append([NAMES[ds], CODECS[codec], bits, code,
                        f"{speed(ds, 'tieredq', code):.0f} / {speed(ds, codec, bits):.0f} ({ratio:.2f}×)",
                        f"{speed(ds, 'tieredq', code,1,1,'p50_ms'):.3f} / {speed(ds,codec,bits,1,1,'p50_ms'):.3f}"])
                eq = by_equal[ds, codec, bits]
                if not eq['reached']:
                    equal_rows.append([NAMES[ds], CODECS[codec], bits, f"{eq['reference_recall10']:.4f}", 'unreached', '—', '—', '—', '—'])
                    continue
                code = eq['code_bits']
                tq_mem = max(float(r['disk_bits_per_vector']) for r in eq['memory_rows'])
                ref_mem = max(float(r['disk_bits_per_vector']) for r in ref)
                ratios = []
                for threads,batch in ((1,32),(8,32)):
                    best = fastest(ds, codec, eq['reference_recall10'], threads,batch)
                    ratio = speed(ds,'tieredq',code,threads,batch)/speed(ds,codec,best,threads,batch)
                    speed_ratios[codec,threads,batch].append(ratio)
                    ratios.append(f'{ratio:.2f}× ({best}b)')
                latency_bits=min((budget for budget in (1,2,4) if ref_mean(ds,codec,budget)>=eq['reference_recall10']-1e-12),
                                 key=lambda budget:speed(ds,codec,budget,1,1,'p50_ms'))
                equal_rows.append([NAMES[ds], CODECS[codec], bits,
                    f"{eq['reference_recall10']:.4f} / {eq['recall10_mean']:.4f} ± {eq['recall10_std']:.4f}",
                    code, f'{tq_mem:.1f} / {ref_mem:.1f} ({tq_mem/ref_mem:.2f}×)', *ratios,
                    f"{speed(ds,'tieredq',code,1,1,'p50_ms'):.3f} / {speed(ds,codec,latency_bits,1,1,'p50_ms'):.3f} ({latency_bits}b)"])
                tq_best = max((r for (dataset,name,budget,_,_),r in timing.items()
                               if dataset==ds and name=='tieredq' and budget==code), key=lambda r:float(r['qps']))
                eligible = [r for (dataset,name,budget,_,_),r in timing.items()
                            if dataset==ds and name==codec and ref_mean(ds,codec,budget)>=eq['reference_recall10']-1e-12]
                ref_best = max(eligible,key=lambda r:float(r['qps']))
                label = lambda r:f"{r['threads']}t·b{r['batch']}"
                fastest_rows.append([NAMES[ds],CODECS[codec],bits,
                    f"{float(tq_best['qps']):.1f} ({label(tq_best)})",
                    f"{float(ref_best['qps']):.1f} ({ref_best['target_bits']}b, {label(ref_best)})",
                    f"{float(tq_best['qps'])/float(ref_best['qps']):.2f}×"])

    minimum_rows = []
    minimums={}
    for ds in NAMES:
        for codec in CODECS:
            points = [(max(float(r['disk_bits_per_vector']) for r in rows), budget,
                       st.mean(float(r['recall10']) for r in rows), rows)
                      for (dataset,name,budget), rows in grouped.items() if dataset == ds and name == codec]
            reached = sorted(pt for pt in points if pt[2] >= .95)
            if reached:
                bits,budget,recall,rows = reached[0]
                minimums[ds,codec]=(bits,budget,recall,rows)
                minimum_rows.append([NAMES[ds], CODECS[codec], budget, f'{bits:.1f}', f'{bits/8:.2f}',
                                     f'{32*DIMS[ds]/bits:.2f}×', mean_std(r['recall10'] for r in rows)])
            else:
                minimum_rows.append([NAMES[ds], CODECS[codec], 'unreached', '—', '—', '—',
                                     f'best {max(pt[2] for pt in points):.4f}'])
    speed95=[]
    for ds in NAMES:
        for codec in ('rabitq','qdrant'):
            if (ds,'tieredq') not in minimums:
                speed95.append([NAMES[ds],CODECS[codec],'unreached','—','—','—'])
                continue
            code=minimums[ds,'tieredq'][1]
            tq_rows=[r for (dataset,name,budget,_,_),r in timing.items() if dataset==ds and name=='tieredq' and budget==code]
            candidates=[r for (dataset,name,budget,_,_),r in timing.items()
                        if dataset==ds and name==codec and ref_mean(ds,codec,budget)>=.95]
            if not candidates:
                speed95.append([NAMES[ds],CODECS[codec],code,'reference unreached','—','—'])
                continue
            tq=max(tq_rows,key=lambda r:float(r['qps']))
            ref=max(candidates,key=lambda r:float(r['qps']))
            tq_latency=min(float(r['p50_ms']) for r in tq_rows if r['batch']==1)
            ref_latency=min(float(r['p50_ms']) for r in candidates if r['batch']==1)
            speed95.append([NAMES[ds],CODECS[codec],code,
                f"{tq['qps']:.0f} / {ref['qps']:.0f} ({tq['qps']/ref['qps']:.2f}×)",
                f"{tq['threads']}t·b{tq['batch']} / {ref['target_bits']}b {ref['threads']}t·b{ref['batch']}",
                f'{tq_latency:.3f} / {ref_latency:.3f}'])

    build_rows = []
    for ds in NAMES:
        for codec in CODECS:
            rows = [b for b in builds if f'cell-{ds}-{codec}-' in b['cell']]
            assert rows, (ds,codec)
            def span(key, factor=1):
                values = [float(r[key])*factor for r in rows]
                return f'{min(values):.1f}–{max(values):.1f}'
            n = rows[0]['base_n']
            timed={budget:r for (dataset,name,budget,threads,batch),r in timing.items()
                   if dataset==ds and name==codec and threads==8 and batch==32}
            def timed_span(key):
                values=[float(r[key]) for r in timed.values()]
                return f'{min(values):.1f}–{max(values):.1f}'
            build_rows.append([NAMES[ds],CODECS[codec],len(rows),len(timed),timed_span('build_s'),timed_span('encode_vps'),
                               span('build_peak_rss_bytes',8/n),span('build_peak_total_rss_bytes',8/n)])

    # Same index and budget, real versus held-out document queries; all three seeds.
    query_rows = []
    for codec in CODECS:
        for bits in (1,2,4):
            if codec == 'tieredq':
                code = matches['msmarco1m','rabitq',bits,'disk']['code_bits']
                rows = grouped['msmarco1m',codec,code]
                budget = f'{code} code bits (RaBitQ disk match)'
            else:
                rows = grouped['msmarco1m',codec,bits]
                budget = f'{bits} bits/dim'
            query_rows.append([CODECS[codec], budget, mean_std(r['recall10'] for r in rows),
                               mean_std(r['companion_recall10'] for r in rows)])

    supplemental = '# Complete current-run tables\n\n## Memory targets\n\n'
    supplemental += table(['Dataset','Reference','Reference nominal bits/dim','Disk TQ max / ref min (bits/vector; TQ code bits)',
                           'Steady RAM TQ max / ref min','Peak RAM TQ max / ref min'],memory_rows)
    supplemental += '\n\n## Speed at every equal-memory target\n\n'+table(
        ['Dataset','Reference','bits/dim','Match metric','Code bits','Threads','Batch','TieredQ QPS','Reference QPS','QPS ratio'],detail_rows)
    supplemental += '\n\n## Best measured configurations at equal recall\n\n'+table(
        ['Dataset','Reference','Target bits/dim','Best TieredQ QPS (schedule)','Fastest reference QPS reaching target','Ratio'],fastest_rows)
    supplemental += '\n\nRaw single-query p50/p95 and all schedule latencies: `all-timing.csv`. '
    supplemental += 'Raw total bytes, bits/vector, per-seed build time and memory: `all-memory.csv`. '
    supplemental += 'Absolute process build peaks and encoding rates: `all-build-diagnostics.csv`.\n'
    supplemental += '\n\n[Codec-first configuration tables](configurations.md) combine accuracy, storage, serving RAM, build cost and all four timing schedules for each evaluated configuration.\n'
    (w/'tables.md').write_text(supplemental)
    complete=[]
    for (ds,codec,budget),rows in sorted(grouped.items()):
        if (ds,codec,budget,1,1) not in timing:
            continue
        one=timing[ds,codec,budget,1,1]
        qps=[f"{timing[ds,codec,budget,t,b]['qps']:.1f}" for t,b in ((1,1),(1,32),(8,1),(8,32))]
        complete.append([CODECS[codec],NAMES[ds],f'{budget} '+('code bits' if codec=='tieredq' else 'bits/dim'),
            mean_std(r['recall10'] for r in rows),f"{max(float(r['disk_bytes']) for r in rows)/1e6:.2f}",
            f"{max(float(r['ram_loaded_rss_bytes']) for r in rows)/1e6:.2f}",
            f"{max(float(r['ram_peak_rss_bytes']) for r in rows)/1e6:.2f}",
            f"{one['build_s']:.3f}",f"{one['encode_vps']:.0f}",*qps,
            f"{one['p50_ms']:.3f} / {one['p95_ms']:.3f}"])
    (w/'configurations.md').write_text('# Complete evaluated configurations\n\n'
        'Recall is three-seed mean ± sample std. Storage and serving RAM are maximum over seeds; RAM is RSS growth. '
        'Build and encode measurements are isolated eight-thread timing-session measurements. MB is decimal. '
        'Latency is single-query, one thread. All native search uses the default fine-tier configuration.\n\n'+
        table(['Codec','Dataset','Budget','Recall@10 ± std','Disk MB','Steady RAM MB','Peak RAM MB','Build s',
               'Encode vec/s','QPS 1t/b1','QPS 1t/b32','QPS 8t/b1','QPS 8t/b32','p50 / p95 ms'],complete))

    start=dt.datetime.fromisoformat((w/'started-utc.txt').read_text().strip())
    end=dt.datetime.fromisoformat((w/'timing/finished-utc.txt').read_text().strip())
    hours=(end-start).total_seconds()/3600
    acceptance=('This expanded run exceeded one hour; the under-one-hour acceptance is not met for this scope.'
                if hours>1 else 'This observed expanded run completed in under one hour.')
    rev=(w/'source/SOURCE_REVISION').read_text().strip()
    link=f'../benchmark/results/{w.name.replace("TieredQ-", "")}'
    counts='; '.join(f'{metric}: {sum(d>0 for d in vals)}/36 higher, Δ {min(vals):+.2f} to {max(vals):+.2f} points'
                     for metric,vals in deltas.items())
    ratios='; '.join(f'{CODECS[codec]} {threads}t·b{batch}: {min(vals):.2f}–{max(vals):.2f}×'
                    for (codec,threads,batch),vals in speed_ratios.items())
    scan_paths='/'.join(sorted({'scalar' if r['scalar_fast_scan'] else 'AVX-512'
                               for r in preparations if r['codec']=='tieredq'}))
    crossover='; '.join(f"vs {CODECS[codec]} at {bits} bits/dim: faster in {sum(ratio>1 for _,ratio in values)}/{len(values)} datasets (8t·b32)"
                        for (codec,bits),values in memory_speed_wins.items())
    report=f'''# TieredQ vs RaBitQ and Qdrant TQ+: current-main benchmark

Refreshed measurements for issue #4, library `{rev}`, six datasets, flat compressed top-10 search.
TieredQ uses its **default flat fast index, fine tiers enabled**, native L2 on SIFT and cosine elsewhere.
Fine tiers refine compressed candidates when the allocation uses wide tiers, typically from around
3 bits per original dimension; their preparation, scan and rescoring costs are included.
All TieredQ recall/memory cells were remeasured. Both references and every selected TieredQ budget were
retimed in an interleaved plan. The process stopped after 99 complete cells without an exit marker;
the remaining cells were resumed with unchanged binaries and settings. The interrupted cell was rerun,
and `timing/resume.json` records the interruption. [{w.name.replace('TieredQ-', '')} results and provenance]({link}/README.md).

Equal-memory accuracy ({counts}). Equal-recall QPS against each reference's fastest evaluated configuration
reaching the target: {ratios}. These ranges describe this host and adapter matrix.

## Answers to the four questions

1. **Accuracy and compression:** disk, steady serving RAM and peak serving RAM are matched separately below.
   The 0.95 table gives the smallest evaluated serialized storage with a three-seed mean reaching the target.
2. **Speed at equal recall:** selected budgets are confirmed on seeds 0–2 before timing; the speed table
   compares against each reference's fastest measured 1/2/4-bit configuration reaching that target.
3. **Build cost:** fresh-process build peaks, total build time and encoding throughput are reported below.
   The old report's build figures no longer describe this revision.
4. **Crossover at equal steady RAM:** {crossover}.
   The complete per-dataset rows below show losses as well as gains. Claims apply to the measured
   configurations, without extrapolating to IVF or SSD operation.

The benchmark took **{hours:.2f} hours** from preparation start through the final timing cell and diagnostic,
including builds, validation, calibration and interrupted attempts. {acceptance}

## Setup and measurement

| Dataset | Dimensions | Base | Queries | Metric |
|---|---:|---:|---:|---|
| SIFT1M | 128 | 1,000,000 | 1,000 | L2 |
| GloVe-200 | 200 | 1,000,000 | 1,000 | cosine |
| OpenAI-1536 100K | 1,536 | 100,000 | 1,000 | cosine |
| Gemini-768 | 768 | 99,000 | 1,000 | cosine |
| OpenAI-1536 1M | 1,536 | 999,000 | 1,000 | cosine |
| MS MARCO-1M | 1,024 | 1,000,000 | 1,000 | cosine, real dev questions |

Hartmut: Intel i9-7980XE, 18 physical cores, 62 GiB RAM. Native code uses `-O3 -march=native
-ffp-contract=off`; TieredQ's recorded fast-scan path is {scan_paths}, RaBitQ uses its native x86 SIMD
path, and Qdrant TQ+ its production AVX2
scorer through the preserved Rust shim. RaBitQ is pinned to `ced3763857e7e4e48553951bf66fabc8517bc4f0`,
direct flat FHT, implicit row IDs, high-accuracy scoring and native extra-bit bounds. Qdrant is pinned to
`222b2e44ab1c3f17e8246396ead6558e45041df7`, with seeded calibration and the existing adapter.
Exact compiler commands, source hashes, binary hashes, input hashes and process snapshots are in the bundle.

Data and exact FP32 truth were reused from cached prepared arrays. Missing SIFT/GloVe containers were
recovered from retained inputs; all three recovered arrays match the archived #4 hashes byte for byte.
SIFT truth was regenerated with the same exact algorithm and verified by its archived hash.
The primary GloVe base expands the original 100K subset to 1M cached source vectors, keeping
the same byte-identical queries and including every original base vector. Its exact FP32 truth
was recomputed. The original 100K results are retained below as a scaling appendix.
Recall@10 is mean ± sample standard
deviation over seeds 0–2; Qdrant's seeded calibration is fixed across those benchmark seeds. Reference
recall/storage cells were reused only after source and provenance checks; fresh seed-0 reference builds
verify identical recall and serialized bytes. Their timing and build costs are refreshed here.
GloVe-1M reference cells were measured fresh on all three seeds, without reusing 100K measurements.

Matching and recall searches use CPUs 8–31 in three lanes; timing follows after all lanes exit, on CPUs
0–7, with their SMT siblings 18–25 free of benchmark work. Timing uses seed 0, threads 1/8 × batches 1/32,
1,000 queries, three warmup calls and three complete repeats.
Builds use eight codec threads with BLAS restricted to one thread. Cosine inputs are cached, normalized
FP32 vectors; data loading, embedding generation and dataset preparation are outside search timing.
End-to-end search includes query preparation, scan, fine-tier rescoring when enabled, and top-k.
Batch-1 p50/p95 measure single-query response latency;
batch-32 latency is a batch response time, not per-query amortized latency. No FP32 reranking is used.

Serialized bytes include headers, model/shared state, metadata and padding. Effective bytes/vector = total
bytes / N; bits/vector = eight times that; compression versus FP32 = 32d / measured bits/vector.
Steady and peak serving RAM are RSS growth from before reload; peak is the maximum over reload and all
four serving schedules. The raw diagnostics also retain absolute process RSS. Build time covers training,
encoding and packing, excluding data loading, truth generation and the explicit serialization pass.
Qdrant's native constructor also writes its required output metadata/codes, so its build time includes
that work. Build peak growth excludes
the already-loaded FP32 corpus; absolute build process peak includes it. Both are reported, measured in a
fresh process. Encoding throughput uses each codec adapter's encoding interval.

Each equal-memory target is the minimum reference storage across seeds. TieredQ is calibrated using seed-0
storage only, then confirmed on all three seeds, stepping downward if necessary. Its maximum across seeds
must fit. The 1/2/4-bit labels identify reference code widths; matching uses measured serialized bytes
or RSS, never nominal code bits alone. Packing discontinuities and RSS noise can leave unused budget;
the supplementary tables show both actual usages. Disk-matched, steady-RAM-matched and peak-RAM-matched
rows are separate comparisons and can select different TieredQ budgets.
Equal-recall bisection explores seed 0, then budgets step upward until the three-seed mean reaches
the reference. Near-monotonicity is an empirical search assumption; “smallest evaluated” is not a proof of
a global minimum. Reference storage at 0.95 is limited to the evaluated 1/2/4-bit configurations.

## Accuracy at equal memory

TieredQ columns show mean ± std and Δ recall points versus the reference. All three resource targets
are distinct; complete target budgets and bytes are in [the supplementary tables]({link}/tables.md).

'''
    report += table(['Dataset','Reference','Reference nominal bits/dim','Reference R@10','TieredQ disk (Δ pt)',
                     'TieredQ steady RAM (Δ pt)','TieredQ peak RAM (Δ pt)'],recall_rows)
    losses=[]
    for (ds,codec,bits,metric),point in matches.items():
        if point['tieredq'] and point['tieredq']['recall10']['mean']<point['reference']['recall10']['mean']:
            losses.append([NAMES[ds],CODECS[codec],bits,metric,
                f"{point['tieredq']['recall10']['mean']:.4f} / {point['reference']['recall10']['mean']:.4f}",
                f"{100*(point['tieredq']['recall10']['mean']-point['reference']['recall10']['mean']):+.2f}"])
    if losses:
        report += '\n\nObserved losses at the matched resource limits:\n\n'
        report += table(['Dataset','Reference','Nominal bits/dim','Matched resource','Recall TQ / ref','Δ points'],losses)
    report += f'\n\n![Recall versus complete serialized storage]({link}/recall-storage.png)\n'
    report += '\n## Smallest evaluated storage at Recall@10 ≥ 0.95\n\n'
    report += 'All rows require three measured seeds. Budgets are code bits for TieredQ and bits/dim for references.\n\n'
    report += table(['Dataset','Codec','Budget','Disk bits/vector','Bytes/vector','FP32 compression','Recall@10'],minimum_rows)
    report += '\n\nAt the 0.95 target, TieredQ uses that smallest evaluated storage point. '
    report += 'Reference QPS uses the fastest measured budget/schedule reaching at least 0.95; '
    report += 'batch-1 p50 uses the best thread setting at TieredQ’s selected budget and across eligible reference budgets. '
    report += 'Reference recall can overshoot 0.95 because only 1/2/4-bit configurations are evaluated.\n\n'
    report += table(['Dataset','Reference','TieredQ code bits','Best QPS TQ / ref','QPS schedules TQ / ref','Best batch-1 p50 ms TQ / ref'],speed95)
    report += '\n\n## Speed at equal steady serving RAM\n\nQPS is 8 threads, batch 32; p50 is 1 thread, batch 1. Values are TieredQ / reference.\n\n'
    report += table(['Dataset','Reference','bits/dim','TieredQ code bits','QPS (ratio)','p50 ms'],speed_rows)
    report += '\n\n## Speed and storage at equal recall\n\n'
    report += 'QPS ratios use the fastest reference budget reaching each target separately for each schedule; '
    report += 'the selected reference bits/dim are in parentheses. p50 uses the fastest eligible reference budget at 1t·b1.\n\n'
    report += table(['Dataset','Reference','bits/dim','Reference / TieredQ R@10','TQ code bits','Disk bits/vector TQ / ref',
                     'QPS ratio 1t·b32','QPS ratio 8t·b32','p50 1t·b1 ms TQ / fastest eligible ref'],equal_rows)
    report += f'\n\n![Equal-recall QPS versus the fastest reference budget reaching the target]({link}/equal-recall-speed.png)\n'
    report += '\n\n## Query preparation cost\n\n'
    report += 'A separate diagnostic adapter replays each codec’s native query preparation on the same reloaded index, '
    report += 'without scanning or top-k. It runs after headline timing and checks identical search IDs for 32 queries. '
    report += 'The measured library and production Qdrant shim remain unchanged. Diagnostic time includes adapter call overhead; '
    report += 'isolated replay has different cache behavior from a full scan, so percentages are diagnostic estimates. '
    report += 'batch figures are divided by query count only for this decomposition, not reported as response latency. '
    report += 'The table summarizes configurations where preparation exceeds 10% of end-to-end mean time; '
    report += 'all configurations and four schedules are retained in `preparation.csv`.\n\n'
    prep_rows=[]
    for ds in NAMES:
        for codec in CODECS:
            points=[r for r in preparations if r['dataset']==ds and r['codec']==codec and r['fraction_of_end_to_end']>.1]
            if points:
                ms=[r['preparation_ms_per_query'] for r in points]
                share=[100*r['fraction_of_end_to_end'] for r in points]
                prep_rows.append([NAMES[ds],CODECS[codec],len(points),f'{min(ms):.4f}–{max(ms):.4f}',f'{min(share):.1f}–{max(share):.1f}%'])
    report += table(['Dataset','Codec','Configurations/schedules >10%','Preparation ms/query','Share of mean end-to-end'],prep_rows)
    report += '\n\n## Build cost\n\n'
    report += 'Build time and encode throughput come from seed-0 builds in the serialized timing session. '
    report += 'Memory ranges cover fresh-process evaluated TieredQ budgets/seeds, including calibration and equal-recall configurations; '
    report += 'reference memory ranges cover fresh seed-0 builds at 1/2/4 bits/dim. RSS growth and absolute process peak '
    report += 'are in bits per vector. These are build-phase peaks, distinct from serving RAM.\n\n'
    report += table(['Dataset','Codec','Memory builds','Timed budgets','Build seconds (alone)','Encode vectors/s (alone)','Build peak growth bits/vector',
                     'Absolute build peak bits/vector'],build_rows)
    report += '\n\n## Real versus document queries on MS MARCO\n\n'
    report += 'The same base, codec, budget and index are scored with real dev questions and held-out passages. '
    report += 'This isolates query distribution; these are recalls against exact embedding neighbors, not relevance judgments. '
    report += 'TieredQ rows use the indicated equal-disk budget. Only real queries are in the refreshed timing matrix.\n\n'
    report += table(['Codec','Budget','Real-query R@10','Document-query R@10'],query_rows)
    report += '\n\n## Appendix: GloVe-100K and scaling\n\n'
    report += ('The original 100K-vector experiment uses the same queries and a subset of the primary 1M base. '
               'The table retains its three-seed resource matches. Absolute recall across base sizes has different '
               'exact neighbors; compare the codecs within each size. The 100K timings from the interrupted '
               'session are excluded from headline speed comparisons.\n\n')
    small=[]
    for bits in (1,2,4):
        for codec in ('rabitq','qdrant'):
            points=[p for p in json.loads((w/f'match-b{bits}/matches-{codec}.json').read_text()) if p['dataset']=='glove']
            by_metric={p['metric']:p for p in points}
            reference=points[0]['reference']['recall10']
            row=[CODECS[codec],bits,f"{reference['mean']:.4f} ± {reference['std']:.4f}"]
            for metric in ('disk','ram','peak'):
                recall=by_metric[metric]['tieredq']['recall10']
                row.append(f"{recall['mean']:.4f} ± {recall['std']:.4f}")
            small.append(row)
    report += table(['Reference','Nominal bits/dim','Reference R@10','TieredQ disk','TieredQ steady RAM','TieredQ peak RAM'],small)
    report += ('\n\nIn the 100K seed-0 diagnostic, TieredQ’s query-phase peak rises about 1.6 MB above '
               'post-reload RSS, versus 0.52 MB in the fresh 1-bit TQ+ check. This consumes a large share '
               'of the small-dataset budget. The 1M experiment measures the larger-scale outcome rather '
               'than subtracting overhead. The investigation is tracked in '
               '[issue #38](https://github.com/SeanPedersen/TieredQ/issues/38); phase measurements are retained.\n')
    report += f'''\n\n## Scope and review artifacts

The host is shared; affinity prevents competing benchmark jobs in this run, but does not remove operating-system
or other tenant noise. Memory is observed RSS, so allocator/page behavior can move a budget near a target.
This compares explicit in-process adapters, not complete database requests. Qdrant's AVX2 path on this host
limits how far its speed ratios generalize. The report retains all raw p50/p95 and QPS schedules, rather than
treating batch throughput as single-query latency.

`TieredQNestedIndex` is a separate existing single-query path. Historical #13 measurements from
[2026-11-03](../benchmark/results/issue13-nested-20261103/README.md) reported 1t·b1 QPS of 1,251 / 574
against RaBitQ at 100K OpenAI vectors, and 142 / 49 at 1M, with TieredQ / RaBitQ recall of
0.9686 / 0.9674 and 0.9725 / 0.9700 respectively. These historical numbers are not refreshed or mixed
into the current flat-index comparisons. IVF, SSD, external
reranking, nested-index retiming and release packaging are out of scope.

The [results bundle]({link}/README.md) contains the command plan, exit markers, logs, hashes, source provenance,
all calibration probes and seeds, complete [memory and timing tables]({link}/tables.md), and machine-readable CSVs.
'''
    (w/'benchmark-report.md').write_text(report)
    (w/'runtime.json').write_text(json.dumps(dict(started_utc=start.isoformat(), finished_utc=end.isoformat(),
                                                benchmark_wall_s=(end-start).total_seconds()),indent=2))
    plots(w, grouped, timing, equal)


def plots(w, grouped, timing, equal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,ds in zip(axes.flat,NAMES):
        for codec in CODECS:
            points = sorted((max(float(r['disk_bits_per_vector']) for r in rows),
                             st.mean(float(r['recall10']) for r in rows), st.stdev(float(r['recall10']) for r in rows))
                            for (dataset,name,_),rows in grouped.items() if dataset==ds and name==codec)
            ax.errorbar([p[0]/DIMS[ds] for p in points],[p[1] for p in points],yerr=[p[2] for p in points],
                        fmt='o-',markersize=3,label=CODECS[codec],linewidth=1)
        ax.axhline(.95,color='gray',linestyle=':',linewidth=.8)
        ax.set(title=NAMES[ds],xlabel='Serialized bits / original dimension',ylabel='Recall@10')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.savefig(w/'recall-storage.png',dpi=180)
    fig.savefig(w/'recall-storage.pdf')
    plt.close(fig)
    import numpy as np
    lookup={(r['dataset'],r['reference_codec'],r['reference_bits']):r for r in equal}
    values=np.full((12,3),np.nan)
    labels=[]
    budgets={}
    for i,(ds,codec) in enumerate((ds,codec) for ds in NAMES for codec in ('rabitq','qdrant')):
        labels.append(NAMES[ds]+' / '+CODECS[codec])
        for j,bits in enumerate((1,2,4)):
            point=lookup[ds,codec,bits]
            if not point['reached']:
                continue
            candidates=[budget for budget in (1,2,4) if st.mean(float(r['recall10']) for r in grouped[ds,codec,budget])
                        >=point['reference_recall10']-1e-12]
            best=max(candidates,key=lambda budget:float(timing[ds,codec,budget,8,32]['qps']))
            values[i,j]=float(timing[ds,'tieredq',point['code_bits'],8,32]['qps'])/float(timing[ds,codec,best,8,32]['qps'])
            budgets[i,j]=best
    fig,ax=plt.subplots(figsize=(11,7),constrained_layout=True)
    plot=ax.imshow(np.log2(values),cmap='RdYlGn',vmin=-4,vmax=4,aspect='auto')
    ax.set_xticks([0,1,2],['1 bit/dim','2 bits/dim','4 bits/dim'])
    ax.set_yticks(range(12),labels)
    ax.set_xlabel('Reference recall level; seed-0 timing, 8 threads, batch 32')
    ax.set_title('TieredQ / fastest reference QPS reaching the target recall')
    for i in range(12):
        for j in range(3):
            text=f'{values[i,j]:.2f}× (vs {budgets[i,j]}b)' if (i,j) in budgets else 'unreached'
            ax.text(j,i,text,ha='center',va='center',fontsize=9)
    colorbar=fig.colorbar(plot,ax=ax,ticks=[-4,-2,0,2,4])
    colorbar.ax.set_yticklabels(['1/16×','1/4×','1×','4×','16×'])
    fig.savefig(w/'equal-recall-speed.png',dpi=180)
    fig.savefig(w/'equal-recall-speed.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
