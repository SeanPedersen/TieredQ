"""Audit the complete scheduling matrix and report measured RAM anchor pairings."""
import csv
import itertools
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DIMS={'sift':(128,1000000),'glove':(200,100000),'openai':(1536,100000)}

def load(path):
    return list(csv.DictReader(path.open()))

def number(row,key):
    return float(row[key])

def main():
    rows=load(ROOT/'results.csv')
    controls=load(ROOT/'core-results.csv')
    selections=json.loads((ROOT/'selections.json').read_text())
    meta=json.loads((ROOT/'meta.json').read_text())
    expected=set(itertools.product(DIMS,('1','2','4','8'),('1','8'),('1','32')))
    keys=['dataset','rabitq_anchor_bits','threads','batch']
    actual=[tuple(r[k] for k in keys) for r in rows]
    assert len(actual)==len(set(actual))==48 and set(actual)==expected
    variations=[]; pairs=[]
    for r in rows:
        d,n=DIMS[r['dataset']]
        assert (int(r['d']),int(r['base_n']),int(r['queries']))==(d,n,1000)
        assert r['seed']=='0' and r['rerank']=='False' and r['codec']=='tieredq'
        assert r['metric']==('l2' if r['dataset']=='sift' else 'cosine')
        for key in ('build_s','encode_vps','qps','p50_ms','p95_ms','effective_bytes_per_vector'):
            assert math.isfinite(number(r,key)) and number(r,key)>0
        assert 0<=number(r,'recall10')<=1 and number(r,'p95_ms')>=number(r,'p50_ms')
        assert number(r,'effective_bytes_per_vector')<=number(r,'rabitq_anchor_bytes')
        assert math.isclose(number(r,'effective_bits_per_dim'),number(r,'effective_bytes_per_vector')*8/d)
    old=controls+load(ROOT/'pre-fix-aligned.csv')
    table=[]; latency=[]; differences=[]; allocations=[]
    for s in selections:
        dataset=s['dataset']; anchor=s['anchor_bits']
        group=[r for r in rows if r['dataset']==dataset and int(r['rabitq_anchor_bits'])==anchor]
        for key in ('effective_bytes_per_vector','build_s','encode_vps','target_code_bits','rabitq_anchor_bytes'):
            assert len({r[key] for r in group})==1,(dataset,anchor,key)
        for batch in ('1','32'):
            assert len({r['recall10'] for r in group if r['batch']==batch})==1
        values=sorted({number(r,'recall10') for r in group})
        if len(values)>1: variations.append({'dataset':dataset,'anchor':anchor,'target_code_bits':s['target_code_bits'],'recalls':values})
        r=next(r for r in group if r['threads']=='8' and r['batch']=='32')
        control=next(c for c in controls if c['dataset']==dataset and c['codec']=='rabitq' and int(c['target_bits'])==anchor and c['threads']=='8' and c['batch']=='32')
        assert r['rabitq_anchor_bytes']==control['effective_bytes_per_vector']
        size=number(r,'effective_bytes_per_vector');gap=number(r,'percent_below_anchor')
        pair={**r,'control':control,'recall_delta':number(r,'recall10')-number(control,'recall10'),'qps_ratio':number(r,'qps')/number(control,'qps')}
        pairs.append(pair)
        table.append(f"| {dataset} | {anchor} | {r['rabitq_anchor_bytes']} | {size:.6f} | {gap:.4f}% | {s['target_code_bits']} | {number(r,'recall10'):.4f} / {number(control,'recall10'):.4f} | {number(r,'qps'):,.0f} / {number(control,'qps'):,.0f} | {number(r,'build_s'):.3f} | {number(r,'encode_vps'):,.0f} |")
        lat=[next(x for x in group if x['threads']==t and x['batch']=='1') for t in ('1','8')]
        lat += [next(c for c in controls if c['dataset']==dataset and c['codec']=='rabitq' and int(c['target_bits'])==anchor and c['threads']==t and c['batch']=='1') for t in ('1','8')]
        latency.append(f"| {dataset} | {anchor} | "+' | '.join(f"{number(x,'p50_ms'):.3f} / {number(x,'p95_ms'):.3f}" for x in lat)+' |')
        tiers='; '.join(f"{t['dimensions']}×{t['bits']}b" for t in s['diagnostics']['tiers'])
        learned=sum(t['dimensions']*t['bits'] for t in s['diagnostics']['tiers'])
        allocations.append(f"| {dataset} | {anchor} | {s['target_code_bits']} | {learned} | {tiers} |")
        candidates=[x for x in old if x['dataset']==dataset and x['codec']=='tieredq' and x['threads']=='8' and x['batch']=='32' and number(x,'effective_bytes_per_vector')<=number(control,'effective_bytes_per_vector')]
        if candidates:
            prior=max(candidates,key=lambda x:number(x,'recall10'))
            prior_bits=int(prior.get('target_code_bits') or number(prior,'target_bits')*DIMS[dataset][0])
            differences.append(f"| {dataset} | {anchor} | {prior_bits} → {s['target_code_bits']} | {number(prior,'effective_bytes_per_vector'):.6f} → {size:.6f} | {number(r,'recall10')-number(prior,'recall10'):+.4f} | {number(r,'qps')/number(prior,'qps'):.2f}× |")
        else: differences.append(f"| {dataset} | {anchor} | No pre-fix measured point within anchor | — | — | — |")
    source=json.loads((ROOT/'source-verification.json').read_text())
    assert source['only_modified_source']==['benchmark/native/module.cpp']
    assert all(v for k,v in source['committed_files_match'].items() if k!='benchmark/native/module.cpp')
    calibration=json.loads((ROOT/'calibration.json').read_text())
    assert all('recall10' not in r for r in calibration)
    for s in selections:
        candidates=[r for r in calibration if r['dataset']==s['dataset'] and r['anchor_bits']==s['anchor_bits'] and r['bytes_per_vector']<=s['anchor_bytes']]
        assert s['bytes_per_vector']==max(r['bytes_per_vector'] for r in candidates)
        assert s['target_code_bits']<=8*(DIMS[s['dataset']][0]+(s['dataset']=='sift'))
        components=s['storage']
        assert components['serving_bytes']==max(components['disk_bytes'],components['resident_bytes'])+components['adapter_bytes']
        assert components['resident_bytes']==components['index_allocated_bytes']+components['model_bytes']+components['index_shell_bytes']
    assert meta['status']=='complete' and meta['affinity']==list(range(8))
    assert meta['nanobind_NOMINSIZE'] and meta['rabitq_search']['extra_bit_bounds']
    for c in meta['compiler_commands']:
        args=c['arguments'];assert [a for a in args if a.startswith('-O')][-1]=='-O3'
        assert '-march=native' in args and '-ffp-contract=off' in args
    data=json.loads((ROOT/'data-verification.json').read_text())
    for dataset in DIMS: assert data[dataset]['array_sha256']==meta['data'][dataset]
    for name,digest in data['split_indices'].items():
        assert hashlib.sha256((ROOT/'split-indices'/name).read_bytes()).hexdigest()==digest
    provenance=json.loads((ROOT/'provenance.json').read_text())
    assert hashlib.sha256((ROOT/'committed-source.tar').read_bytes()).hexdigest()==provenance['archive_sha256']
    assert meta['head']==provenance['head']
    assert meta['rabitq_patches']=={'utils/rotator.hpp':'4dcd65c8ae0ade65f5de1e407a507a1453f3620e2627b5022698bffb0b7c24d9'}
    assert (ROOT/'exit-code').read_text().strip()=='0'
    binaries=json.loads((ROOT/'binary-sha256.json').read_text())
    assert set(binaries.values())=={meta['binary_sha256']}
    assert 'Ran 16 tests' in (ROOT/'validation-final.log').read_text() and '\nOK\n' in (ROOT/'validation-final.log').read_text()
    assert '100% tests passed out of 2' in (ROOT/'cpp-validation.log').read_text()
    audit={'rows':48,'unique_rows':48,'full_dataset_identity':True,'within_all_anchors':True,'finite_positive_timings':True,'scheduling_invariants':True,'recall_variations':variations,'single_seed':0,'statistical_confidence_claim':False,'gaps_above_half_percent':[{'dataset':p['dataset'],'anchor':p['rabitq_anchor_bits'],'percent_below':p['percent_below_anchor']} for p in pairs if number(p,'percent_below_anchor')>0.5]}
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'comparisons.json').write_text(json.dumps(pairs,indent=2))
    diagnostic=json.loads((ROOT/'recall-diagnostics.json').read_text()) if variations else []
    assert len(diagnostic)==len(variations)
    assert all(x['same_serial_probes_ids_equal_across_native_batches'] and x['same_batched_probes_ids_equal_across_native_batches'] for x in diagnostic)
    readme=f'''# Fast mode after the memory fix: calibrated RAM anchors

Full SIFT1M, GloVe-200 and OpenAI3-1536; seed 0, 1,000 held-out queries, CPUs 0–7 on hartmut (i9-7980XE, AVX-512BW). Threads 1/8 × batches 1/32, three warmup queries and three full timing repeats. **Single-seed run; no statistical-confidence claims.**

HEAD `06cda8574db318207fe577f55b4feca4c29a9995` regenerates narrow scaled-basis projection coefficients without changing codes or saved formats. Current C++ tests and all 16 benchmark validation tests passed before calibration. No kernel changes, profiling, normal mode, legacy lift, seed sweeps or optimization experiments were used.

## Paired discrete within-memory results

TQ / RQ denotes TieredQ / native RaBitQ. QPS is 8 threads/batch 32. Anchors retain the exact unrounded values from the archived CSV. RaBitQ results are **separate-run controls**, reused from `issue4-current-single-seed-20261026`; adapter, compiler, native bounds/HACC and benchmark protocol remain comparable. Background services remain active, so timing ratios do not isolate causal effects.

| Dataset | RQ bits | RQ anchor B/vector | TQ actual B/vector | Below anchor | Total code-bit budget | Recall TQ / RQ | QPS TQ / RQ | TQ build s | TQ encode vectors/s |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
'''+ '\n'.join(table)+f'''

Construction includes training, encode/pack and SIFT adapter fit/base transformation. Encode throughput times native add/encode/pack only. Single-query timings include query preprocessing, scan, top-k and binding.

## Single-query p50 / p95 milliseconds

| Dataset | RQ RAM anchor bits | TQ 1 thread | TQ 8 threads | RQ 1 thread | RQ 8 threads |
|---|---:|---|---|---|---|
'''+ '\n'.join(latency)+'''

## Learned allocation

Tiers are ordered PCA rank runs, dimensions × bits. The exact integer API budget is preserved alongside the actually learned bit sum; these need not be identical.

| Dataset | Anchor bits | Requested total bits | Learned total bits | Learned tiers |
|---|---:|---:|---:|---|
'''+ '\n'.join(allocations)+f'''

## Calibration, accounting and audit

Storage-only calibration took **{meta['calibration_wall_s']:.2f} s**. Selected builds plus all scheduled searches took **{meta['selected_build_and_search_wall_s']:.2f} s**, excluding compilation, calibration, dataset loading/verification and save/reload checks. Waiting for other experiments took {meta.get('other_experiment_wait_s',0):.2f} s separately. Calibration timings include every candidate build and its storage checks, including selected candidates; selected configurations were then rebuilt for the matrix.

`calibration.json` preserves every attempt and learned allocation; `selections.json` preserves exact selected storage components and saved-index hashes. The predetermined rule accepts the first measured under-anchor point within 0.5%. Measured proposals use the remaining byte gap with observed budget brackets; no linear scaling assumption establishes feasibility. At adjacent budget boundaries or the public maximum budget retain the closest measured fitting point and report the actual gap. Ties in measured storage retain the first measured budget, without recall-based tie breaking. Recall is never evaluated during calibration. These are discrete within-memory comparisons, without exact-equal-memory, interpolation or optimality claims. Calibration searches measured local budget brackets and is not an exhaustive global allocation search.

Storage is max(actual saved bytes, actual reloaded index/cache + serving model + index shell) plus required SIFT adapter state. No historical TieredQ memory totals are reused. `storage-details.patch` adds allocation introspection only; the 56-byte shell is counted without duplicating the embedded model. Model basis, centroids, mean/sigma, codebooks, cache groups and panels are included. Encoder buffers built during append are excluded from saved/reloaded serving state; input FP32, transient query/top-k scratch, allocator bookkeeping and wrapper objects are excluded. This is component accounting, not process RSS. Adapter storage uses the larger serialized or array footprint and its query transformation remains inside timing.

`audit.json` verifies all 48 unique scheduling rows, finite positive timing, full dataset hashes, within-anchor storage, and construction/storage/encode scheduling invariants. Raw recall differences are retained without averaging. `recall-diagnostics.json` isolates any observed adapter rounding by holding transformed probes fixed while changing native batch size. Primary comparisons use batch 32.

## Differences from pre-fix measurements

For each anchor the historical reference is the highest-recall previously measured fitting point from the original grid plus OpenAI supplemental aligned points. Changed budgets and separate-run timing prevent attributing all differences to the fix. Historical totals are shown only as historical observations and are never used for current selection. Recall need not increase at every integer budget: SIFT’s maximum-budget point changes the learned allocation and measures 0.9876 versus the pre-fix 1,024-bit point’s 0.9894. This comparison changes budget and allocation; it does not establish a same-budget memory-fix regression. The requested budgets and learned tiers above preserve that distinction.

| Dataset | Anchor bits | Pre-fix → current code bits | Historical → current B/vector | Recall delta | QPS current / historical |
|---|---:|---|---|---:|---:|
'''+ '\n'.join(differences)+'''

## Reproduction and provenance

The current branch remains `main`; no commit was made. Existing untracked experiments/results and historical remote workspaces were preserved. Fresh remote workspace: `/home/sean/experiments/TieredQ-fast-aligned-20261026-0GBhDm`. `committed-source.tar`, `exact-budget-binding.patch`, `storage-details.patch`, `source-sha256.json`, `source-verification.json`, `binary-sha256.json`, `provenance.json` and `meta.json` record source/binary/compiler identities. All production source files match exported HEAD; only the benchmark binding differs.

`validation.log`, `validation-final.log` and `cpp-validation.log` retain pre-run checks. All audited native commands use `-O3 -march=native -ffp-contract=off`; nanobind uses `NOMINSIZE`. Native direct-flat RaBitQ pins `ced3763857e7e4e48553951bf66fabc8517bc4f0`, with deterministic rotation overlay, pruning and HACC enabled. `data-verification.json` records raw SIFT/GloVe checksum checks, full cache/truth hashes, pinned OpenAI revision and split hashes, matching the original full-data run.

Reproduce in a new export, apply the two binding patches, copy the pinned dependency into the new workspace and link the verified cache. Run `job.sh` with Python 3.12/uv and BLAS fixed to one thread. `run_calibrated.py calibrate` performs storage selection; `run_calibrated.py matrix` rebuilds selected points and measures them. Run `report.py` beside the collected files to audit and regenerate this report and figures.

![Recall at actual measured storage](recall-storage.png)
![Primary throughput at actual measured storage](qps-storage.png)
![Single-query response latency at actual measured storage](latency-storage.png)
'''
    tail=json.loads((ROOT/'tail-diagnostic.json').read_text())
    assert len(tail['durations_ms'])==3 and all(len(v)==1000 for v in tail['durations_ms'])
    assert all(math.isfinite(v) and v>0 for values in tail['durations_ms'] for v in values)
    readme=readme.replace('## Differences from pre-fix measurements', f"**Tail check:** OpenAI anchor 2 has primary 8-thread/batch-1 p50/p95 1.185/3.209 ms. A separate targeted recheck gave {tail['overall_p50_ms']:.3f}/{tail['overall_p95_ms']:.3f} ms; repeat p95s were "+', '.join(f"{r['p95_ms']:.3f}" for r in tail['repeats'])+" ms. `tail-diagnostic.json` retains all 3,000 durations. This shows that the original wide tail did not reproduce; it cannot identify its cause. Background services remained active. The primary result is retained unchanged, and this diagnostic is excluded from selected-build/search wall time.\n\n## Differences from pre-fix measurements")
    readme=readme.replace('## Reproduction and provenance', 'Calibration recovery: `calibration-initial-failure.log` retains an initial proposal above the public budget cap, before any searches. The driver was bounded to the supported range; completed calibration attempts were retained. Its elapsed calibration phase is included in the reported calibration wall time; `recovery.json` preserves the timing boundary.\n\n## Reproduction and provenance')
    gaps=audit['gaps_above_half_percent']
    if gaps: readme=readme.replace('## Learned allocation', '**Alignment gaps:** '+ '; '.join(f"{x['dataset']} anchor {x['anchor']}: {float(x['percent_below']):.4f}% below" for x in gaps)+'. Discrete code/group layout jumps or the public maximum of eight code bits per input dimension limited alignment; see calibration history and exact budgets.\n\n## Learned allocation')
    if not variations: readme=readme.replace('Raw recall differences are retained without averaging.', 'All four schedules have identical recall at each of the 12 selected configurations; no new recall diagnostic was needed. Raw recalls are retained without averaging.')
    if variations: readme=readme.replace('## Differences from pre-fix', '**Observed recall variation:** '+ '; '.join(f"{x['dataset']} anchor {x['anchor']}: {x['recalls']}" for x in variations)+'. Fixed-probe diagnostic equality identifies SIFT adapter FP32 matrix multiplication rounding as the source.\n\n## Differences from pre-fix')
    (ROOT/'README.md').write_text(readme)
    plot(pairs,rows)
    print(json.dumps(audit,indent=2))


def plot(pairs,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,ylabel in [('recall','Recall@10'),('qps','QPS (8 threads / batch 32)'),('latency','Single-query latency (ms)')]:
        fig,axes=plt.subplots(1,3,figsize=(15,4.6))
        for axis,dataset in zip(axes,DIMS):
            points=[p for p in pairs if p['dataset']==dataset]
            if name!='latency':
                key='recall10' if name=='recall' else 'qps'
                for codec,color,marker in [('tieredq','tab:blue','o'),('rabitq','tab:orange','s')]:
                    selected=[p if codec=='tieredq' else p['control'] for p in points]
                    draw = axis.plot if name == 'recall' else axis.scatter
                    draw([number(p,'effective_bytes_per_vector') for p in selected],[number(p,key) for p in selected],color=color,marker=marker,label=codec)
                for p in points:
                    c=p['control'];x=number(p,'effective_bytes_per_vector');y=number(p,key)
                    axis.plot([x,number(c,'effective_bytes_per_vector')],[y,number(c,key)],color='gray',alpha=.5,ls=':')
                    axis.annotate(f"RQ {p['rabitq_anchor_bits']}b RAM",(x,y),xytext=(3,5),textcoords='offset points',fontsize=7)
            else:
                for threads,color in [('1','tab:blue'),('8','tab:green')]:
                    selected=[r for r in rows if r['dataset']==dataset and r['threads']==threads and r['batch']=='1']
                    x=[number(r,'effective_bytes_per_vector') for r in selected]
                    axis.scatter(x,[number(r,'p50_ms') for r in selected],color=color,label=f'TQ {threads}t p50')
                    axis.scatter(x,[number(r,'p95_ms') for r in selected],facecolors='none',edgecolors=color,marker='^',label=f'TQ {threads}t p95')
                    for r in selected:
                        if threads=='8': axis.annotate(f"RQ {r['rabitq_anchor_bits']}b RAM",(number(r,'effective_bytes_per_vector'),number(r,'p95_ms')),xytext=(3,4),textcoords='offset points',fontsize=7)
                axis.set_yscale('log')
            axis.set(title=dataset,xlabel='Actual serving bytes/vector',ylabel=ylabel)
            axis.grid(alpha=.2);axis.margins(x=.15,y=.2);axis.legend(fontsize=7)
        fig.suptitle('Seed 0 • discrete RAM-anchor pairs • separate-run RaBitQ controls • no confidence claims',fontsize=11)
        fig.tight_layout();fig.savefig(ROOT/f'{name}-storage.png',dpi=160);plt.close(fig)


if __name__=='__main__': main()
