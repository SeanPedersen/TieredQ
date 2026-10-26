"""Audit exact-budget supplemental rows and report measured memory-anchor pairs."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CORE=ROOT.parent/'issue4-current-single-seed-20261026'


def main():
    rows=list(csv.DictReader((ROOT/'results.csv').open()))
    core=list(csv.DictReader((CORE/'results.csv').open()))
    meta=json.loads((ROOT/'meta.json').read_text())
    assert meta['status']=='complete' and meta['cpu_affinity']==list(range(8))
    assert meta['commit']=='3c9dbf590209dabf8ecccd3385fd56254779c70d'
    assert len(rows)==12
    actual={(int(r['rabitq_anchor_bits']),int(r['threads']),int(r['batch'])) for r in rows}
    assert actual==set(itertools.product((2,4,8),(1,8),(1,32)))
    source=json.loads((ROOT/'source-sha256.json').read_text())
    original=json.loads((CORE/'meta.json').read_text())['source_sha256']
    assert all(h==original[p] for p,h in source.items() if p.startswith(('src/','include/')))
    assert meta['data_manifest']==json.loads((CORE/'data-verification.json').read_text())['openai']['manifest']
    assert len(meta['compiler_commands'])==5
    for c in meta['compiler_commands']:
        args=c['arguments'];assert [a for a in args if a.startswith('-O')][-1]=='-O3'
        assert '-march=native' in args and '-ffp-contract=off' in args
    comparisons=[]
    for anchor in (2,4,8):
        group=[r for r in rows if int(r['rabitq_anchor_bits'])==anchor]
        for key in ('recall10','effective_bytes_per_vector','build_s','encode_vps','target_code_bits'):
            assert len({r[key] for r in group})==1
        r=next(r for r in group if r['threads']=='8' and r['batch']=='32')
        rb=next(r for r in core if r['dataset']=='openai' and r['codec']=='rabitq' and r['threads']=='8' and r['batch']=='32' and int(r['target_bits'])==anchor)
        used=float(r['effective_bytes_per_vector']);limit=float(rb['effective_bytes_per_vector'])
        assert used<=limit and (limit-used)/limit<=meta['storage_tolerance']
        comparisons.append({'rabitq':rb,'tieredq':r,'memory_shortfall_percent':100*(limit-used)/limit,'recall_delta':float(r['recall10'])-float(rb['recall10']),'qps_ratio':float(r['qps'])/float(rb['qps'])})
    for r in rows:
        assert (r['dataset'],r['codec'],r['seed'],r['base_n'],r['queries'],r['d'])==('openai','tieredq','0','100000','1000','1536')
        for key in ('recall10','qps','build_s','encode_vps','p50_ms','p95_ms','effective_bytes_per_vector'):
            assert math.isfinite(float(r[key])) and float(r[key])>0
        assert math.isclose(float(r['target_bits']),int(r['target_code_bits'])/1536)
    (ROOT/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    audit={'unique_rows':len(actual),'full_dataset_identity_verified':True,'codec_sources_identical_to_core':True,'finite_timings':True,'scheduling_invariants_verified':True,'all_points_within_anchor':True,'max_memory_shortfall_percent':max(c['memory_shortfall_percent'] for c in comparisons),'compiler_commands_audited':5,'single_seed':0,'statistical_confidence_claim':False}
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# OpenAI3-1536: additional memory-aligned TieredQ points','',
        'Three newly measured TieredQ fast-mode points sit within 0.5% below the current RaBitQ 2/4/8-bit serving-memory anchors. Full 100K base vectors, 1,000 held-out queries, seed 0, CPUs 0–7 on hartmut; threads 1/8 and batches 1/32, three warmup queries and three timing repeats. Single-seed results provide no statistical-confidence claim. RaBitQ rows are reused from the [completed current core run](../issue4-current-single-seed-20261026/README.md), not remeasured simultaneously. Native bounds/HACC remain enabled in those anchors.','',
        'The existing TieredQ total-code-bit API supports finer budgets than the original whole-bit CLI grid. A benchmark-only constructor overload exposes that API; codec source, precision and routing are unchanged. Storage-only calibration selects the first point under each anchor within 0.5%, without consulting recall. This is not an exhaustive search for an optimum or an exact-equal-memory claim. All calibration attempts are retained in `calibration.json`.', '',
        '## Primary comparison', '', 'QPS is 8 threads/batch 32; construction includes training and encoding/packing. Encode throughput times native add/encode/pack. Serving storage is freshly measured as max(saved bytes, reloaded index/cache + serving model bytes). No historical memory totals are reused for the new points.', '',
        '| RaBitQ bits | Codec | Code bits/vector | Nominal bits/dim | Bytes/vector | Recall@10 | QPS | Build s | Encode vectors/s |', '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in comparisons:
        for name in ('rabitq','tieredq'):
            r=c[name];code=r.get('target_code_bits',str(int(float(r['target_bits'])*1536)));nominal=float(r['target_bits'])
            lines.append(f'| {r.get("rabitq_anchor_bits",r["target_bits"])} | {name} | {code} | {nominal:.4f} | {float(r["effective_bytes_per_vector"]):.3f} | {float(r["recall10"]):.4f} | {float(r["qps"]):,.0f} | {float(r["build_s"]):.3f} | {float(r["encode_vps"]):,.0f} |')
    lines += ['', '## New single-query latencies', '', 'Batch 1, complete response latency in milliseconds.', '', '| RaBitQ memory anchor | TieredQ bytes/vector | 1 thread p50 / p95 | 8 threads p50 / p95 |', '|---:|---:|---:|---:|']
    for c in comparisons:
        r=c['tieredq'];cells=[]
        for threads in ('1','8'):
            x=next(x for x in rows if x['rabitq_anchor_bits']==r['rabitq_anchor_bits'] and x['threads']==threads and x['batch']=='1')
            cells.append(f'{float(x["p50_ms"]):.3f} / {float(x["p95_ms"]):.3f}')
        lines.append(f'| {float(r["rabitq_anchor_bytes"]):.3f} | {float(r["effective_bytes_per_vector"]):.3f} | {cells[0]} | {cells[1]} |')
    lines += ['', '## Measured outcome', '']
    for c in comparisons:
        r=c['tieredq'];rb=c['rabitq'];lines.append(f'- At the {float(rb["effective_bytes_per_vector"]):.3f}-byte anchor, TieredQ uses {c["memory_shortfall_percent"]:.3f}% less memory, has recall difference {c["recall_delta"]:+.4f}, and {c["qps_ratio"]:.2f}× RaBitQ throughput. Separate-run timing variation is not quantified.')
    lines += ['', '## Why the 402-byte point trails RaBitQ', '', 'This continues the historical OpenAI pattern rather than showing a newly observed recall regression. The original issue #4 TieredQ 1-bit point used 375.762 bytes/vector for recall 0.8939, while RaBitQ 2-bit used 404.070 bytes/vector for recall 0.9041. The current original-grid TieredQ point uses 372.063 bytes/vector at the same recall; the new 402.199-byte point improves it to 0.8991.', '', 'At the new point, measured serving storage decomposes into 241.000 bytes/vector for the fast vector representation, 88.813 bytes/vector of additional fast-index/cache allocations, and 72.386 bytes/vector of reloaded serving model state. Total: 402.199. Shared/cache allocations are amortized over only 100K vectors. The nominal residual-code budget is 1,695 bits/vector (1.104 bits/dimension), allocated over 918 PCA ranks, whereas RaBitQ fits a nominal 2-bit representation at almost the same total memory. This explains why equal nominal-bit recall advantages cannot be assumed at equal serving memory; these measurements do not isolate every cause of the remaining recall difference.', '']
    lines += ['', f'Total supplemental wall time, including storage calibration and selected builds/searches: **{meta["total_wall_s"]:.2f} s**. Selected construction plus all search measurements: **{meta["selected_build_and_search_wall_s"]:.2f} s**, excluding storage calibration, serialization/reload, compilation and dataset preparation. This supplemental time is separate from the original 1,076.91-second core matrix.', '',
        '## Verification and reproduction', '',
        '- `audit.json`: 12 unique rows, full original dataset manifest and base/query hashes, finite timings, invariant recall/storage/build/encode across scheduling rows, and every selected point within its measured RaBitQ anchor.',
        '- `validation.log`: all 16 native validation tests pass. `meta.json` records all five audited compiler commands, native binary hash, affinity and exact selected budgets. Effective flags are `-O3 -march=native -ffp-contract=off`; nanobind uses `NOMINSIZE`.',
        '- `source-sha256.json`, `exact-budget-binding.patch`, `provenance.json`: committed source is `3c9dbf5`; all codec source hashes match the core run. Fresh isolated build/environment at `sean@hartmut:/home/sean/experiments/TieredQ-openai-aligned-20261026-akXzMJ`. Historical workspaces and results remain intact.',
        '- `run_aligned.py` and `setup-aligned.py` preserve the calibration and measurement procedure. Use the committed archive in the sibling core bundle, apply the binding patch, provide the verified caches, run `./validate.sh`, then run the driver with the native build on `PYTHONPATH`, CPUs 0–7 and BLAS threads fixed to one. No normal mode, seed sweep, kernel changes or profiling.', '',
        '![Measured OpenAI recall and storage](recall-storage-aligned.png)', '']
    (ROOT/'README.md').write_text('\n'.join(lines))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
