"""Generate the SAQ comparison from CSVs without altering the interrupted RaBitQ report."""
from pathlib import Path
from report_data import HERE,REFRESH,NAMES,CODECS,all_points,point,choice
REPO=HERE.parents[2]
REL='../benchmark/results/saq-tieredq-single-cluster-20261027'
OLD='../benchmark/results/issue4-range-refresh-20261027'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    points=all_points();serving=[p for p in points if p['threads']==8 and p['batch']==32]
    main_rows=[];latency=[];construction=[]
    for d,name in NAMES.items():
        for bits in (1,2,4):
            for c,codec in CODECS.items():
                p=point(points,d,c,bits);one=point(points,d,c,bits,1)
                main_rows.append([name,codec,bits,f"{p['recall']:.4f} ± {p['std']:.4f}",f"{p['storage']:.3f}",f"{p['storage']*8:,.1f}",f"{one['qps']:,.0f}",f"{p['qps']:,.0f}"])
                p1=point(points,d,c,bits,1,1);p8=point(points,d,c,bits,8,1)
                latency.append([name,codec,bits,f"{p1['p50']:.3f} / {p1['p95']:.3f}",f"{p8['p50']:.3f} / {p8['p95']:.3f}"])
                if c!='saq-native':construction.append([name,codec,bits,f"{p['build']:.3f}",f"{p['encode']:,.0f}"])
    caps=[];threshold=[]
    for d,name in NAMES.items():
        anchors=[p for p in serving if p['dataset']==d and p['codec'] in ('tieredq','saq') and not p['supplemental']]
        for anchor in sorted(anchors,key=lambda p:p['storage']):
            cells=[]
            for c in CODECS:
                eligible=[p for p in serving if p['dataset']==d and p['codec']==c and p['storage']<=anchor['storage']]
                cells.append(choice(max(eligible,key=lambda p:p['recall'])) if eligible else 'None measured')
            caps.append([name,f"{CODECS[anchor['codec']]} {anchor['bits']:g}b",f"{anchor['storage']:.3f}",*cells])
        for c,codec in CODECS.items():
            eligible=[p for p in serving if p['dataset']==d and p['codec']==c and p['recall']>=.95]
            if not eligible:threshold.append([name,codec,'Not reached','—','—','—']);continue
            p=min(eligible,key=lambda p:p['storage'])
            threshold.append([name,codec,f"{p['code_bits']:,}{'†' if p['supplemental'] else ''}",f"{p['storage']:.3f}",f"{p['recall']:.4f}",f"{p['qps']:,.0f}"])
    t_wins=s_wins=0;ratios=[];losses=[]
    for d in NAMES:
        for bits in (1,2,4):
            t,s,n=(point(points,d,c,bits) for c in CODECS)
            t_wins+=t['storage']<=s['storage'] and t['recall']>=s['recall']
            s_wins+=s['storage']<=t['storage'] and s['recall']>=t['recall']
            ratios.append(t['qps']/s['qps']);losses.append(100*(s['recall']-n['recall']))
    summary=dict(tieredq_nominal_pareto_points=t_wins,saq_nominal_pareto_points=s_wins,
        tieredq_over_saq_full_qps_min=min(ratios),tieredq_over_saq_full_qps_max=max(ratios),
        full_minus_native_recall_pp_min=min(losses),full_minus_native_recall_pp_max=max(losses))
    import json
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text=f'''# SAQ versus TieredQ fast: full-scan CPU comparison

Hartmut i9-7980XE · CPUs 0–7 · 27 October 2026 · no original-vector reranking

## Findings

At the same nominal code budget, TieredQ fast has at least as much mean recall with no more accounted serving storage at **{t_wins} of 12** main-grid points; SAQ full accurate does so at **{s_wins} of 12**. These counts compare measured discrete points and do not imply dominance at unmeasured budgets. TieredQ’s eight-thread batch-32 QPS divided by SAQ full-accurate QPS ranges from **{min(ratios):.2f}× to {max(ratios):.2f}×**. The table retains actual storage and recall alongside throughput.

SAQ’s native pruned scan is a separate operating mode: the full-accurate minus native-pruned mean recall difference ranges from **{min(losses):+.2f} to {max(losses):+.2f} percentage points**. Its faster timings, where observed, must be read with its own recall. The primary full-accurate path computes every compressed vector’s complete score; it does not use SAQ’s variance or progressive pruning.

## What was measured

SAQ is pinned to `2163ebcedd0ad9c9f4de326e6ca7a860f9eafe52`. The adapter uses upstream segmented CAQ encoding, random orthogonal segment rotations, block lookup preparation, full-bit compressed distance estimation, and native top-k selection. SAQ is built as a C++20 shared library; Python 3.12/uv orchestrates calls. This is a one-cluster, fully probed flat adapter, rather than SAQ’s supplied IVF top-100 benchmark. It does not measure multi-cluster SAQ, alternative segment tuning, or the paper’s best ANN operating points.

The four datasets and byte-identical FP32 base/query/truth arrays come from the [audited refresh]({OLD}/output/inputs.json): SIFT1M (1M × 128, L2), GloVe-200 (100K × 200), OpenAI3-1536 (100K × 1,536), and Gemini/DBpedia (99K × 768). All use 1,000 queries and exact top-10 truth. Gemini uses disjoint held-out documents, not separately embedded search queries. The three embedding datasets are normalized; SAQ uses the equivalent L2 ranking on these arrays, preserving the original cosine ground truth.

SAQ preprocessing fits a full orthonormal PCA basis to the entire corpus: centered FP64 covariance, descending eigenvectors, FP32 serving basis/mean, and no PCA dimension truncation. Zero padding brings dimensions to a multiple of 64. Its requested average is scaled by original/padded dimension so the nominal budget refers to the original dimensions. SAQ’s own segmentation can retain zero-bit tails and budgets its own factors. The code allowance therefore remains distinct from measured serving bytes. PCA fitting and base projection are construction work; each native query includes centering, PCA projection and padding. The basis/mean and segment rotations are included in serving storage.

Recall uses seeds **0/1/2**, with **population SD**. Full-corpus PCA is fixed; `srand(seed+1)` seeds upstream Eigen segment rotations, avoiding the C library’s identical `srand(0)`/`srand(1)` streams. Seed 0 supplies timings/storage; seeds 1/2 are recall-only and are copied across scheduling rows. SAQ validates identical serial/batch/thread predictions before timing. Every timed repeat must reproduce all 1,000 reference ID rows.

Both adapters search the full candidate space without a graph, selective IVF probing, FP32 backing vectors or reranking. TieredQ fast retains its native bounds/selection and can avoid some candidate arithmetic; SAQ full accurate performs full-bit scoring for every candidate. Thus throughput compares these explicit end-to-end adapters, rather than equal arithmetic kernels. The separately timed SAQ native-pruned route retains upstream variance-bound parameter 4 and progressive scan behavior. Parallel SAQ work is across queries; eight configured threads do not accelerate batch-1 scans. TieredQ can parallelize a single-query scan.

The retained C++ numerical policy is GCC 16.2.1, **`-O3 -march=native -ffp-contract=off`**, C++20, Release. SAQ’s upstream CMake defaults to `-Ofast`; this experiment deliberately uses the audited TieredQ numerical policy instead. This compiler difference limits comparisons to upstream published timings. Native AVX-512/AVX2 dispatch follows the host. The [compile commands]({REL}/compile_commands.json), binary hashes, dependency identities and overlay are retained. No quantization or scoring formula is replaced.

SAQ has a small [overlay]({REL}/upstream.patch): owned-allocation and encode-timer observations, plus a correction to a single-segment scan that otherwise skips its last block when the vector count is divisible by 32. The correction affects the native-pruned control; the primary full-accurate loop enumerates every vector independently. A last-block regression and independent upstream full-distance enumeration validate the integration; see [native reference audit]({REL}/native-reference-audit.json).

## Recall, actual serving storage, and batch throughput

Recall/storage use the eight-thread batch-32 rows. B/vector is `max(serialized serving bytes, owned resident allocations) / corpus size`. Stored bits are eight times bytes; totals include PCA, centroids, rotation matrices, compressed codes, factors, stored IDs and adapter/model shells. Transient query/top-k buffers, allocator overhead and language/thread runtimes are excluded. No original FP32 corpus backing is retained by the serving index. These are allocation measures, not RSS or peak build memory.

{table(['Dataset','Codec','Nominal bits/dim','Recall@10 mean ± SD','B/vector','Stored bits/vector','QPS · 1 thread','QPS · 8 threads'],main_rows)}

![Recall and accounted serving storage]({REL}/recall-storage.png)

Lines connect all measured points in code-budget order, including the six existing storage-selected TieredQ points (stars, † in tables). They are visual guides, with no interpolation. [Vector PDF]({REL}/recall-storage.pdf).

## Single-query latency and batch throughput

Latency is complete batch-1 call **p50 / p95 ms**, not timing confidence intervals. Batch-32 throughput counts completed queries divided by total search-call time. Each schedule receives three single-query warmups and three full-query repeats. Batch-32 call percentiles in the CSV are whole-batch response latency, weighted by batch query count, including the final partial batch; they are not single-query latency.

{table(['Dataset','Codec','Nominal bits/dim','Batch-1 p50 / p95 · 1 thread','Batch-1 p50 / p95 · 8 threads'],latency)}

![Single-query latency with separate thread panels]({REL}/latency-recall.png)

Solid codec lines show p50, dashed lines p95. [Vector PDF]({REL}/latency-recall.pdf).

![Batch-32 throughput with separate thread panels]({REL}/qps-recall.png)

Thread panels remain separate; eight-thread throughput does not substitute for single-query response time. [Vector PDF]({REL}/qps-recall.pdf).

## Discrete measured storage caps

Each cap is the actual storage of a main-grid TieredQ or SAQ point. Select the highest measured three-seed mean recall fitting each cap, including the six earlier TieredQ supplements. Entries are requested code bits/vector; serving bytes; mean recall. **†** marks an earlier storage-selected TieredQ point. The SAQ grid has no additional cap-selected points: “None measured” means the measured grid does not fit, not that SAQ cannot fit a different fractional budget. No equal-recall performance or continuously optimal allocation is inferred.

{table(['Dataset','Cap source','Cap · B/vector','TieredQ fast: code bits; bytes; recall','SAQ full accurate: code bits; bytes; recall','SAQ native pruned: code bits; bytes; recall'],caps)}

## Minimum measured storage reaching Recall@10 ≥0.95

The threshold applies to mean recall, not a confidence bound. Minima cover only the measured main grid and the six existing TieredQ supplements; they are not global storage requirements.

{table(['Dataset','Codec','Code bits/vector','Minimum B/vector','Recall@10','QPS · 8 threads, batch 32'],threshold)}

## Construction and encoding

Construction uses the eight-thread configuration with external BLAS pools limited to one. SAQ’s one cluster is encoded by one upstream cluster task, so that construction is effectively serial inside its encoder despite the eight-worker pool. SAQ build seconds include full-corpus PCA fit, base projection, segmentation/rotation preparation, encoding, save and reload. PCA is fitted once per dataset and its measured time is charged to every reported construction. TieredQ’s archived build timer includes its preprocessing/training/add, but excludes persistence and storage inspection, and it trains on at most 32,768 seeded rows. Those different training and persistence boundaries prevent reading the build ratio as a pure encoder comparison.

SAQ encode throughput isolates the upstream cluster-quantization region, including its per-segment base rotation and code packing, excluding PCA fit/projection, segmentation/rotation-model preparation and persistence. TieredQ encode throughput covers native index addition. Native-pruned SAQ reuses the same construction evidence; its timing-only seed-0 rebuild first reproduces all original native IDs and storage.

{table(['Dataset','Codec','Nominal bits/dim','Build · s','Encode · vectors/s'],construction)}

## Provenance and limitations

TieredQ’s measured source is **`575e167a0ad66920119bd0d0836b6c0c92c0a4e5`**. Current checkout `1363199` changes the public API: fast is default, `mode="full"` is explicit, and the boolean API was removed. Those usage changes are distinct from the archived measurement adapter. TieredQ timings and supplemental storage points are reused from 27 October on the same host/CPU affinity; SAQ timings are a later, separate shared-host run. Frequency, contention and cache state can vary between runs. Three repeats and recall SDs do not establish timing confidence intervals. The unrelated 100-query normal-mode control remains outside this comparison.

The [SAQ repository](https://github.com/howarlii/SAQ/tree/2163ebcedd0ad9c9f4de326e6ca7a860f9eafe52) supplies the algorithm, not these new measurements. The [README]({REL}/README.md) records reproduction, dependencies, validation, storage boundaries and retained remote models. [Primary CSV]({REL}/results.csv), [native-pruned CSV]({REL}/native-pruned/results.csv), [metadata]({REL}/meta.json), [raw-evidence audit]({REL}/audit.json) and [independent report verification]({REL}/report-audit.json) are authoritative for this report. TieredQ sources are [main CSV]({OLD}/output/results.csv), [first supplement]({OLD}/matched-storage/results.csv) and [remaining supplements]({OLD}/storage-sweep/results.csv), with their original audits preserved. A failed adapter validation is retained separately and contributes no reported row. No commit was made; the prior RaBitQ/TurboQuant report was not changed.
'''
    (REPO/'docs/saq-vs-tieredq-benchmark.md').write_text(text)
    print('Generated SAQ report')


if __name__=='__main__':main()
