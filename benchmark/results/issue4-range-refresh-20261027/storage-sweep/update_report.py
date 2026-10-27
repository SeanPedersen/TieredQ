"""Update the benchmark report from all measured storage-cap points, retaining main-grid tables."""
from pathlib import Path
import re

from report_data import (ROOT, DATASET_NAMES, CODEC_NAMES, MAIN_BUDGETS, all_points,
                         serving_points, anchor_point, schedule)

REPORT = ROOT.parents[2] / 'docs/tieredq-vs-rabitq-benchmark.md'
REL = '../benchmark/results/issue4-range-refresh-20261027'


def table(headers, rows):
    return ('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join('---' for _ in headers) + ' |\n' +
            '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows))


def replace_section(text, title, next_title, body):
    start = text.index('## ' + title + '\n')
    stop = text.index('## ' + next_title + '\n', start)
    return text[:start] + '## ' + title + '\n\n' + body.strip() + '\n\n' + text[stop:]


def supplements(points):
    return sorted((p for p in points if p.supplemental), key=lambda p: (p.dataset == 'gemini', p.anchor_bits))


def findings(points):
    added = supplements(points)
    wins = [p for p in added if p.recall > anchor_point(points, p.dataset, p.anchor_bits).recall]
    losses = [p for p in added if p not in wins]
    win_labels = ', '.join(f'{DATASET_NAMES[p.dataset]} M{p.anchor_bits}' for p in wins)
    loss_labels = ', '.join(f'{DATASET_NAMES[p.dataset]} M{p.anchor_bits}' for p in losses)
    outcome = f'TieredQ has higher mean recall at {len(wins)} of the six measured near-matched-storage comparisons: {win_labels}.'
    if losses:
        outcome += f' RaBitQ has higher mean recall at {loss_labels}.'
    if len(wins) == len(added):
        gains = [100 * (p.recall - anchor_point(points, p.dataset, p.anchor_bits).recall) for p in added]
        savings = [100 * (1 - p.storage / anchor_point(points, p.dataset, p.anchor_bits).storage) for p in added]
        outcome = (f'At all six tested OpenAI/Gemini storage caps, TieredQ achieves higher mean recall '
                   f'with {min(savings):.2f}–{max(savings):.2f}% less accounted serving storage. '
                   f'Recall gains range from {min(gains):.2f} to {max(gains):.2f} percentage points.')
    low_speed = [p.qps / anchor_point(points, p.dataset, p.anchor_bits).qps for p in added if p.anchor_bits == 1]
    higher_speed = [anchor_point(points, p.dataset, p.anchor_bits).qps / p.qps for p in added if p.anchor_bits != 1]
    original = next(p for p in added if p.dataset == 'openai' and p.anchor_bits == 2)
    baseline = anchor_point(points, 'openai', 2)
    return f'''TieredQ’s strongest main-grid result is SIFT1M: at every measured nominal budget it delivers higher recall with less serving storage and higher batch throughput. GloVe also shows useful storage savings. The added OpenAI and Gemini measurements compare actual accounted storage directly, resolving gaps left by the nominal-budget grid. {outcome} Search throughput remains a separate tradeoff.

- **SIFT:** main-grid recall gains are 8.88–24.81 percentage points. Serving storage is 1.5–20.5% lower; eight-thread batch-32 throughput is 1.09–1.17× RaBitQ’s. The smaller timing gains need caution because the runs were on different days.
- **GloVe:** TieredQ’s 2- and 4-bit main points save 19.1% and 20.9% storage respectively, with slightly higher mean recall. Its 4-bit batch throughput is lower; its 1-bit mean recall is slightly below RaBitQ’s.
- **Actual stored-bit comparison:** the OpenAI point just below RaBitQ’s 2-bit storage cap reaches **{original.recall:.4f} ± {original.std:.4f}** recall at **{original.storage * 8:,.1f} stored bits/vector**, versus RaBitQ’s **{baseline.recall:.4f} ± {baseline.std:.4f}** at **{baseline.storage * 8:,.1f}**. All six storage-cap comparisons are reported below.
- **Serving speed:** at the low storage caps, TieredQ’s observed eight-thread batch-32 QPS is {min(low_speed):.2f}–{max(low_speed):.2f}× RaBitQ’s. At the middle and high caps, RaBitQ is {min(higher_speed):.2f}–{max(higher_speed):.2f}× faster, with lower recall. These separately timed runs do not establish equal-recall throughput superiority.
- **Construction:** TieredQ takes longer to build at every main-grid point. Encoding alone has a different tradeoff: RaBitQ leads at 1 bit; TieredQ leads at 2 and 4 bits.'''


def choice(point):
    marker = '†' if point.supplemental else ''
    return f'{point.code_bits:,}{marker}; {point.storage:.3f} B; {point.recall:.4f}'


def cap_section(points):
    rows = []
    for dataset, name in DATASET_NAMES.items():
        for budget in MAIN_BUDGETS:
            cap = anchor_point(points, dataset, budget).storage
            cells = []
            for codec in CODEC_NAMES:
                eligible = [p for p in points if p.dataset == dataset and p.codec == codec and p.storage <= cap]
                cells.append(choice(max(eligible, key=lambda p: p.recall)) if eligible else 'None')
            rows.append([name, f'{cap:.3f}', *cells])
    caps = table(['Dataset', 'Cap · B/vector', 'TieredQ: code bits; bytes; recall', 'RaBitQ: code bits; bytes; recall'], rows)
    minima = []
    conclusions = []
    for dataset, name in DATASET_NAMES.items():
        chosen = {}
        for codec, label in CODEC_NAMES.items():
            eligible = [p for p in points if p.dataset == dataset and p.codec == codec and p.recall >= .95]
            if not eligible:
                minima.append([name, label, 'Not reached', '—', '—', '—'])
                continue
            point = min(eligible, key=lambda p: p.storage)
            chosen[codec] = point
            budget_label = f'{point.code_bits:,}' + ('†' if point.supplemental else '')
            minima.append([name, label, budget_label, f'{point.storage:.3f}', f'{point.recall:.4f}', f'{point.qps:,.0f}'])
        if len(chosen) == 2:
            t, r = chosen['tieredq'], chosen['rabitq']
            winner = 'TieredQ' if t.storage < r.storage else 'RaBitQ'
            conclusions.append(f'{name}: {winner} has the smaller measured point reaching the threshold '
                               f'(TieredQ {t.storage:.3f} versus RaBitQ {r.storage:.3f} B/vector).')
    minimum_table = table(['Dataset', 'Codec', 'Code bits/vector', 'Minimum B/vector', 'Recall@10', 'QPS · 8 threads, batch 32'], minima)
    return f'''Each storage cap is the actual size of a measured RaBitQ point. For each codec, select the **highest mean recall among all measured points fitting that cap**, including the six supplemental TieredQ points. Storage and recall use the eight-thread, batch-32 schedule. Entries give **nominal code-bit budget/vector**, actual serving bytes/vector, and mean recall. A **†** marks a supplemental point. Nominal code budgets exclude representation padding and serving overhead; actual stored bits are eight times the bytes shown. “None” means no measured point fits. No unmeasured operating point is estimated.

{caps}

These comparisons maximize recall subject to each measured cap; they do not require equal recall or identical byte counts. In particular, “below the cap” does not by itself demonstrate better compression: the measured recall must also be considered. The supplemental selection protocol and paired results are given below.

### Minimum measured storage reaching Recall@10 ≥0.95

The threshold applies to the three-seed **mean** recall, not a confidence bound. Selection now includes the original main grid **and all six supplemental points**. These are minima among measured configurations, not the codec’s globally minimum required storage.

{minimum_table}

Neither codec reaches the threshold on SIFT or GloVe within the measured budgets. {' '.join(conclusions)} These results replace the earlier main-grid-only threshold comparison. The selected points have different recall above 0.95 and do not establish matched-recall serving performance.'''


def supplemental_section(points, full_points):
    added = supplements(points)
    paired = []
    summary = []
    timings = []
    for p in added:
        baseline = anchor_point(points, p.dataset, p.anchor_bits)
        label = DATASET_NAMES[p.dataset] + f' M{p.anchor_bits}'
        for codec, point in [('TieredQ fast', p), ('RaBitQ', baseline)]:
            single_thread = schedule(full_points, point, 1, 32)
            paired.append([label, codec, f'{point.storage:.3f}', f'{point.storage * 8:,.1f}',
                           f'{point.recall:.4f} ± {point.std:.4f}', f'{single_thread.qps:,.0f}', f'{point.qps:,.0f}'])
        summary.append([label, f'{p.code_bits:,}', f'{100 * (1 - p.storage / baseline.storage):.2f}%',
                        f'{100 * (p.recall - baseline.recall):+.2f}', f'{p.qps / baseline.qps:.2f}×'])
        one = schedule(full_points, p, 1, 1)
        eight = schedule(full_points, p, 8, 1)
        timings.append([label, f'{one.p50:.3f} / {one.p95:.3f}', f'{eight.p50:.3f} / {eight.p95:.3f}',
                        f'{p.build_s:.3f}', f'{p.encode_vps:,.0f}'])
    paired_table = table(['Dataset / cap', 'Codec', 'B/vector', 'Stored bits/vector', 'Recall@10 mean ± SD', 'Batch-32 QPS · 1 thread', 'Batch-32 QPS · 8 threads'], paired)
    summary_table = table(['Dataset / cap', 'TieredQ code bits/vector', 'Storage saving', 'Recall gain · percentage points', 'TieredQ / RaBitQ QPS · 8 threads, batch 32'], summary)
    timing_table = table(['TieredQ point', 'Batch-1 p50 / p95 · ms, 1 thread', 'Batch-1 p50 / p95 · ms, 8 threads', 'Build · s', 'Encode · vectors/s'], timings)
    band_count = sum(.99 <= p.storage / anchor_point(points, p.dataset, p.anchor_bits).storage <= .995 for p in added)
    return f'''Six TieredQ fast points fill the missing actual-storage comparisons on OpenAI and Gemini. **M1, M2, and M4** mean that TieredQ was selected below the serving storage of the corresponding RaBitQ 1-, 2-, or 4-bit point. They are not TieredQ nominal budgets.

All supplemental measurements use the original measured TieredQ source, verified byte-identical inputs, the isolated exact-code-budget adapter, the audited compiler policy, CPUs 0–7, 1,000 queries, and recall seeds 0/1/2. Timing and serving storage are seed 0, with threads 1/8 and batches 1/32 and the original warmups/repeats. Before the five-point extension, one-bit controls on **both datasets** reproduced the archived seed-0 batch-32 top-10 IDs and serving storage exactly. Native quantizer and search kernels were unchanged.

**Selection uses storage alone, before candidate recall is measured.** The target is 99.0–99.5% of the baseline cap. Integer code budgets are selected using a storage-based initial estimate followed by bounded bisection. Fast-layout changes can make storage discrete; if the target band is not found, the protocol retains the largest observed serving size below the cap, without optimizing recall. Every attempt and any fallback are recorded. **{band_count} of six** selected points fall in the target band. The original OpenAI M2 selection and the five-point extension have separate logs and metadata; the main audit and original figures are unchanged.

{paired_table}

{summary_table}

![Measured OpenAI and Gemini recall against stored bits, including all six storage-selected points]({REL}/storage-sweep/recall-storage-matched.png)

*Figure 4. Discrete measured points; stars mark the supplemental configurations. M1/M2/M4 refer to RaBitQ storage caps. Error bars are population standard deviations over recall seeds. Storage is seed 0 from the eight-thread, batch-32 row. Lines connect all measured points in code-budget order as visual guides; no unmeasured results are inferred. A [vector PDF]({REL}/storage-sweep/recall-storage-matched.pdf) is available.*

All six comparisons show higher measured mean recall at slightly lower actual serving storage. This tests the six explicit caps; it does not establish dominance between them or at equal recall. The QPS ratios retain the separate shared-host timing runs, so small differences are not stable speed claims.

The following table reports each supplemental TieredQ point’s complete single-query latency and construction cost. RaBitQ’s matching nominal-budget latency and construction measurements remain in the main tables above.

{timing_table}

Gemini M2 records an eight-thread single-query p95 of **3.192 ms** alongside a p50 of **0.406 ms**. This tail result is retained as observed; its cause was not isolated.

The first OpenAI M2 point is retained in [matched-storage/results.csv]({REL}/matched-storage/results.csv), with its [metadata]({REL}/matched-storage/meta.json), [selection]({REL}/matched-storage/selection.json), and [reproduction instructions]({REL}/matched-storage/README.md). The remaining five points are in [storage-sweep/results.csv]({REL}/storage-sweep/results.csv), with [metadata]({REL}/storage-sweep/meta.json), [control parity]({REL}/storage-sweep/controls.json), [audit]({REL}/storage-sweep/audit.json), [per-point storage selections and reproduction instructions]({REL}/storage-sweep/README.md), and [job log]({REL}/storage-sweep/job.log). The extension contains 60 rows, including 20 timed seed-0 rows. Its audit is separate from the original 288-row audit. RaBitQ timings are still reused from 26 October; all new TieredQ runs are dated in their metadata.'''


def main():
    full_points = all_points()
    points = serving_points(full_points)
    assert len(supplements(points)) == 6
    text = REPORT.read_text()
    text = replace_section(text, 'Findings', 'Experimental design', findings(points))
    text = replace_section(text, 'Comparisons under measured storage caps', 'Construction and encoding', cap_section(points))
    old_title = 'Supplemental OpenAI point just below RaBitQ’s stored-bit budget'
    new_title = 'Supplemental comparisons at actual RaBitQ storage caps'
    if old_title in text:
        text = text.replace('## ' + old_title, '## ' + new_title)
    text = replace_section(text, new_title, 'Scope, provenance, and reproduction', supplemental_section(points, full_points))
    text = text.replace('All **48 paired ID sets matched exactly**', 'For the original main grid, all **48 paired ID sets matched exactly**')
    if 'The supplemental points were not included in this ablation.' not in text:
        text = text.replace('The ablation supports parity on these tested outputs;', 'The supplemental points were not included in this ablation. The ablation supports parity on these tested outputs;')
    text = text.replace('three main budgets plus one supplemental OpenAI budget', 'three main budgets plus six supplemental OpenAI/Gemini points')
    for kind in ('recall-storage', 'qps-recall', 'latency-recall'):
        text = text.replace(f'{REL}/output/{kind}.png', f'{REL}/storage-sweep/{kind}-all.png')
    captions = {
        1: 'All main-grid and supplemental points, using eight-thread, batch-32 serving storage and recall. Storage is shown in actual accounted bits/vector. Stars M1/M2/M4 mark TieredQ points selected below the corresponding RaBitQ storage caps. Lines connect all measured points in code-budget order as visual guides; no unmeasured results are inferred.',
        2: 'Batch-1 latency for all measured points, with thread settings in separate rows. Solid lines show p50 and dashed lines p95, connecting all measured points in code-budget order. Supplemental filled stars show p50 and outlined stars p95; dotted vertical segments connect those two observed percentiles.',
        3: 'Batch-32 throughput for all measured points, with thread settings in separate rows. Stars mark the six storage-selected TieredQ configurations. Lines connect all measured points in code-budget order as visual guides. Timing comparisons retain the separate shared-host-run limitation.'}
    figure_kinds = {1: 'recall-storage', 2: 'latency-recall', 3: 'qps-recall'}
    for number, caption in captions.items():
        caption += f' [Vector PDF]({REL}/storage-sweep/{figure_kinds[number]}-all.pdf).'
        text = re.sub(rf'\*Figure {number}\..*?\*', f'*Figure {number}. {caption}*', text, count=1, flags=re.DOTALL)
    text = text.replace('the main audit and original figures are unchanged', 'the main audit and original plot artifacts are unchanged; the report figures have been regenerated to include all measured points')
    text = text.replace('matched-storage/verify_report.py', 'storage-sweep/verify_report.py')
    text = text.replace('means use population standard deviation', 'recall deviations use the population definition')
    REPORT.write_text(text)
    print(f'Updated {REPORT}')


if __name__ == '__main__':
    main()
