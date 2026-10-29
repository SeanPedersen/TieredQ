"""Plot all measured budgets per codec, keeping scheduling panels separate."""
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

DATASET_NAMES = {'sift': 'SIFT1M', 'glove': 'GloVe-200', 'openai': 'OpenAI3-1536', 'gemini': 'DBpedia Gemini-768'}
CODEC_STYLES = {'tieredq': ('TieredQ', '#1769aa'), 'rabitq': ('RaBitQ', '#d95f02'), 'qdrant': ('Qdrant TQ+', '#1b9e77')}
# Vertical label offsets (points) keep budget labels of overlapping codecs apart.
LABEL_OFFSETS = {'tieredq': 12, 'rabitq': -17, 'qdrant': 24}
BUDGET_MARKERS = {1: 'o', 2: 's', 3: 'v', 4: '^', 5: 'P', 6: 'X', 7: 'h', 8: 'D'}
FIGURES = (
    ('recall-storage', 'effective_bytes_per_vector', 'recall10', 'Measured bytes / vector', 'Recall@10'),
    ('qps-recall', 'recall10', 'qps', 'Recall@10', 'Queries / second · batch 32'),
    ('latency-recall', 'recall10', 'p50_ms', 'Recall@10', 'Single-query latency (ms) · batch 1'),
)


def budget_points(rows, dataset, codec, threads, batch):
    groups = defaultdict(list)
    for row in rows:
        if (row['dataset'], row['codec'], int(row['threads']), int(row['batch'])) != (dataset, codec, threads, batch):
            continue
        groups[int(row['target_bits'])].append(row)
    points = []
    for bits, group in sorted(groups.items()):
        timed = next((r for r in group if int(r['seed']) == 0), None)
        if timed is None:
            continue
        point = {key: float(timed[key]) for key in ('effective_bytes_per_vector', 'qps', 'p50_ms', 'p95_ms')}
        point.update(bits=bits, recall10=float(np.mean([float(r['recall10']) for r in group])))
        points.append(point)
    return points


def draw_series(ax, points, codec, xkey, ykey, latency):
    label, color = CODEC_STYLES[codec]
    ax.plot([p[xkey] for p in points], [p[ykey] for p in points], color=color, linewidth=1.8, label=label)
    if latency:
        ax.plot([p[xkey] for p in points], [p['p95_ms'] for p in points], color=color, linestyle='--', alpha=.7)
    for point in points:
        ax.scatter(point[xkey], point[ykey], marker=BUDGET_MARKERS[point['bits']], color=color, s=44, zorder=3)
        ax.annotate(f"{point['bits']}b", (point[xkey], point[ykey]), xytext=(0, LABEL_OFFSETS[codec]),
                    textcoords='offset points', ha='center', color=color, fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.75, pad=.3))


def plot_results(root, rows):
    datasets = [name for name in DATASET_NAMES if any(r['dataset'] == name for r in rows)]
    seeds = sorted({int(r['seed']) for r in rows})
    seed_note = f'Fixed seed {seeds[0]}; one run' if len(seeds) == 1 else f'Recall averaged over {len(seeds)} seeds; timing seed 0'
    batches = {int(row['batch']) for row in rows}
    budgets = ' → '.join(str(bits) for bits in sorted({int(row['target_bits']) for row in rows}))
    for name, xkey, ykey, xlabel, ylabel in FIGURES:
        batch = 32 if name == 'qps-recall' else 1 if name == 'latency-recall' else min(batches)
        if batch not in batches:
            continue
        thread_counts = sorted({int(row['threads']) for row in rows if int(row['batch']) == batch})
        if name == 'recall-storage':
            thread_counts = thread_counts[:1]
        fig, axes = plt.subplots(len(thread_counts), len(datasets), figsize=(6 * len(datasets), 4.4 * len(thread_counts)), squeeze=False)
        for row_index, threads in enumerate(thread_counts):
            for dataset, ax in zip(datasets, axes[row_index], strict=True):
                for codec in CODEC_STYLES:
                    points = budget_points(rows, dataset, codec, threads, batch)
                    if points:
                        draw_series(ax, points, codec, xkey, ykey, name == 'latency-recall')
                suffix = '' if name == 'recall-storage' else f' · {threads} thread' + ('s' if threads > 1 else '')
                ax.set(title=DATASET_NAMES[dataset] + suffix, xlabel=xlabel, ylabel=ylabel)
                ax.margins(x=.10, y=.20)
                ax.grid(alpha=.2)
                present = [c for c in CODEC_STYLES if any(r['codec'] == c for r in rows)]
                handles = [Line2D([], [], color=CODEC_STYLES[c][1], label=CODEC_STYLES[c][0]) for c in present]
                if name == 'latency-recall':
                    handles += [Line2D([], [], color='#555', label='p50'), Line2D([], [], color='#555', linestyle='--', label='p95')]
                ax.legend(handles=handles, fontsize=9)
        fig.suptitle(f'{seed_note} · lines connect {budgets} bits per original dimension', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .95 if len(thread_counts) == 1 else .97))
        fig.savefig(root / f'{name}.png', dpi=160)
        plt.close(fig)


MEMORY_METRICS = (('disk_bits_per_vector', 'Disk bits / vector'),
                  ('ram_loaded_bits_per_vector', 'Steady RAM bits / vector'),
                  ('ram_peak_bits_per_vector', 'Peak RAM bits / vector'))


def plot_memory(root, rows):
    """Recall@10 against disk, steady-RAM and peak-RAM bits/vector; one point per codec and budget.

    Each row is one (dataset, codec, target_bits) summary: memory is the maximum over seeds, recall the mean.
    """
    datasets = [name for name in DATASET_NAMES if any(r['dataset'] == name for r in rows)]
    present = [c for c in CODEC_STYLES if any(r['codec'] == c for r in rows)]
    fig, axes = plt.subplots(len(MEMORY_METRICS), len(datasets), figsize=(5.2 * len(datasets), 3.9 * len(MEMORY_METRICS)), squeeze=False)
    for metric_axes, (key, xlabel) in zip(axes, MEMORY_METRICS, strict=True):
        for dataset, ax in zip(datasets, metric_axes, strict=True):
            for codec in present:
                points = sorted((dict(bits=int(r['target_bits']), x=float(r[key]), y=float(r['recall10']))
                                 for r in rows if r['dataset'] == dataset and r['codec'] == codec), key=lambda p: p['bits'])
                draw_series(ax, points, codec, 'x', 'y', False)
            ax.set(title=DATASET_NAMES[dataset], xlabel=xlabel, ylabel='Recall@10')
            ax.margins(x=.10, y=.20)
            ax.grid(alpha=.2)
            ax.legend(handles=[Line2D([], [], color=CODEC_STYLES[c][1], label=CODEC_STYLES[c][0]) for c in present], fontsize=9)
    fig.suptitle('Measured memory per vector vs Recall@10 · lines connect 1 → 2 → 4 bits per original dimension', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(root / 'recall-memory.png', dpi=160)
    plt.close(fig)


MATCHED_METRICS = (('disk', 'Disk bits / vector'), ('ram', 'Steady RAM bits / vector'), ('peak', 'Peak RAM bits / vector'))


def plot_matched(root, rows, reference):
    """Recall@10 of TieredQ and one reference codec at each equal-measured-memory target (labels = reference bits/dim).

    Rows come from finalize.matched_rows: the reference sits at its target, TieredQ at its max memory over seeds.
    """
    rows = [r for r in rows if r['reference_codec'] == reference]
    datasets = [name for name in DATASET_NAMES if any(r['dataset'] == name for r in rows)]
    fig, axes = plt.subplots(len(MATCHED_METRICS), len(datasets), figsize=(5.2 * len(datasets), 3.9 * len(MATCHED_METRICS)), squeeze=False)
    for metric_axes, (metric, xlabel) in zip(axes, MATCHED_METRICS, strict=True):
        for dataset, ax in zip(datasets, metric_axes, strict=True):
            targets = sorted((r for r in rows if (r['dataset'], r['metric']) == (dataset, metric)), key=lambda r: r['reference_bits'])
            draw_series(ax, [dict(bits=r['reference_bits'], x=r['target_bits_per_vector'], y=r['reference_recall10']) for r in targets],
                        reference, 'x', 'y', False)
            draw_series(ax, [dict(bits=r['reference_bits'], x=r['tieredq_memory_max'], y=r['tieredq_recall10'])
                             for r in targets if 'tieredq_recall10' in r], 'tieredq', 'x', 'y', False)
            ax.set(title=DATASET_NAMES[dataset], xlabel=xlabel, ylabel='Recall@10')
            ax.margins(x=.10, y=.20)
            ax.grid(alpha=.2)
            ax.legend(handles=[Line2D([], [], color=CODEC_STYLES[c][1], label=CODEC_STYLES[c][0]) for c in ('tieredq', reference)], fontsize=9)
    fig.suptitle(f'TieredQ at {CODEC_STYLES[reference][0]}\'s measured memory · labels = {CODEC_STYLES[reference][0]} bits per original dimension',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(root / f'recall-memory-{reference}.png', dpi=160)
    plt.close(fig)
