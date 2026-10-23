"""Plot all four measured budgets per codec, keeping scheduling panels separate."""
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

DATASET_NAMES = {'sift': 'SIFT1M', 'glove': 'GloVe-200', 'openai': 'OpenAI3-1536'}
CODEC_STYLES = {'tieredq': ('TieredQ', '#1769aa'), 'rabitq': ('RaBitQ', '#d95f02')}
BUDGET_MARKERS = {1: 'o', 2: 's', 4: '^', 8: 'D'}
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
        offset = 12 if codec == 'tieredq' else -17
        ax.annotate(f"{point['bits']}b", (point[xkey], point[ykey]), xytext=(0, offset),
                    textcoords='offset points', ha='center', color=color, fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.75, pad=.3))


def plot_results(root, rows):
    datasets = [name for name in DATASET_NAMES if any(r['dataset'] == name for r in rows)]
    seeds = sorted({int(r['seed']) for r in rows})
    seed_note = f'Fixed seed {seeds[0]}; one run' if len(seeds) == 1 else f'Recall averaged over {len(seeds)} seeds; timing seed 0'
    for name, xkey, ykey, xlabel, ylabel in FIGURES:
        thread_counts = [1] if name == 'recall-storage' else [1, 8]
        fig, axes = plt.subplots(len(thread_counts), len(datasets), figsize=(6 * len(datasets), 4.4 * len(thread_counts)), squeeze=False)
        for row_index, threads in enumerate(thread_counts):
            for dataset, ax in zip(datasets, axes[row_index], strict=True):
                for codec in CODEC_STYLES:
                    points = budget_points(rows, dataset, codec, threads, 32 if name == 'qps-recall' else 1)
                    if points:
                        draw_series(ax, points, codec, xkey, ykey, name == 'latency-recall')
                suffix = '' if name == 'recall-storage' else f' · {threads} thread' + ('s' if threads > 1 else '')
                ax.set(title=DATASET_NAMES[dataset] + suffix, xlabel=xlabel, ylabel=ylabel)
                ax.margins(x=.10, y=.20)
                ax.grid(alpha=.2)
                handles = [Line2D([], [], color=color, label=label) for label, color in CODEC_STYLES.values()]
                if name == 'latency-recall':
                    handles += [Line2D([], [], color='#555', label='p50'), Line2D([], [], color='#555', linestyle='--', label='p95')]
                ax.legend(handles=handles, fontsize=9)
        fig.suptitle(f'{seed_note} · lines connect 1 → 2 → 4 → 8 bits per original dimension', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .95 if len(thread_counts) == 1 else .97))
        fig.savefig(root / f'{name}.png', dpi=160)
        plt.close(fig)
