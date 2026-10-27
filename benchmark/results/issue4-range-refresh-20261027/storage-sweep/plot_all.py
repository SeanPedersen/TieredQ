"""Draw complete recall/storage, throughput, and latency grids with supplemental points."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from report_data import all_points, DATASET_NAMES
from plot_style import COLORS, MARKERS

OUTPUT = Path(__file__).resolve().parent
FIGURE_WIDTH_PER_DATASET = 5.8
FIGURE_HEIGHT_PER_ROW = 4.5
DPI = 200


def coordinates(point, kind):
    if kind == 'recall-storage':
        return point.storage * 8, point.recall
    return point.recall, point.qps if kind == 'qps-recall' else point.p50


def series(ax, points, codec, kind):
    ordered = sorted((p for p in points if p.codec == codec), key=lambda p: p.code_bits)
    color = COLORS[codec]
    ax.plot(*zip(*(coordinates(p, kind) for p in ordered), strict=True), color=color, linewidth=1.3)
    if kind == 'latency-recall':
        ax.plot([p.recall for p in ordered], [p.p95 for p in ordered], color=color, linestyle='--', alpha=.7)
    for point in ordered:
        x, y = coordinates(point, kind)
        ax.scatter(x, y, color=color, marker='*' if point.supplemental else MARKERS[point.nominal_bits],
                   s=95 if point.supplemental else 38, edgecolors='black' if point.supplemental else color,
                   linewidths=.4, zorder=4)
        if point.supplemental and kind == 'latency-recall':
            ax.plot([x, x], [y, point.p95], color=color, linestyle=':', alpha=.6)
            ax.scatter(x, point.p95, marker='*', s=75, facecolors='white', edgecolors=color, linewidths=.9, zorder=3)
        label = f'M{point.anchor_bits}' if point.supplemental else f'{point.nominal_bits:g}b'
        offset = (-10, 11) if point.supplemental else (8, 7 if codec == 'tieredq' else -15)
        ax.annotate(label, (x, y), xytext=offset, textcoords='offset points', fontsize=8, color=color,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.65, pad=.4))


def legend(latency):
    handles = [Line2D([], [], color=COLORS['tieredq'], label='TieredQ fast'),
               Line2D([], [], color=COLORS['rabitq'], label='RaBitQ'),
               Line2D([], [], color=COLORS['tieredq'], marker='*', markersize=10, linestyle='', label='TieredQ fast · storage-selected')]
    if latency:
        handles += [Line2D([], [], color='#555555', label='p50 · filled points'),
                    Line2D([], [], color='#555555', linestyle='--', label='p95 · dashed lines / outlined stars')]
    return handles


def draw(kind, points):
    storage = kind == 'recall-storage'
    latency = kind == 'latency-recall'
    threads = (8,) if storage else (1, 8)
    batch = 1 if latency else 32
    fig, axes = plt.subplots(len(threads), len(DATASET_NAMES),
                             figsize=(FIGURE_WIDTH_PER_DATASET * len(DATASET_NAMES), FIGURE_HEIGHT_PER_ROW * len(threads)),
                             squeeze=False)
    fig.subplots_adjust(left=.045, right=.99, bottom=.18 if storage else .11,
                        top=.79 if storage else .88, wspace=.25, hspace=.34)
    for row, thread_count in enumerate(threads):
        for (dataset, name), ax in zip(DATASET_NAMES.items(), axes[row], strict=True):
            selected = [p for p in points if p.dataset == dataset and p.threads == thread_count and p.batch == batch]
            for codec in COLORS:
                series(ax, selected, codec, kind)
            xlabel = 'Accounted serving storage (bits per base vector)' if storage else 'Recall@10'
            ylabel = 'Recall@10' if storage else 'Queries / second · batch 32' if not latency else 'Single-query call latency (ms) · batch 1'
            suffix = '' if storage else f' · {thread_count} thread' + ('s' if thread_count > 1 else '')
            ax.set(title=name + suffix, xlabel=xlabel, ylabel=ylabel)
            ax.margins(x=.13, y=.22)
            ax.grid(alpha=.2)
            if not storage:
                ax.set_ylim(bottom=0)
    title = 'Recall: mean over three seeds · timing/storage: seed 0 · M1/M2/M4: TieredQ below corresponding RaBitQ storage cap'
    if storage:
        title += '\nStorage uses 8-thread, batch-32 rows; lines connect all measured points in code-budget order'
    else:
        title += '\nThread settings shown separately; lines connect all measured points in code-budget order'
    fig.suptitle(title, fontsize=11, y=.98)
    handles = legend(latency)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .01), ncol=len(handles), frameon=False)
    fig.savefig(OUTPUT / f'{kind}-all.png', dpi=DPI, bbox_inches='tight')
    fig.savefig(OUTPUT / f'{kind}-all.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    points = all_points()
    for kind in ('recall-storage', 'qps-recall', 'latency-recall'):
        draw(kind, points)
        print(f'Updated complete {kind} figure', flush=True)


if __name__ == '__main__':
    main()
