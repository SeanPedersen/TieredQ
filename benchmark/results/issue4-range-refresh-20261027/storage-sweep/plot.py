"""Plot all measured OpenAI/Gemini points, marking six storage-selected additions."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from report_data import all_points, serving_points, DATASET_NAMES

from plot_style import COLORS, MARKERS
ROOT = Path(__file__).resolve().parent


def panel(ax, dataset, points):
    selected = [p for p in points if p.dataset == dataset]
    for codec, color in COLORS.items():
        ordered = sorted((p for p in selected if p.codec == codec), key=lambda p: p.code_bits)
        ax.plot([p.storage * 8 for p in ordered], [p.recall for p in ordered], color=color, linewidth=1.3)
    for point in selected:
        marker = '*' if point.supplemental else MARKERS[point.nominal_bits]
        ax.errorbar(point.storage * 8, point.recall, yerr=point.std,
                    fmt=marker, color=COLORS[point.codec], markersize=12 if point.supplemental else 6,
                    capsize=3, markeredgecolor='black' if point.supplemental else COLORS[point.codec],
                    markeredgewidth=.5)
        label = f'M{point.anchor_bits}' if point.supplemental else f'{point.nominal_bits:g}b'
        offset = (0, 13) if point.supplemental else (8, -14 if point.codec == 'rabitq' else 7)
        ax.annotate(label, (point.storage * 8, point.recall), xytext=offset,
                    textcoords='offset points', fontsize=9, color=COLORS[point.codec])
    ax.set(title=DATASET_NAMES[dataset], xlabel='Accounted serving storage (bits per base vector)', ylabel='Recall@10')
    ax.margins(x=.08, y=.15)
    ax.grid(alpha=.2)


def main():
    points = serving_points(all_points())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, dataset in zip(axes, ('openai', 'gemini'), strict=True):
        panel(ax, dataset, points)
    handles = [Line2D([], [], marker='o', linestyle='-', color=COLORS['tieredq'], label='TieredQ fast'),
               Line2D([], [], marker='o', linestyle='-', color=COLORS['rabitq'], label='RaBitQ'),
               Line2D([], [], marker='*', markersize=12, linestyle='', color=COLORS['tieredq'], label='TieredQ fast · storage-selected')]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Measured recall–storage tradeoff · 1,000 queries · mean ± population SD over three seeds\n'
                 'M1 / M2 / M4: TieredQ selected below the corresponding RaBitQ storage cap', fontsize=11)
    fig.savefig(ROOT / 'recall-storage-matched.png', dpi=200)
    fig.savefig(ROOT / 'recall-storage-matched.pdf')
    plt.close(fig)
    from plot_all import main as draw_complete_figures
    draw_complete_figures()


if __name__ == '__main__':
    main()
