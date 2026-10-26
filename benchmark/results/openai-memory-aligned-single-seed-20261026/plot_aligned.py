"""Show measured OpenAI storage points and explicit within-memory anchor pairs."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def main():
    core=list(csv.DictReader((ROOT/'core-results.csv').open()))
    extra=list(csv.DictReader((ROOT/'results.csv').open()))
    rows=[r for r in core if r['dataset']=='openai' and r['threads']=='8' and r['batch']=='32']
    fresh=[r for r in extra if r['threads']=='8' and r['batch']=='32']
    fig,axes=plt.subplots(1,2,figsize=(15,5.6))
    for ax in axes:
        for codec,color,label in [('tieredq','#1769aa','TieredQ original grid'),('rabitq','#d95f02','RaBitQ native anchors')]:
            group=sorted([r for r in rows if r['codec']==codec],key=lambda r:float(r['target_bits']))
            ax.plot([float(r['effective_bytes_per_vector']) for r in group],[float(r['recall10']) for r in group],marker='o',color=color,label=label,alpha=.65)
            for r in group:
                ax.annotate(r['target_bits']+'b',(float(r['effective_bytes_per_vector']),float(r['recall10'])),xytext=(0,9 if codec=='tieredq' else -16),textcoords='offset points',ha='center',fontsize=9,color=color)
        ax.scatter([float(r['effective_bytes_per_vector']) for r in fresh],[float(r['recall10']) for r in fresh],marker='D',s=65,color='#15945b',label='TieredQ new measured points',zorder=5)
        for r in fresh:
            x=float(r['effective_bytes_per_vector']);y=float(r['recall10']);limit=float(r['rabitq_anchor_bytes'])
            rb=next(p for p in rows if p['codec']=='rabitq' and p['target_bits']==r['rabitq_anchor_bits'])
            ax.axvline(limit,color='#888',linestyle=':',alpha=.4)
            ax.plot([x,limit],[y,float(rb['recall10'])],color='#555',linestyle='--',linewidth=1)
            ax.annotate(f'{x:.1f} B; {float(r["target_bits"]):.3f}b',(x,y),xytext=(8,-20),textcoords='offset points',fontsize=9,color='#087840')
        ax.set(xlabel='Measured serving bytes / vector',ylabel='Recall@10');ax.grid(alpha=.2)
    axes[0].set_title('OpenAI3-1536 · full measured curve');axes[0].legend(loc='lower right',fontsize=9)
    axes[1].set(xlim=(700,1750),ylim=(.955,1.005),title='High-recall detail · new points sit below memory anchors')
    fig.suptitle('Seed 0 only · no statistical-confidence claim\nDashed pairs are measured within-memory comparisons; curves imply no interpolated results',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(ROOT/'recall-storage-aligned.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
