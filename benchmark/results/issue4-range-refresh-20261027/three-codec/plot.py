"""Render all measured points in code-budget order with distinct codec and scheduling styles."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from report_data import all_points,DATASET_NAMES,CODEC_NAMES,label
HERE=Path(__file__).resolve().parent
STYLES={'tieredq':('#1769aa','-'),'rabitq':('#c35400','-.'),'turboquant':('#7a3ba0',':')}
MARKERS={1:'o',2:'s',4:'^'}


def series(ax,points,codec,kind):
    ordered=sorted((p for p in points if p.codec==codec),key=lambda p:p.code_bits)
    color,line=STYLES[codec]
    x=[p.storage*8 if kind=='storage' else p.recall for p in ordered]
    y=[p.recall if kind=='storage' else p.qps if kind=='qps' else p.p50 for p in ordered]
    ax.plot(x,y,color=color,linestyle=line,linewidth=1.5)
    if kind=='latency':ax.plot(x,[p.p95 for p in ordered],color=color,linestyle='--',alpha=.7)
    for p,xx,yy in zip(ordered,x,y,strict=True):
        ax.errorbar(xx,yy,yerr=p.std if kind=='storage' else None,fmt='*' if p.supplemental else MARKERS[p.nominal_bits],
                    color=color,markersize=9 if p.supplemental else 5,capsize=2,markeredgewidth=.5,
                    markeredgecolor='black' if p.supplemental else color,zorder=4)
        if kind=='latency' and p.supplemental:
            ax.plot([xx,xx],[yy,p.p95],color=color,linestyle=':',alpha=.45)
            ax.scatter(xx,p.p95,marker='*',s=65,facecolors='white',edgecolors=color,zorder=3)
        dy=10 if codec=='tieredq' else -14 if codec=='rabitq' else 3
        if p.supplemental:dy=12 if label(p).startswith('M') else -12
        ax.annotate(label(p),(xx,yy),xytext=(5,dy),textcoords='offset points',fontsize=7,color=color,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.7,pad=.4))
    ax.grid(alpha=.2)
    ax.margins(x=.12,y=.2)


def draw(kind,focused=False):
    points=all_points();datasets={k:v for k,v in DATASET_NAMES.items() if not focused or k in ('openai','gemini')}
    threads=(8,) if kind=='storage' else (1,8)
    batch=1 if kind=='latency' else 32
    fig,axes=plt.subplots(len(threads),len(datasets),figsize=(6.3*len(datasets),4.8*len(threads)),squeeze=False)
    fig.subplots_adjust(left=.055,right=.99,bottom=.19 if len(threads)==1 else .11,top=.82 if len(threads)==1 else .89,wspace=.3,hspace=.35)
    for row,t in enumerate(threads):
        for ax,(d,name) in zip(axes[row],datasets.items(),strict=True):
            selected=[p for p in points if p.dataset==d and p.threads==t and p.batch==batch]
            for codec in STYLES:series(ax,selected,codec,kind)
            ax.set_title(name+(f' · {t} thread'+('s' if t>1 else '') if kind!='storage' else ''))
            ax.set_xlabel('Accounted serving bits / base vector' if kind=='storage' else 'Recall@10')
            ax.set_ylabel('Recall@10' if kind=='storage' else 'Queries / second · batch 32' if kind=='qps' else 'Single-query latency (ms) · batch 1')
            if kind!='storage':ax.set_ylim(bottom=0)
    handles=[Line2D([],[],color=c,linestyle=l,label=CODEC_NAMES[k]) for k,(c,l) in STYLES.items()]
    handles.append(Line2D([],[],color=STYLES['tieredq'][0],marker='*',linestyle='',label='TieredQ · storage selected (M: RaBitQ; Q: TQ+)'))
    if kind=='latency':handles.append(Line2D([],[],color='#555',linestyle='--',label='p95 (other lines: p50)'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=2 if focused else len(handles),frameon=False,fontsize=9)
    fig.suptitle('Recall: mean ± population SD over seeds 0/1/2 · timing/storage: seed 0\nLines connect every measured point in code-budget order; no interpolation of results',fontsize=11,y=.98)
    filename='recall-storage-matched' if focused else {'storage':'recall-storage-all','qps':'qps-recall-all','latency':'latency-recall-all'}[kind]
    for ext in ('png','pdf'):fig.savefig(HERE/f'{filename}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    for kind in ('storage','qps','latency'):draw(kind)
    draw('storage',True)
