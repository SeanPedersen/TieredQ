"""Plot every measured point in budget order, separating query schedules and SAQ paths."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from report_data import HERE,NAMES,CODECS,all_points
STYLES={'tieredq':('#1769aa','-','o'),'saq':('#7a3ba0',':','s'),'saq-native':('#c35400','-.','^')}


def draw(kind):
    points=all_points()
    threads=(8,) if kind=='recall-storage' else (1,8)
    batch=1 if kind=='latency-recall' else 32
    fig,axes=plt.subplots(len(threads),len(NAMES),figsize=(6.2*len(NAMES),4.6*len(threads)),squeeze=False)
    fig.subplots_adjust(left=.05,right=.99,bottom=.19 if len(threads)==1 else .12,top=.82 if len(threads)==1 else .9,wspace=.3,hspace=.36)
    for row,t in enumerate(threads):
        for ax,(d,name) in zip(axes[row],NAMES.items(),strict=True):
            for c,(color,line,marker) in STYLES.items():
                selected=sorted((p for p in points if p['dataset']==d and p['codec']==c and p['threads']==t and p['batch']==batch),key=lambda p:p['code_bits'])
                x=[p['storage']*8 if kind=='recall-storage' else p['recall'] for p in selected]
                y=[p['recall'] if kind=='recall-storage' else p['p50'] if kind=='latency-recall' else p['qps'] for p in selected]
                ax.plot(x,y,color=color,linestyle=line,linewidth=1.5)
                if kind=='latency-recall':ax.plot(x,[p['p95'] for p in selected],color=color,linestyle='--',alpha=.65)
                for p,xx,yy in zip(selected,x,y,strict=True):
                    ax.errorbar(xx,yy,yerr=p['std'] if kind=='recall-storage' else None,fmt='*' if p['supplemental'] else marker,color=color,
                                markersize=10 if p['supplemental'] else 5,capsize=2,markeredgewidth=.5,zorder=4)
                    label=f"{p['bits']:g}b" if not p['supplemental'] else f"{p['code_bits']}†"
                    ax.annotate(label,(xx,yy),xytext=(5,8 if c=='tieredq' else -13 if c=='saq' else 3),textcoords='offset points',fontsize=7,color=color,
                                bbox=dict(facecolor='white',edgecolor='none',alpha=.7,pad=.3))
            ax.set_title(name+(f' · {t} thread'+('s' if t>1 else '') if kind!='recall-storage' else ''))
            ax.set_xlabel('Accounted serving bits / base vector' if kind=='recall-storage' else 'Recall@10')
            ax.set_ylabel('Recall@10' if kind=='recall-storage' else 'Single-query latency (ms) · batch 1' if kind=='latency-recall' else 'Queries / second · batch 32')
            ax.grid(alpha=.2);ax.margins(x=.12,y=.18)
            if kind!='recall-storage':ax.set_ylim(bottom=0)
    handles=[Line2D([],[],color=color,linestyle=line,marker=marker,label=CODECS[c]) for c,(color,line,marker) in STYLES.items()]
    handles.append(Line2D([],[],color=STYLES['tieredq'][0],marker='*',linestyle='',label='TieredQ · earlier storage-selected point'))
    if kind=='latency-recall':handles.append(Line2D([],[],color='#555',linestyle='--',label='p95; other codec lines show p50'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.01),ncol=len(handles),frameon=False,fontsize=9)
    fig.suptitle('SAQ full accurate / native pruned versus TieredQ fast · same corpus and truth\nRecall: three-seed mean ± population SD · timing/storage: seed 0 · all points connected in code-budget order',fontsize=11,y=.98)
    for extension in ('png','pdf'):fig.savefig(HERE/f'{kind}.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    for kind in ('recall-storage','qps-recall','latency-recall'):draw(kind)
