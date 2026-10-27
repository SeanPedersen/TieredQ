"""Plot measured OpenAI recall against accounted stored bits, including the added point."""
import csv
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent

def points(path,codec):
    rows=[r for r in csv.DictReader(path.open()) if r['dataset']=='openai' and r['codec']==codec and r['threads']=='8' and r['batch']=='32']
    result=[]
    for budget in sorted({float(r['target_bits']) for r in rows}):
        group=[r for r in rows if float(r['target_bits'])==budget]
        timed=next(r for r in group if r['seed']=='0')
        recalls=[float(r['recall10']) for r in group]
        result.append((budget,float(timed['effective_bytes_per_vector'])*8,statistics.mean(recalls),statistics.pstdev(recalls)))
    return result


def main():
    fig,ax=plt.subplots(figsize=(8.2,5.1),layout='constrained')
    for codec,color,label in [('tieredq','#1769aa','TieredQ fast · main grid'),('rabitq','#c35400','RaBitQ · main grid')]:
        for i,(budget,stored,mean,std) in enumerate(points(ROOT.parent/'output/results.csv',codec)):
            ax.errorbar(stored,mean,yerr=std,fmt={1:'o',2:'s',4:'^'}[budget],color=color,capsize=3,markersize=7,label=label if i==0 else None)
            ax.annotate(f'{budget:g}b', (stored,mean),xytext=(7,-14 if codec=='rabitq' else 9),textcoords='offset points',color=color)
    _,stored,mean,std=points(ROOT/'results.csv','tieredq')[0]
    ax.errorbar(stored,mean,yerr=std,fmt='*',markersize=14,capsize=4,color='#1769aa',markeredgecolor='black',markeredgewidth=.6,label='TieredQ fast · added storage match')
    ax.annotate('Added: 3,215.7 stored bits\nRecall@10 = 0.9326', (stored,mean),xytext=(15,-4),textcoords='offset points',fontsize=10)
    ax.set(xlabel='Accounted serving storage (bits per base vector)',ylabel='Recall@10',title='OpenAI3-1536 · measured recall–storage tradeoff',ylim=(.79,1.005),xlim=(1400,7600))
    ax.grid(alpha=.2)
    ax.legend(loc='lower right',framealpha=.95)
    fig.suptitle('1,000 queries · mean ± population SD over three recall seeds · storage seed 0',fontsize=10)
    fig.savefig(ROOT/'openai-matched-storage.png',dpi=200)


if __name__=='__main__': main()
