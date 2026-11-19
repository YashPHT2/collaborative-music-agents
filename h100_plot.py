"""Render actual run loss and per-role held-out event ratios for notebook display."""
import argparse
import json
import os
from pathlib import Path
# Jupyter may export an inline backend absent from the isolated plot environment.
# Set the file renderer before importing matplotlib, which reads MPLBACKEND.
os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    state=json.loads((args.run/'training_log.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    colors={'phrases':'#297a9c','partner':'#c06437'}
    for stage,color in colors.items():
        rows=[r for r in state.get('updates',[]) if r['stage']==stage]
        if not rows:continue
        values=[r['loss'] for r in rows];xs=[r['step'] for r in rows]
        axes[0].plot(xs,values,color=color,alpha=.18)
        smooth=[sum(values[max(0,i-7):i+1])/len(values[max(0,i-7):i+1]) for i in range(len(values))]
        axes[0].plot(xs,smooth,label=stage,color=color)
    axes[0].set(title='Actual supervised loss (8-update mean)',xlabel='Optimizer update',ylabel='Balanced loss')
    if axes[0].lines:axes[0].legend()
    records=[v for v in state.get('validations',[]) if v['stage']==state.get('stage')]
    initial=state.get('initial')
    if records and initial:
        last=records[-1];roles=list(last['scores']);ratio=[last['scores'][r]['event']/initial[r]['event'] for r in roles]
        axes[1].barh([r.replace('_',' ') for r in roles],ratio,color=['#297a9c' if x<1 else '#c06437' for x in ratio])
        axes[1].axvline(1,color='black',linestyle='--');axes[1].invert_yaxis();axes[1].set(title=f"Held-out event ratio: {state['stage']}, step {last['step']}",xlabel='Relative to this stage start; lower is better')
    else:axes[1].text(.5,.5,'Held-out validation pending',ha='center',va='center',transform=axes[1].transAxes);axes[1].axis('off')
    fig.suptitle('Prediction metrics do not establish musical enjoyment or focus benefits',fontsize=11)
    fig.savefig(args.run/'training_progress.png',dpi=130);plt.close(fig)


if __name__=='__main__':main()
