"""Phrase curation and a differentiable melodic-contour objective."""
from collections import Counter
import torch
import torch.nn.functional as F


def melodic_features(notes,channel):
    own=[n for n in notes if n[3]==channel]
    # The upper voice is an analysis view only; original ensemble labels stay intact.
    line=[max(n[4] for n in own if n[1]==t) for t in sorted({n[1] for n in own})]
    intervals=[b-a for a,b in zip(line,line[1:])]
    motifs=[tuple(intervals[i:i+3]) for i in range(len(intervals)-2)]
    repeated=sum(v-1 for k,v in Counter(motifs).items() if k!=(0,0,0))
    return dict(onsets=len(line),distinct_pitches=len(set(line)),repeated_motifs=repeated,
                stepwise_fraction=sum(0<abs(v)<=5 for v in intervals)/max(1,len(intervals)))


def preferred(window,role):
    channel={'organ':0,'strings':1,'cello':2}[role]
    f=melodic_features(window['notes'],channel)
    if role in ('organ','strings'):
        return f['onsets']>=12 and f['distinct_pitches']>=5 and f['stepwise_fraction']>=.35 and f['repeated_motifs']>=1
    own=[n for n in window['notes'] if n[3]==channel]
    return len(own)>=6 and any(n[2]>=480 for n in own) and f['distinct_pitches']>=3


def choose_window(windows,role,rng):
    focused=[w for w in windows if preferred(w,role)]
    # Keep general material and rests; do not force every composition into a style.
    return rng.choice(focused if focused and rng.random()<.75 else windows)


def contour_loss(stats,groups):
    """Match pitch intervals across note events in each real four-bar phrase."""
    by_group={}
    for (prediction,target),group in zip(stats,groups):
        by_group.setdefault(group,[]).append((prediction,target))
    losses=[]
    for rows in by_group.values():
        predictions=torch.cat([p for p,t in rows]);targets=torch.cat([t for p,t in rows])
        if len(predictions)>1:
            losses.append(F.smooth_l1_loss((predictions[1:]-predictions[:-1])/12,(targets[1:]-targets[:-1])/12))
    return torch.stack(losses).mean() if losses else None
