"""Versioned, causal ensemble heuristics. Human preference is measured separately."""
import math
from collections import Counter

TPB=480
VERSION='band-heuristic-v2'

def attacks(notes,left,right,ch):
    return sorted((n for n in notes if n[3]==ch and left<=n[1]<right),key=lambda n:(n[1],n[4]))

def topline(notes):
    groups={}
    for n in notes: groups[n[1]]=max(groups.get(n[1],0),n[4])
    return sorted(groups.items())

def coverage(notes,left,right,ch):
    boundaries=sorted({left,right}|{max(left,n[1]) for n in notes if n[3]==ch and n[1]<right and n[1]+n[2]>left}
                      |{min(right,n[1]+n[2]) for n in notes if n[3]==ch and n[1]<right and n[1]+n[2]>left})
    return sum(b-a for a,b in zip(boundaries,boundaries[1:]) if any(n[3]==ch and n[1]<=a<n[1]+n[2] for n in notes))/(right-left)

def ensemble_reward_v2(ledger,start,end,brief):
    # Discard every future event, even if accidentally supplied by a caller.
    notes=[n for n in ledger if n[1]<end]
    beats=(end-start)/TPB
    active=[n for n in notes if n[1]+n[2]>start]
    piano=[n for n in active if n[3]==0]; bass=[n for n in active if n[3]==1]
    total=support=0.
    for p in piano:
        for b in bass:
            duration=max(0,min(end,p[1]+p[2],b[1]+b[2])-max(start,p[1],b[1]))
            if not duration: continue
            pc=(p[4]-b[4])%12
            fit=1. if pc in [0,3,4,5,7,8,9] else .45
            if pc in [1,2,6,10,11]:
                # Credit passing tension when an observed later step resolves it.
                following=[n for n in notes if n[3]==0 and p[1]<n[1]<min(end,p[1]+2*TPB+1)
                           and abs(n[4]-p[4])<=2 and (n[4]-b[4])%12 in [0,3,4,5,7,8,9]
                           and n[1]<b[1]+b[2]]
                if following: fit=.9
            total+=duration; support+=duration*fit
    harmony=support/total if total else 0.
    low=attacks(notes,start,end,1)
    drums=attacks(notes,start,end,9)
    kicks=[n for n in drums if n[4]==36]
    # Bass can sit around a kick; simultaneous onsets are not required everywhere.
    coupling=sum(any(abs(b[1]-k[1])<=120 for k in kicks) for b in low)/max(1,len(low))
    def pattern(left,right):
        return Counter(((n[1]-left)%1920//120,n[4]) for n in attacks(notes,left,right,9))
    now=pattern(start,end); before=pattern(max(0,start-1920),start) if start else Counter()
    intersection=sum((now&before).values()); union=sum((now|before).values())
    continuity=intersection/union if union else 0.
    groove=.6*coupling+.4*continuity
    # Only upper voices at each onset define melody; chords do not become fake runs.
    recent=topline(attacks(notes,max(0,end-3840),end,0))
    past=topline(attacks(notes,max(0,end-15360),max(0,end-3840),0))
    def motifs(line):
        return [(tuple(line[j+1][1]-line[j][1] for j in range(i,i+2)),
                 tuple((line[j+1][0]-line[j][0])//120 for j in range(i,i+2))) for i in range(len(line)-2)]
    cm,pm=motifs(recent),set(motifs(past))
    recurrence=sum(m in pm for m in cm)/max(1,len(cm)) if pm else 0.
    variation=min(1,len(set(m[0] for m in cm))/3) if cm else 0.
    phrase=recurrence*(.5+.5*variation)
    energy=brief.energy(start/TPB)
    targets={0:.4+1.6*energy,1:.2+.8*energy,9:1+3*energy}
    arrangement=sum(math.exp(-abs(len(attacks(notes,start,end,ch))/beats-target)/(target+1))
                    for ch,target in targets.items())/3
    # A phrase may rest; assess contribution across the last four bars instead.
    window=max(0,end-7680)
    participation=sum(min(1,coverage(notes,window,end,ch)/target)
                      for ch,target in [(0,.2),(1,.25),(9,.05)])/3
    penalty=1. if not active else 0.
    lead=topline(attacks(notes,max(0,end-7680),end,0))
    if len(lead)>=8:
        dominance=max(Counter(p for _,p in lead).values())/len(lead)
        penalty+=max(0,(dominance-.85)/.15)*.35
    # Flag identical melodic bars repeated three times; bass/drum ostinatos remain valid.
    if end>=5760:
        signatures=[]
        for left in [end-5760,end-3840,end-1920]:
            signatures.append(tuple((t-left,p) for t,p in topline(attacks(notes,left,left+1920,0))))
        if signatures[0] and signatures[0]==signatures[1]==signatures[2]: penalty+=.25
    density=sum(start<=n[1]<end for n in notes)/beats
    penalty+=.4*min(1,max(0,density-12)/6)
    penalty=min(1,penalty)
    parts=dict(harmony=harmony,groove=groove,phrase=phrase,arrangement=arrangement,
               participation=participation,degeneracy=penalty)
    score=(harmony+groove+phrase+arrangement+participation)/5-penalty
    return score,parts
