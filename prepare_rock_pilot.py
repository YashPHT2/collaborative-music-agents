"""Build composition-separated, percussion-preserving six-bar rock windows."""
import collections,gzip,hashlib,io,json,re,tarfile
from pathlib import Path
import mido
from rock_ensemble import ROLES

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/rock_training_v1'
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def parse(data):
    midi=mido.MidiFile(file=io.BytesIO(data));parts=collections.defaultdict(list);tempos=set();meters=set()
    for ti,track in enumerate(midi.tracks):
        tick=0;programs=collections.defaultdict(int);active=collections.defaultdict(list)
        for msg in track:
            tick+=msg.time
            if msg.type=='set_tempo':tempos.add(msg.tempo)
            if msg.type=='time_signature':meters.add((msg.numerator,msg.denominator))
            if msg.type=='program_change':programs[msg.channel]=msg.program
            if msg.type=='note_on' and msg.velocity:
                pg=programs[msg.channel];family='drums' if msg.channel==9 else 'guitar' if 24<=pg<=31 else 'bass' if 32<=pg<=39 else 'keyboard' if pg<8 else None
                active[msg.channel,msg.note].append((tick,msg.velocity,family))
            elif msg.type=='note_off' or msg.type=='note_on' and msg.velocity==0:
                seq=active[msg.channel,msg.note]
                if not seq:continue
                start,velocity,family=seq.pop(0)
                if family and tick>start:
                    onset=round(start*480/midi.ticks_per_beat/30)*30
                    duration=60 if family=='drums' else max(60,round((tick-start)*480/midi.ticks_per_beat/60)*60)
                    parts[family,ti,msg.channel].append(['note',onset,duration,0,msg.note,max(32,min(124,round(velocity/4)*4))])
    if len(tempos)>1 or meters and meters!={(4,4)}:raise ValueError('Variable tempo or non-4/4 score')
    bpm=round(60000000/next(iter(tempos),500000))
    if not 40<=bpm<=180:raise ValueError('Tempo outside scheduler support')
    notes=[];available=[]
    for family in ['guitar','bass','drums','keyboard']:
        sources=sorted([(key,ns) for key,ns in parts.items() if key[0]==family],key=lambda x:(-len(x[1]),x[0]))
        for index,(key,ns) in enumerate(sources[:2 if family=='guitar' else 1]):
            role=f'guitar_{index+1}' if family=='guitar' else family
            if len(ns)<16:continue
            available.append(role)
            # Exact duplicate attacks collapse; retriggers terminate the old voice.
            unique={}
            for n in ns:
                n=n.copy();n[3]=ROLES[role]['id'];k=(n[1],n[4])
                if k not in unique or unique[k][2]<n[2]:unique[k]=n
            by_pitch=collections.defaultdict(list)
            for n in sorted(unique.values(),key=lambda n:n[1]):by_pitch[n[4]].append(n)
            for seq in by_pitch.values():
                for a,b in zip(seq,seq[1:]):
                    if a[1]+a[2]>b[1]:a[2]=(b[1]-a[1])//60*60
                notes.extend(n for n in seq if n[2]>=60)
    return sorted(notes,key=lambda n:(n[1],n[3],n[4])),available,bpm

def windows(notes,available,bpm,group,split):
    result=[];end=max((n[1]+n[2] for n in notes),default=0)
    for start in range(0,max(0,end-11520)+1,3840):
        crossing=[n for n in notes if n[1]<start<n[1]+n[2]]
        if crossing:continue # no synthetic reattacks at left boundary
        ns=[n.copy() for n in notes if start<=n[1]<start+11520]
        if any(n[2]>3840 or n[1]+n[2]>start+11520 for n in ns):continue
        for n in ns:n[1]-=start
        roles=[r for r in available if any(n[3]==ROLES[r]['id'] for n in ns)]
        if not all(r in roles for r in ['guitar_1','bass','drums']):continue
        valid=True
        for role in roles:
            own=[n for n in ns if n[3]==ROLES[role]['id']];s=ROLES[role]
            if max(collections.Counter(n[1]//480 for n in own).values())>s['cap']:valid=False
            if any(sum(p[1]<=n[1]<p[1]+p[2] for p in own)>s['polyphony'] for n in own):valid=False
        if valid and len(ns)<1024:result.append(dict(id=digest([group,start]),group=group,split=split,notes=ns,available_roles=roles,bpm=bpm,start=start))
    return result

def finalize(examples,manifest):
    presence=collections.defaultdict(set)
    for w in examples:presence[w['group']].update(w['available_roles'])
    remaining=set(presence);assignment={}
    # Role-stratified composition split, based only on availability, never losses.
    for split,minimum in [('validation',3),('test',2)]:
        counts=collections.Counter()
        while any(counts[r]<minimum for r in ROLES):
            choices=[]
            for g in remaining:
                if any(sum(r in presence[h] for h in remaining if h!=g)<4 for r in ROLES):continue
                gain=sum(counts[r]<minimum for r in presence[g])
                if gain:choices.append((-gain,digest(g),g))
            if not choices:raise ValueError('Cannot stratify independent compositions while preserving four train groups per role')
            g=min(choices)[2];assignment[g]=split;remaining.remove(g);counts.update(presence[g])
    assignment.update({g:'train' for g in remaining})
    for w in examples:w['split']=assignment[w['group']]
    for source in manifest['sources']:
        if source['group'] in assignment:source['split']=assignment[source['group']]
    counts={s:{r:len({w['group'] for w in examples if w['split']==s and r in w['available_roles']}) for r in ROLES} for s in ['train','validation','test']}
    path=OUT/'rock_examples.json.gz'
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(examples,f)
    manifest.update(composition_groups=counts,pilot_data_gate_passed=all(counts['train'][r]>=4 and counts['validation'][r]>=3 and counts['test'][r]>=2 for r in ROLES),examples_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),split_method='Deterministic role-stratified composition groups; role availability only, no loss or listener outcomes used.')
    (OUT/'rock_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['sources','rejected']},indent=2),flush=True)

def main():
    import sys
    if '--resplit' in sys.argv:
        with gzip.open(OUT/'rock_examples.json.gz','rt') as f:examples=json.load(f)
        return finalize(examples,json.loads((OUT/'rock_manifest.json').read_text()))
    audit=json.loads((OUT/'data_audit.json').read_text());lookup={r['mid'].lstrip('./'):r for r in audit['bands'] if r['license'] in ['cc-zero','publicdomain'] and r['license_conflict']=='False'}
    parsed=[];rejected=[]
    with tarfile.open(ROOT/'assets/composition_v1/mid.tar.gz','r:gz') as archive:
        for member in archive:
            key=member.name.lstrip('./')
            if key not in lookup:continue
            row=lookup[key]
            try:
                data=archive.extractfile(member).read()
                assert hashlib.sha256(data).hexdigest()==row['midi_sha256']
                notes,roles,bpm=parse(data)
                title=re.sub(r'[^a-z0-9]','',row['title'].lower())
                # Transposition-invariant full-score fingerprint, drums unshifted.
                pitched=[n[4] for n in notes if n[3]!=ROLES['drums']['id']];low=min(pitched,default=0)
                fp=digest([[n[1],n[2],n[3],n[4] if n[3]==ROLES['drums']['id'] else n[4]-low] for n in notes])
                parsed.append(dict(row=row,notes=notes,roles=roles,bpm=bpm,title=title,fp=fp))
            except Exception as exc:rejected.append(dict(id=row['id'],reason=str(exc)))
    # Connected components join duplicate titles and duplicate normalized scores.
    parent=list(range(len(parsed)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,item in enumerate(parsed):
        for token in [('title',item['title']),('score',item['fp'])]:
            if token in seen:parent[find(i)]=find(seen[token])
            seen[token]=i
    components=collections.defaultdict(list)
    for i in range(len(parsed)):components[find(i)].append(i)
    ordered=sorted(components.values(),key=lambda ids:digest(sorted(parsed[i]['fp'] for i in ids)))
    examples=[];provenance=[];fingerprints=set()
    for index,ids in enumerate(ordered):
        group=digest(sorted(parsed[i]['title'] for i in ids));split='validation' if index%7==0 else 'test' if index%7==1 else 'train'
        for i in ids:
            item=parsed[i]
            if item['fp'] in fingerprints:continue
            fingerprints.add(item['fp']);ws=windows(item['notes'],item['roles'],item['bpm'],group,split);examples.extend(ws)
            provenance.append(dict(source=item['row']['id'],group=group,split=split,windows=len(ws),license=item['row']['license'],source_sha256=item['row']['midi_sha256']))
    counts={s:{r:len({w['group'] for w in examples if w['split']==s and r in w['available_roles']}) for r in ROLES} for s in ['train','validation','test']}
    passed=all(counts['train'][r]>=4 and counts['validation'][r]>=3 for r in ROLES)
    path=OUT/'rock_examples.json.gz'
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(examples,f)
    manifest=dict(schema='rock-ensemble-pilot-v1',examples=len(examples),composition_groups=counts,pilot_data_gate_passed=passed,examples_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),happy_labels=False,
        role_assignment='Top two real guitar source tracks by note count; numbered, not claimed lead/rhythm ground truth. Most populated real bass/drum/keyboard tracks retained.',
        deduplication='Connected components of normalized titles and transposition-invariant full-score fingerprints; approximate arrangement duplicates may remain.',
        limitations=['Genre metadata is unverified; happy mood is unlabelled.','Pilot validation is small; not a production-quality certification.'],rejected=rejected,sources=provenance)
    finalize(examples,manifest)

if __name__=='__main__':main()
