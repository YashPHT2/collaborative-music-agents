"""Track-aware partial-ensemble data for all 19 orchestral role slots."""
import csv
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import mido
import audit_functional_roles as audit
import composition_data as cd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/orchestra_learning_v1/data'
SLOT_GROUPS={'pipe_organ':[16,17,18,19,20],'violins':[40],'violas':[41],'cellos':[42],'double_basses':[43],
 'piano_anchor':[0,1,2,3],'flutes':[73],'oboes':[68],'clarinets':[71],'bassoons':[70],
 'french_horns':[60],'trombones':[57],'tuba':[58],'synth_pad':list(range(88,96)),'choir':[52,53,54]}
SLOTS={'violins':['violins','second_violins'],'piano_anchor':['piano_anchor','piano_motion','piano_pulse','piano_counterline']}


def write(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(value):return hashlib.sha256(value).hexdigest()


def recover(job):
    meta,blob,directory=job;path=Path(directory)/(digest(meta['song'].encode())+'.json.gz')
    if path.exists():return {**meta,'cache':str(path)}
    try:
        midi=mido.MidiFile(file=io.BytesIO(blob))
        if any(m.type=='time_signature' and (m.numerator,m.denominator)!=(4,4) for t in midi.tracks for m in t):return None
        parsed=audit.parse_with_tracks(blob);parsed['midi_sha256']=digest(blob)
        with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(parsed,f)
        return {**meta,'cache':str(path)}
    except Exception:return None


def metadata():
    with gzip.open(ROOT/'assets/composition_v1/corpus.json.gz','rt') as f:original=json.load(f)
    known_songs={r['song'] for r in original};known_groups={r['group'] for r in original}
    result=[]
    for r in original:
        path=ROOT/'outputs/functional_role_audit_v1/source_tracks'/(digest(r['song'].encode())+'.json.gz')
        if not path.exists():raise ValueError('Original verified source cache missing')
        result.append({k:r[k] for k in ('song','group','source','split')}|dict(cache=str(path),fingerprint=r['fingerprint']))
    for r in read(ROOT/'outputs/supported_corpus_expansion_v1/accepted_sources.json'):
        known_songs.add(r['song']);known_groups.add(r['group'])
        path=ROOT/'outputs/supported_corpus_expansion_v1/source_tracks'/(digest(r['song'].encode())+'.json.gz')
        result.append({**r,'cache':str(path)})
    rare=[70,68,41,43,58,89,19];candidates={}
    with (ROOT/'assets/composition_v1/PDMX.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if row.get('license_conflict','').lower()!='false' or row.get('is_best_unique_arrangement','').lower()!='true':continue
            if row.get('license','').lower() not in ['publicdomain','cc0','cc-zero'] or not row.get('mid','').endswith('.mid'):continue
            key=Path(row['mid']).stem;song='pdmx:'+key;group=(row.get('best_path') or row.get('song_name') or key).strip().lower()
            if song in known_songs or group in known_groups:continue
            programs={int(x) for x in row.get('tracks','').split('-') if x.isdigit()}
            if not programs.intersection(rare) or len(programs)<2:continue
            split=['test','validation','train'][min(2,int(digest(group.encode())[:8],16)%10)]
            if split=='test':continue
            item=dict(song=song,group=group,source='pdmx',split=split,licence=row['license'],metadata_path=row['mid'],programs=sorted(programs))
            if group not in candidates or song<candidates[group]['song']:candidates[group]=item
    selected={}
    for program in rare:
        for split,cap in [('train',60),('validation',50)]:
            pool=sorted([r for r in candidates.values() if r['split']==split and program in r['programs']],key=lambda r:digest(r['group'].encode()))
            for r in pool[:cap]:selected[r['song']]=r
    selected=list(selected.values());write(OUT/'additional_sources_selected.json',selected)
    print('Additional rare-role candidates:',dict(Counter(r['split'] for r in selected)),flush=True)
    lookup={Path(r['metadata_path']).stem:r for r in selected};cache=OUT/'source_tracks';cache.mkdir(exist_ok=True)
    def jobs():
        found=set()
        with tarfile.open(ROOT/'assets/composition_v1/mid.tar.gz','r|gz') as tar:
            for member in tar:
                key=Path(member.name).stem
                if not member.isfile() or key not in lookup or key in found:continue
                found.add(key);meta=lookup[key];path=cache/(digest(meta['song'].encode())+'.json.gz')
                yield meta,None if path.exists() else tar.extractfile(member).read(),str(cache)
        if found!=set(lookup):raise ValueError('Pinned archive missing a selected source')
    with ProcessPoolExecutor(max_workers=4) as pool:
        for index,item in enumerate(pool.map(recover,jobs(),chunksize=4)):
            if item is not None:result.append(item)
            if (index+1)%50==0:print('Recovered rare-role sources',index+1,flush=True)
    return result


def project(part,role,role_specs):
    settings=role_specs[role];result=[];counts=Counter();active=[]
    for n in sorted(part,key=lambda n:(n['start'],n['pitch'])):
        if n['pitch'] not in settings['pitches']:continue
        tick=round(n['start']/30)*30;duration=int(n['duration']//60)*60
        if duration<60:continue
        if counts[tick//480]>=settings['cap']:continue
        active=[x for x in active if x[1]+x[2]>tick]
        if len(active)>=settings['polyphony'] or any(x[4]==n['pitch'] for x in active):continue
        row=['note',tick,duration,settings['id'],n['pitch'],min(84,max(32,round(n['velocity']/4)*4))]
        result.append(row);active.append(row);counts[tick//480]+=1
    return result


def prepare_work(meta,role_specs):
    with gzip.open(meta['cache'],'rt',encoding='utf-8') as f:parsed=json.load(f)
    parts=defaultdict(list)
    for n in parsed['notes']:parts[n['source_track'],n['source_channel'],n['program']].append(n)
    candidates={base:sorted([key for key in parts if key[2] in programs]) for base,programs in SLOT_GROUPS.items()}
    end=max((n['start']+n['duration'] for n in parsed['notes']),default=0);starts=list(range(3840,max(3840,int(end)-7680+1),7680))
    cap=16 if meta['split']=='train' else 40
    starts=sorted(sorted(starts,key=lambda x:digest((meta['song']+':'+str(x)).encode()))[:cap])
    windows=[];rejections=Counter()
    for start in starts:
        left=start-3840;right=start+7680;assignment={};notes=[];unsupported=False
        for base,keys in candidates.items():
            slots=SLOTS.get(base,[base]);rotation=int(digest((meta['song']+':'+str(start)+':'+base).encode())[:8],16)%len(slots)
            rotated=slots[rotation:]+slots[:rotation]
            for role,key in zip(rotated,keys):
                raw=parts[key]
                touching=[n for n in raw if n['start']<right and n['start']+n['duration']>left]
                if not touching:continue
                if any(n['duration']>3840 and n['pitch'] in role_specs[role]['pitches'] for n in touching):unsupported=True;break
                full=project(raw,role,role_specs);own=[]
                for n in full:
                    # Onsets before the two-bar prefix cannot carry across its
                    # boundary under the eight-beat support cap.
                    if left<=n[1]<right:
                        n=copy.deepcopy(n);n[1]-=left;n[2]=min(n[2],11520-n[1])//60*60
                        if n[2]>0:own.append(n)
                ranged=sum(n['pitch'] in role_specs[role]['pitches'] for n in touching if start<=n['start']<right)
                kept=sum(n[1]>=3840 for n in own)
                if ranged and kept/ranged<.6:continue
                if not own:continue
                assignment[role]=dict(source_track=key[0],source_channel=key[1],source_program=key[2],
                    source_name=raw[0]['source_name'],provenance='exact' if key[2]==role_specs[role]['patch'] else 'family_patch_variant',
                    weight=1. if key[2]==role_specs[role]['patch'] else .5)
                notes.extend(own)
            if unsupported:break
        if unsupported:rejections['long_sustain_window']+=1;continue
        if len(assignment)<2:rejections['fewer_than_two_supported_parts']+=1;continue
        if len({role_specs[r]['family'] for r in assignment})<2:rejections['single_family_only']+=1;continue
        notes.sort(key=lambda n:(n[1],n[3],n[4]))
        if not any(n[1]>=3840 for n in notes):continue
        # Part ownership is one-to-one in each phrase. Symmetric piano/violin
        # slots rotate between windows, never clone one part inside a window.
        assert len({(v['source_track'],v['source_channel'],v['source_program']) for v in assignment.values()})==len(assignment)
        anchor=next((n[4] for n in notes if n[1]>=3840),0)
        normalized=sorted((list(role_specs.values())[n[3]]['family'],n[1]-3840,n[2],n[4]-anchor) for n in notes if n[1]>=3840)
        windows.append(dict(id=digest((meta['song']+':'+str(start)).encode())[:16],song=meta['song'],group=meta['group'],
            split=meta['split'],start_tick=start,notes=notes,assignment=assignment,available_roles=list(assignment),bars=6,
            prefix_ticks=3840,target_ticks=7680,role_weights={r:v['weight'] for r,v in assignment.items()},
            normalized_fingerprint=digest(json.dumps(normalized).encode()),source_midi_sha256=parsed['midi_sha256']))
    return windows,dict(rejections)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    from orchestra_learning import ROLES,commit,style
    sources=metadata();sources.sort(key=lambda r:({'test':0,'validation':1,'train':2}[r['split']],r['group'],r['song']))
    write(OUT/'sources.json',sources)
    all_windows=[];seen_whole={};rejections=Counter()
    for index,meta in enumerate(sources):
        if not meta.get('cache'):continue
        with gzip.open(meta['cache'],'rt',encoding='utf-8') as f:parsed=json.load(f)
        identity=parsed.get('whole_identity') or cd.identity(parsed['notes'])
        if identity in seen_whole and seen_whole[identity]!=meta['group']:rejections['duplicate_work']+=1;continue
        seen_whole[identity]=meta['group'];del parsed
        rows,excluded=prepare_work(meta,ROLES);all_windows.extend(rows);rejections.update(excluded)
        if (index+1)%200==0:print('Prepared partial ensembles',index+1,'windows',len(all_windows),flush=True)
    # Any cross-split normalized target duplicate excludes the lower-priority
    # composition in its entirety, not just the matching window.
    priority={'test':0,'validation':1,'train':2};owner={};excluded_groups=set()
    for w in all_windows:
        key=w['normalized_fingerprint'];previous=owner.get(key)
        if previous and previous['split']!=w['split']:excluded_groups.add((w['split'],w['group']))
        else:owner.setdefault(key,w)
    kept=[];seen=set()
    for w in all_windows:
        if (w['split'],w['group']) in excluded_groups or w['normalized_fingerprint'] in seen:continue
        seen.add(w['normalized_fingerprint']);ledger=[]
        for beat in range(24):ledger=commit(ledger,{r:[n for n in w['notes'] if n[3]==s['id'] and n[1]//480==beat] for r,s in ROLES.items()},beat,6)
        assert ledger==w['notes'];w['style']=style(w['notes'],w['available_roles']).tolist();kept.append(w)
    groups={s:{r:len({w['group'] for w in kept if w['split']==s and r in w['available_roles']}) for r in ROLES} for s in priority}
    with gzip.open(OUT/'examples.json.gz','wt',encoding='utf-8') as f:json.dump(kept,f)
    summary=dict(examples=len(kept),splits={s:sum(w['split']==s for w in kept) for s in priority},composition_groups=groups,
        minimum_validation_groups=12,validation_gate_passed=all(v>=12 for v in groups['validation'].values()),
        rejections=dict(rejections),excluded_cross_split_groups=len(excluded_groups),
        examples_sha256=digest((OUT/'examples.json.gz').read_bytes()),training_started=False,
        limitations=['Partial ensembles, not 19-part complete target scores.','Piano/violin slots share family adapters and rotate real source parts between windows; no part is cloned inside a phrase.','Organ/synth/choir patch-family variants are weak timbre labels with weight 0.5.','Calm/major-like statistics are controls, not human mood labels.'])
    write(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
