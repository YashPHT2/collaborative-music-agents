"""Pinned, audited Slakh MIDI ensemble data; source instruments stay explicit."""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
import urllib.request
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mido
ROOT=Path(__file__).resolve().parent

DATA=ROOT/'assets'/'trio_ensemble'
REVISION='82aa87e2f328259b9399912c71e80ac4a20619ae'
ARCHIVE_SHA='ecab00007eeb88ee8d347b4f4d96cf25ce36771ff3a627818c5d302265b0a607'
URL=f'https://huggingface.co/datasets/projectlosangeles/Slakh2100/resolve/{REVISION}/slakh2100_redux-mix_midis_only.tar.gz'
# No electric bass is silently renamed as cello; no piano is labelled organ.
PROGRAMS={'organ':set(range(16,21)),'strings':{40,41,48,49},'cello':{42}}
RANGES={'organ':(48,88),'strings':(55,91),'cello':(36,67)}


def download(url,path,sha=None):
    if not path.exists():
        temp=path.with_suffix('.part')
        with urllib.request.urlopen(url,timeout=90) as src,temp.open('wb') as out:
            while chunk:=src.read(1024*1024):out.write(chunk)
        if sha and hashlib.sha256(temp.read_bytes()).hexdigest()!=sha:raise ValueError('Dataset checksum mismatch')
        temp.replace(path)
    if sha and hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Dataset checksum mismatch')


def parse_midi(data):
    midi=mido.MidiFile(file=io.BytesIO(data))
    programs=defaultdict(int); active=defaultdict(list); notes=[]; tick=0; tempos=[]; meters=[]
    for msg in mido.merge_tracks(midi.tracks):
        tick+=msg.time
        if msg.type=='set_tempo':tempos.append(msg.tempo)
        elif msg.type=='time_signature':meters.append((msg.numerator,msg.denominator))
        elif msg.type=='program_change':programs[msg.channel]=msg.program
        elif msg.type=='note_on' and msg.velocity:
            active[msg.channel,msg.note].append((tick,msg.velocity,programs[msg.channel]))
        elif msg.type=='note_off' or (msg.type=='note_on' and not msg.velocity):
            key=msg.channel,msg.note
            if active[key]:
                start,vel,program=active[key].pop(0)
                if tick>start and msg.channel!=9:
                    notes.append(dict(start=start*480/midi.ticks_per_beat,duration=(tick-start)*480/midi.ticks_per_beat,
                                      pitch=msg.note,velocity=vel,program=program))
    return notes,tempos,meters


def canonicalize(raw,max_duration=1920):
    roles={r:[] for r in PROGRAMS}; source=defaultdict(Counter)
    for n in raw:
        for r,pg in PROGRAMS.items():
            if n['program'] not in pg:continue
            lo,hi=RANGES[r]
            if not lo<=n['pitch']<=hi:continue
            start=max(0,round(n['start']/30)*30)
            duration=max(60,min(max_duration,round(n['duration']/60)*60))
            velocity=max(32,min(96,round(n['velocity']/4)*4))
            ch={'organ':0,'strings':1,'cello':2}[r]
            roles[r].append(['note',start,duration,ch,n['pitch'],velocity]);source[r][n['program']]+=1
    # Merge duplicate onsets and trim repeated pitches to avoid ambiguous labels.
    for r,ns in roles.items():
        unique={}
        for n in ns:
            key=n[1],n[4]
            if key not in unique or n[2]>unique[key][2]:unique[key]=n
        ns=sorted(unique.values(),key=lambda n:(n[1],n[4]))
        by_pitch=defaultdict(list)
        for n in ns:by_pitch[n[4]].append(n)
        for seq in by_pitch.values():
            for a,b in zip(seq,seq[1:]):a[2]=min(a[2],(b[1]-a[1])//60*60)
        roles[r]=[n for n in ns if n[2]>=60]
    return roles,{r:dict(c) for r,c in source.items()}


def audit_song(job):
    song,split,data=job
    cache=DATA/'parsed_v2'/f'{song}.json.gz'
    if cache.exists():
        with gzip.open(cache,'rt',encoding='utf-8') as f:return json.load(f)
    try:raw,tempos,meters=parse_midi(data)
    except (ValueError,EOFError,KeyError,OSError):return dict(invalid=True,song=song,split=split)
    roles,source=canonicalize(raw)
    present=[r for r,ns in roles.items() if len(ns)>=16]
    notes=sorted([n for r,ns in roles.items() if r in present for n in ns],key=lambda n:(n[1],n[3],n[4]))
    result=dict(song=song,split=split,roles=source,available_roles=present,notes=notes,
        canonical_sha256=hashlib.sha256(json.dumps(notes).encode()).hexdigest(),
        source_bpm=round(60_000_000/(tempos[0] if tempos else 500000),2),
        variable_tempo=len(set(tempos))>1,non_4_4=bool(meters and any(m!=(4,4) for m in meters)),
        program_notes=dict(Counter(n['program'] for n in raw)))
    with gzip.open(cache,'wt',encoding='utf-8') as f:json.dump(result,f)
    return result


def prepare(workers=4):
    DATA.mkdir(parents=True,exist_ok=True); corpus=DATA/'corpus';corpus.mkdir(exist_ok=True)
    archive=DATA/'slakh_midi.tar.gz';download(URL,archive,ARCHIVE_SHA)
    redux_path=DATA/'redux.json'
    download('https://raw.githubusercontent.com/ethman/slakh-utils/master/splits/redux.json',redux_path)
    redux=json.loads(redux_path.read_text())
    audit=dict(files=0,omitted=0,invalid=0,non_4_4=0,variable_tempo=0,
               role_songs={r:Counter() for r in PROGRAMS},strict_trio_songs=Counter(),program_notes=Counter())
    records=[];jobs=[];(DATA/'parsed_v2').mkdir(exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        members=[f for f in tar.getmembers() if f.isfile() and f.name.endswith('.mid')]
        for index,f in enumerate(members):
            parts=Path(f.name).parts; song=next(s for s in parts if s.startswith('Track'))
            split=next((s for s in parts if s in ['train','validation','test','omitted']),None)
            audit['files']+=1
            if song not in redux:raise ValueError('Song absent from official split manifest')
            if redux[song]['action']=='omit' or split=='omitted':audit['omitted']+=1;continue
            jobs.append((song,split,tar.extractfile(f).read()))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index,record in enumerate(pool.map(audit_song,jobs,chunksize=4)):
            if record.get('invalid'):audit['invalid']+=1;continue
            split=record['split']
            audit['program_notes'].update(record['program_notes'])
            for r in record['available_roles']:audit['role_songs'][r][split]+=1
            if record['variable_tempo']:audit['variable_tempo']+=1
            if record['non_4_4']:audit['non_4_4']+=1
            else:
                if len(record['available_roles'])==3:audit['strict_trio_songs'][split]+=1
                if len(record['available_roles'])>=2:records.append({k:v for k,v in record.items() if k!='program_notes'})
            if (index+1)%200==0:print(f'Audited/cached {index+1}/{len(jobs)} eligible MIDI files',flush=True)
    # Canonical duplicates cannot cross our adaptation splits. Prefer held-out data.
    groups=defaultdict(list)
    for r in records:groups[r['canonical_sha256']].append(r)
    priority={'test':0,'validation':1,'train':2}
    kept=[]
    for group in groups.values():kept.append(min(group,key=lambda r:(priority[r['split']],r['song'])))
    for r in kept:
        with gzip.open(corpus/f'{r["song"]}.json.gz','wt',encoding='utf-8') as out:json.dump(r,out)
    report=dict(format_version=2,dataset='Slakh2100 MIDI-only mirror',revision=REVISION,archive_sha256=ARCHIVE_SHA,
                source_url=URL,official_split_source='https://github.com/ethman/slakh-utils/tree/master/splits',
                redux_sha256=hashlib.sha256(redux_path.read_bytes()).hexdigest(),
                role_programs={r:sorted(p) for r,p in PROGRAMS.items()},proxy_instruments_used=False,
                audit=audit,kept_songs=[{k:v for k,v in r.items() if k!='notes'} for r in kept],
                canonicalization='480 TPB, 30-tick onset grid, durations 60..1920 ticks by 60, velocity 32..96/4; role ranges and duplicate/repeated-pitch cleanup. Non-4/4 excluded. Variable source tempo normalized in beat time to fixed generation tempo.',
                missing_roles='Only real instruments present in source are retained. Supervised losses skip missing roles; pairs and full trios are eligible. Three-agent PPO follows separately.',
                attribution='Manilow et al., Cutting Music Source Separation Some Slakh, WASPAA 2019. Original Slakh CC BY 4.0; MIDI mirror card CC BY-NC-SA 4.0. Research use; preserve both notices.')
    (DATA/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'coverage':audit['role_songs'],'strict_trio':audit['strict_trio_songs'],
                      'kept_splits':Counter(r['split'] for r in kept),'proxies':False},indent=2),flush=True)
    return report


if __name__=='__main__':prepare()
