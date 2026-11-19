"""Audited broader-ensemble corpus for the offline composition campaign."""
import csv, gzip, hashlib, io, json, random, tarfile, urllib.request
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from pathlib import Path
from collections import Counter
import ensemble_data as ed

ROOT=ed.ROOT/'assets'/'composition_v1'
FILES={'PDMX.csv':'30392ccf38bb63ce70e7afae70f9c88c','mid.tar.gz':'d920a21b2fcd99a56d9c381b39debbb2'}

def fetch(name):
    ROOT.mkdir(parents=True,exist_ok=True);path=ROOT/name
    if name=='mid.tar.gz' and not path.exists():
        size=214395208;chunk=(size+7)//8
        def part(i):
            first=i*chunk;last=min(size-1,first+chunk-1);target=ROOT/f'midi.range{i}'
            if target.exists() and target.stat().st_size==last-first+1:return target
            request=urllib.request.Request(f'https://zenodo.org/records/15571083/files/mid.tar.gz?download=1&part={i}',headers={'Range':f'bytes={first}-{last}'})
            with urllib.request.urlopen(request,timeout=120) as src,target.open('wb') as out:
                if src.status!=206 or src.headers.get('Content-Range')!=f'bytes {first}-{last}/{size}':raise ValueError('Unexpected range response')
                while block:=src.read(1024*1024):out.write(block)
            if target.stat().st_size!=last-first+1:raise ValueError('Truncated dataset range')
            return target
        with ThreadPoolExecutor(max_workers=8) as pool:parts=list(pool.map(part,range(8)))
        with path.with_suffix('.part').open('wb') as out:
            for item in parts:
                with item.open('rb') as src:
                    while block:=src.read(1024*1024):out.write(block)
        path.with_suffix('.part').replace(path)
    if not path.exists():
        request=urllib.request.Request(f'https://zenodo.org/api/records/15571083/files/{name}/content',headers={'User-Agent':'ensemble-research/1.0','Accept-Encoding':'gzip'})
        with urllib.request.urlopen(request,timeout=120) as src,path.with_suffix('.part').open('wb') as out:
            stream=gzip.GzipFile(fileobj=src) if src.headers.get('Content-Encoding')=='gzip' else src
            while chunk:=stream.read(1024*1024):out.write(chunk)
        path.with_suffix('.part').replace(path)
    if hashlib.md5(path.read_bytes()).hexdigest()!=FILES[name]:raise ValueError('PDMX checksum mismatch: '+name)
    return path

def identity(raw):
    # Transposition-invariant onset/interval fingerprint, independent of programs.
    ns=sorted({(round(n['start']/30),n['pitch'],round(n['duration']/60)) for n in raw})
    if not ns:return ''
    t,p,_=ns[0]
    return hashlib.sha256(json.dumps([(a-t,b-p,d) for a,b,d in ns]).encode()).hexdigest()

def parse_job(job):
    source,song,split,group,blob=job
    try:raw,_,meters=ed.parse_midi(blob)
    except (ValueError,EOFError,KeyError,OSError,IndexError):return None
    if any(x!=(4,4) for x in meters):return None
    return source,song,split,group,raw

def build(limit=3000):
    path=ROOT/'corpus.json.gz'
    if path.exists() and (ROOT/'manifest.json').exists() and json.loads((ROOT/'manifest.json').read_text()).get('version')==3:
        with gzip.open(path,'rt') as f:return json.load(f)
    metadata=fetch('PDMX.csv');archive=fetch('mid.tar.gz');eligible={}
    with metadata.open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if row.get('license_conflict','').lower()!='false' or row.get('is_best_unique_arrangement','').lower()!='true':continue
            if row.get('license','').lower() not in ('publicdomain','cc0','cc-zero'):continue
            if not row.get('mid','').endswith('.mid') or float(row.get('n_tracks') or 0)<2:continue
            key=Path(row['mid']).stem
            group=(row.get('best_path') or row.get('song_name') or key).strip().lower()
            split=['test','validation','train'][min(2,int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%10)]
            tracks={int(p) for p in row.get('tracks','').split('-') if p.isdigit()}
            eligible[key]=(split,group,[r for r,programs in ed.PROGRAMS.items() if tracks&programs],float(row.get('rating') or 0))
    ranked=sorted(eligible,key=lambda k:(-eligible[k][3],hashlib.sha256(k.encode()).hexdigest()))
    selected=set(ranked[:limit//2])
    for role in ed.PROGRAMS:selected.update([k for k in ranked if role in eligible[k][2]][:limit//6])
    eligible={k:eligible[k] for k in selected}
    records=[]
    def add(song,split,raw,group,source):
        if len(raw)<32:return
        programs=sorted({n['program'] for n in raw})
        if len(programs)>15 or len(programs)<2:return
        roles,_=ed.canonicalize(raw,max_duration=3840)
        records.append(dict(song=song,group=group,source=source,split=split,raw=raw,
            fingerprint=identity(raw),notes=sorted(sum(roles.values(),[]),key=lambda n:(n[1],n[3],n[4])),
            available_roles=[r for r,ns in roles.items() if len(ns)>=16]))
    redux=json.loads((ed.DATA/'redux.json').read_text())
    jobs=[]
    with tarfile.open(ed.DATA/'slakh_midi.tar.gz') as tar:
        for m in tar:
            if not m.isfile() or not m.name.endswith('.mid'):continue
            split=next((p for p in Path(m.name).parts if p in ['train','validation','test']),None)
            if not split:continue
            song=next(p for p in Path(m.name).parts if p.startswith('Track'))
            if redux.get(song,{}).get('action')=='omit':continue
            jobs.append(('slakh','slakh:'+song,split,'slakh:'+song,tar.extractfile(m).read()))
    with tarfile.open(archive) as tar:
        accepted=0
        for m in tar:
            key=Path(m.name).stem
            if not m.isfile() or key not in eligible:continue
            split,group,_,_=eligible[key]
            jobs.append(('pdmx','pdmx:'+key,split,group,tar.extractfile(m).read()))
    print(f'Parsing {len(jobs)} selected ensemble files with four CPU workers',flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for index,result in enumerate(pool.map(parse_job,jobs,chunksize=8)):
            if result is not None:
                source,song,split,group,raw=result;add(song,split,raw,group,source)
            if (index+1)%200==0:print(f'Audited {index+1}/{len(jobs)} source files',flush=True)
    # Held-out priority; duplicate works are excluded, never moved into train.
    priority={'test':0,'validation':1,'train':2};seen=set();kept=[]
    for r in sorted(records,key=lambda r:priority[r['split']]):
        if r['fingerprint'] in seen:continue
        seen.add(r['fingerprint']);kept.append(r)
    def passages(record):
        groups={}
        for n in record['raw']:groups.setdefault(int(n['start']//7680),[]).append(n)
        return {identity(ns) for ns in groups.values() if len(ns)>=32 and len({n['pitch'] for n in ns})>=8}
    heldout=set().union(*(passages(r) for r in kept if r['split']!='train'))
    rejected={r['group'] for r in kept if r['split']=='train' and passages(r)&heldout}
    kept=[r for r in kept if not (r['split']=='train' and r['group'] in rejected)]
    manifest=dict(version=3,pdmx_record=15571083,checksums=FILES,records=len(kept),passage_overlap_training_groups_removed=len(rejected),
        sources=dict(Counter(r['source'] for r in kept)),
        role_compositions={s:{role:len({r['group'] for r in kept if r['split']==s and role in r['available_roles']}) for role in ed.PROGRAMS} for s in priority},
        limitations=['Full-score and aligned four-bar transposition-invariant deduplication; approximate arrangements and shifted boundaries can remain.',
                    'No Zimmer recordings downloaded. Instrument program labels are preserved.'])
    with gzip.open(path,'wt') as f:json.dump(kept,f)
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return kept

if __name__=='__main__':
    build();print((ROOT/'manifest.json').read_text(),flush=True)
