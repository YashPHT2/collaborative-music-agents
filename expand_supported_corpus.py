"""Versioned public-domain/CC0 expansion, preserving the original corpus/splits.

Uses additional entries already present in the pinned local PDMX archive. Frozen
metadata eligibility and hash ranking; no quality-score or listening selection.
New train/validation works are rejected on original whole/passage overlap. The
existing test pool is retained; no test work is moved to satisfy validation.
"""
import csv
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import audit_functional_roles as a
import audit_projected_roles as p
import audit_supported_phrase_gate as g
import composition_data as cd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/supported_corpus_expansion_v1'
PROTOCOL=dict(version=1,source='Pinned local PDMX record 15571083; additional public-domain/CC0 entries',
    licences=['publicdomain','cc0','cc-zero'],no_recorded_licence_conflict=True,best_unique_arrangement=True,
    group='Existing best_path/song_name rule; new group cannot already be present in original corpus',
    split='Original SHA256(group) modulo-10 split rule; no reassignment and no additional test selection',
    selection='One source per group, ascending SHA256(group); up to 320 new train groups and all eligible new validation groups',
    role_programs='Unchanged functional_role_audit_v1 sets; requires metadata candidate coverage for all roles',
    transfer_thresholds='Unchanged functional_role_audit_v1',
    support='Unchanged supported_phrase_gate_v1: two-bar context, four-bar target, exclude relevant notes above eight beats',
    duplicates='Reject new work if whole score or aligned four-bar source passage matches any original work; compare new works across splits with held-out priority',
    passage_min_notes=32,passage_min_pitches=8,
    training_started=False,original_corpus_replaced=False)


def passage_ids(raw):
    buckets=defaultdict(list)
    for n in raw:buckets[int(n['start']//7680)].append(n)
    return {cd.identity(ns) for ns in buckets.values() if len(ns)>=32 and len({n['pitch'] for n in ns})>=8}


def metadata_candidates(original):
    songs={r['song'] for r in original};groups={r['group'] for r in original};candidates={}
    with (ROOT/'assets/composition_v1/PDMX.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if row.get('license_conflict','').lower()!='false' or row.get('is_best_unique_arrangement','').lower()!='true':continue
            if row.get('license','').lower() not in PROTOCOL['licences'] or not row.get('mid','').endswith('.mid'):continue
            key=Path(row['mid']).stem;song='pdmx:'+key
            group=(row.get('best_path') or row.get('song_name') or key).strip().lower()
            if song in songs or group in groups:continue
            programs={int(x) for x in row.get('tracks','').split('-') if x.isdigit()}
            if not all(programs&(a.EXACT[r]|a.TRANSFER[r]) for r in a.ROLES):continue
            split=['test','validation','train'][min(2,int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%10)]
            if split=='test':continue
            item=dict(song=song,group=group,split=split,source='pdmx',licence=row['license'],
                      metadata_path=row['mid'],licence_conflict=False,best_unique_arrangement=True)
            if group not in candidates or song<candidates[group]['song']:candidates[group]=item
    selected=[]
    for split,cap in [('train',320),('validation',1000000)]:
        selected.extend(sorted([r for r in candidates.values() if r['split']==split],key=lambda r:hashlib.sha256(r['group'].encode()).hexdigest())[:cap])
    return selected


def audit_job(job):
    meta,blob,directory=job;directory=Path(directory)
    cache=directory/'source_tracks'/(hashlib.sha256(meta['song'].encode()).hexdigest()+'.json.gz')
    source_log=directory/'source_windows'/(cache.stem+'.json.gz')
    if cache.exists() and source_log.exists():return dict(meta=meta,cache=str(cache),windows=str(source_log))
    try:
        parsed=a.parse_with_tracks(blob)
        midi=mido_info(blob)
        if midi['non_four_four']:return dict(meta=meta,error='non_4_4_source')
        parsed['midi_sha256']=hashlib.sha256(blob).hexdigest()
        parsed['notes_identity']=a.note_identity(parsed['notes'])
        parsed['whole_identity']=cd.identity(parsed['notes'])
        parsed['passage_ids']=sorted(passage_ids(parsed['notes']))
        with gzip.open(cache,'wt',encoding='utf-8') as f:json.dump(parsed,f)
        windows=io.StringIO();rejections=io.StringIO()
        a.audit_record(meta,parsed,windows,rejections)
        qualified=[r for line in windows.getvalue().splitlines() if (r:=json.loads(line))['category'] in ('exact_full_trio','transferred_full_trio')]
        for row in qualified:row['source_cache_root']=str((directory/'source_tracks').resolve())
        with gzip.open(source_log,'wt',encoding='utf-8') as f:json.dump(qualified,f)
        return dict(meta=meta,cache=str(cache),windows=str(source_log))
    except Exception as error:return dict(meta=meta,error=str(error))


def mido_info(blob):
    import mido
    midi=mido.MidiFile(file=io.BytesIO(blob))
    return dict(non_four_four=any(msg.type=='time_signature' and (msg.numerator,msg.denominator)!=(4,4) for track in midi.tracks for msg in track))


def jobs(selected,directory):
    lookup={Path(r['metadata_path']).stem:r for r in selected};found=set()
    with tarfile.open(ROOT/'assets/composition_v1/mid.tar.gz','r|gz') as tar:
        for member in tar:
            key=Path(member.name).stem
            if not member.isfile() or key not in lookup or key in found:continue
            found.add(key)
            meta=lookup[key];cache=directory/'source_tracks'/(hashlib.sha256(meta['song'].encode()).hexdigest()+'.json.gz')
            log=directory/'source_windows'/(cache.stem+'.json.gz')
            yield meta,None if cache.exists() and log.exists() else tar.extractfile(member).read(),str(directory)
    if set(lookup)!=found:raise ValueError('Selected sources missing from pinned archive')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['source_tracks','source_windows']:(OUT/name).mkdir(exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists() and p.read_json(path)!=PROTOCOL:raise ValueError('Expansion protocol changed; choose new version')
    if not path.exists():a.write(path,PROTOCOL)
    original_path=ROOT/'assets/composition_v1/corpus.json.gz';original_hash=a.digest(original_path)
    with gzip.open(original_path,'rt') as f:original=json.load(f)
    selected=metadata_candidates(original);a.write(OUT/'selected_sources.json',selected)
    print('Additional source candidates:',dict(Counter(r['split'] for r in selected)),flush=True)
    original_whole={r['fingerprint'] for r in original};original_passages=set()
    for record in original:original_passages.update(passage_ids(record['raw']))
    del original
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for index,result in enumerate(pool.map(audit_job,jobs(selected,OUT),chunksize=2)):
            results.append(result)
            if (index+1)%25==0:print(f'Additional source track audit {index+1}/{len(selected)}',flush=True)
    rejections=[];accepted=[]
    for result in results:
        if 'error' in result:rejections.append(dict(**result['meta'],reason=result['error']));continue
        with gzip.open(result['cache'],'rt',encoding='utf-8') as f:parsed=json.load(f)
        if parsed['whole_identity'] in original_whole or set(parsed['passage_ids'])&original_passages:
            rejections.append(dict(**result['meta'],reason='duplicate_original_whole_or_aligned_passage'));continue
        result['whole']=parsed['whole_identity'];result['passages']=set(parsed['passage_ids']);accepted.append(result)
    # New works cannot introduce a known cross-split passage leak. Exclude the
    # lower-priority new composition as a whole, rather than moving its split.
    seen_whole=set();seen_passages=set();new_windows=[];accepted_sources=[]
    for result in sorted(accepted,key=lambda r:(r['meta']['split']!='validation',r['meta']['group'])):
        if result['whole'] in seen_whole or result['passages']&seen_passages:
            rejections.append(dict(**result['meta'],reason='duplicate_another_expansion_work'));continue
        seen_whole.add(result['whole']);seen_passages.update(result['passages'])
        with gzip.open(result['windows'],'rt',encoding='utf-8') as f:windows=json.load(f)
        accepted_sources.append(result['meta']);new_windows.extend(windows)
    a.write(OUT/'source_rejections.json',rejections);a.write(OUT/'accepted_sources.json',accepted_sources)
    print('Auditing supported windows:',len(new_windows),flush=True)
    rows=p.read_json(g.OUT/'windows.json')
    for index,window in enumerate(new_windows):
        rows.append(g.audit_window(window))
        if (index+1)%50==0:print(f'Additional supported phrases {index+1}/{len(new_windows)}',flush=True)
    a.write(OUT/'windows.json',rows)
    kept,dedup=p.deduplicate(rows,'eligible');a.write(OUT/'deduplication.json',dedup)
    a.write(OUT/'eligible_examples.json',kept)
    summary=dict(gates={trainer:g.group_gate(kept,minimum) for trainer,minimum in g.PROTOCOL['validation_requirements'].items()},
        splits={s:dict(windows=sum(r['split']==s for r in kept),groups=len({r['group'] for r in kept if r['split']==s})) for s in ['train','validation','test']},
        additional_sources_selected=len(selected),additional_sources_accepted=len(accepted_sources),source_rejections=len(rejections),
        eligible_new_groups={s:len({r['group'] for r in kept if r['split']==s and r['id'] not in {x['id'] for x in p.read_json(g.OUT/'windows.json')}}) for s in ['train','validation']},
        original_corpus_unchanged=a.digest(original_path)==original_hash,original_corpus_sha256=original_hash,
        protocol_sha256=a.digest(path),eligible_examples_sha256=a.digest(OUT/'eligible_examples.json'),
        training_loader_integrated=False,supervised_training_allowed=False,training_started=False,
        pdmx_metadata_md5=cd.FILES['PDMX.csv'],pdmx_archive_md5=cd.FILES['mid.tar.gz'])
    if not summary['original_corpus_unchanged']:raise ValueError('Original corpus changed')
    a.write(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
