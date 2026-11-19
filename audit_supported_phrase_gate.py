"""Fail-closed validation audit for legal boundary sustains; no model training.

Audits the fixed 443-window candidate pool. Complete notes over eight beats are
excluded when relevant to context/targets, never split into fake repeated attacks.
Supported notes keep their original quantized onset in a positive-time two-bar
prefix. Loss is restricted to the following four bars. The current trainers are
NOT silently switched to this new example schema.
"""
import argparse
from collections import Counter, defaultdict
import copy
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

import audit_functional_roles as a
import audit_projected_roles as p

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/supported_phrase_gate_v1'
ROLES = a.ROLES
CAP = 3840
TARGET = 7680
PROTOCOL = dict(version=1, input='Fixed exact and transferred full-trio candidate pool from functional_role_audit_v1',
    source_notes='Original complete notes with recovered track/channel/program identity',
    duration_support='Exclude a window if any selected in-range source note overlapping its context/target exceeds 3840 ticks. No ties, synthesized onsets, or long-note truncation to make a window pass.',
    prefix_ticks=3840, target_ticks=7680, grids=dict(onset=30,duration=60),
    boundary='Canonicalize/project the source part before cropping. Include supported incoming target sustains at their original quantized onset in the preceding positive-time prefix.',
    losses='Only decisions at target beats are supervised. Prefix notes are context, not new target attacks. Missing roles receive no loss; genuine rests remain targets.',
    role_states=['missing','present_resting','present_sustaining','present_sparse','present_active'],
    state_limits='Sparse/active describes 1..7 versus >=8 onsets; it is not eligibility. A genuine rest requires adjacent role evidence and no original target activity removed by projection.',
    projection='Existing se.playable_projection under offline limits on whole source parts, then supported context/target cropping and exact commit validation',
    behavioral_gates='Original functional-role thresholds unchanged; recheck target sounding state including incoming notes',
    dedup='Role-aware exact/global-transposition target fingerprints include target-crossing notes at original relative onsets; held-out priority and whole-group exclusions as in projection audit',
    validation_requirements=dict(train_offline_ensemble=10,train_ensemble_repair=12),
    scope='Counts apply to this candidate pool; no claim about broader unchanged corpus or newly added scores',
    training_adapter_integrated=False, training_started=False,
    limitations=['Source labels and functional transfers are weak musical evidence, not a quality guarantee.',
                 'Future continuation is used as target labels, never exposed as conditioning notes.',
                 'A data-count pass alone does not enable a trainer without the new boundary/role-mask loader.',
                 'No splits, transfer thresholds, model weights, PPO gates or existing training caches change.'])


def overlap(note, start, stop):
    return note['start'] < stop and note['start'] + note['duration'] > start


def canonical_part(source, role):
    """Canonical full part, with explicit exclusion audited separately by crop."""
    channel = ROLES.index(role)
    lo, hi = a.RANGES[role]
    unique = {}
    for n in source:
        if not lo <= n['pitch'] <= hi: continue
        tick = max(0,round(n['start']/30)*30)
        duration = max(60,min(CAP,round(n['duration']/60)*60))
        velocity = max(32,min(96,round(n['velocity']/4)*4))
        note = ['note',tick,duration,channel,n['pitch'],velocity]
        key = tick,n['pitch']
        if key not in unique or duration > unique[key][2]: unique[key] = note
    result = sorted(unique.values(),key=lambda n:(n[1],n[4]))
    by_pitch = defaultdict(list)
    for n in result: by_pitch[n[4]].append(n)
    for voice in by_pitch.values():
        for note, following in zip(voice,voice[1:]):
            note[2] = min(note[2],(following[1]-note[1])//60*60)
    return [n for n in result if n[2] >= 60]


@lru_cache(maxsize=256)
def full_projected_part(song, role, track, channel, program, midi_sha256, cache_root=None):
    from supervised_ensemble import playable_projection
    import offline_ensemble as oe
    cache = (Path(cache_root) if cache_root else a.OUT/'source_tracks')/(hashlib.sha256(song.encode()).hexdigest()+'.json.gz')
    with gzip.open(cache,'rt',encoding='utf-8') as f: parsed=json.load(f)
    if parsed['midi_sha256'] != midi_sha256: raise ValueError('Changed source MIDI')
    source = [dict(n,source_note_id=i) for i,n in enumerate(parsed['notes']) if
              n['source_track']==track and n['source_channel']==channel and n['program']==program]
    canonical = canonical_part(source,role)
    with oe.limits(): projected=playable_projection(canonical)
    # Validating only the sliced container below retains existing event support.
    return source, projected


def slice_supported(projected, target_start):
    """Preserve attacks in context. Never move a carry-in onset to target zero."""
    prefix_start = max(0,target_start-CAP)
    stop = target_start+TARGET
    result=[]
    for n in projected:
        if not prefix_start <= n[1] < stop: continue
        note=copy.deepcopy(n)
        note[1]-=prefix_start
        note[2]=min(note[2],stop-n[1])//60*60
        if note[2]>=60: result.append(note)
    return result,target_start-prefix_start


def role_state(source, projected, target_start, role):
    stop=target_start+TARGET
    channel=ROLES.index(role)
    own=[n for n in projected if n[3]==channel]
    inside=[n for n in own if target_start<=n[1]<stop]
    incoming=[n for n in own if n[1]<target_start<n[1]+n[2]]
    if inside:
        return ('present_sparse' if len(inside)<8 else 'present_active'),True
    if incoming: return 'present_sustaining',True
    raw_activity=any(overlap(n,target_start,stop) for n in source)
    adjacent=any(overlap(n,max(0,target_start-CAP),target_start) or overlap(n,stop,stop+CAP) for n in source)
    if not raw_activity and adjacent: return 'present_resting',True
    if raw_activity: return 'missing',False  # projection loss is not a genuine rest label
    return 'missing',False


def validate_container(notes, loss_start):
    import note_trio as nt
    import offline_ensemble as oe
    total_ticks=loss_start+TARGET
    brief=nt.Brief(total_ticks//1920)
    ledger=[]
    with oe.limits():
        for beat in range(total_ticks//480):
            actions={role:[n for n in notes if n[3]==i and n[1]//480==beat] for i,role in enumerate(ROLES)}
            ledger=nt.commit(ledger,actions,beat,brief)
    if ledger!=sorted(notes,key=lambda n:(n[1],n[3],n[4])): raise ValueError('Commit changed sliced notes')
    if any(n[2]>CAP for n in notes): raise ValueError('Duration exceeds sampler support')


def audit_window(window):
    target_start=window['start_tick'];prefix_start=max(0,target_start-CAP)
    parts={};projected=[];role_results={};long=[]
    for role,assignment in window['assignment'].items():
        source,full=full_projected_part(window['song'],role,assignment['source_track'],assignment['source_channel'],assignment['source_program'],window['midi_sha256'],window.get('source_cache_root'))
        parts[role]=source;projected.extend(full)
        lo,hi=a.RANGES[role]
        unsupported=[dict(role=role,source_note_id=n['source_note_id'],onset_tick=n['start'],duration_beats=n['duration']/480,pitch=n['pitch'])
                     for n in source if lo<=n['pitch']<=hi and n['duration']>CAP and overlap(n,prefix_start,target_start+TARGET)]
        long.extend(unsupported)
        state,mask=role_state(source,full,target_start,role)
        relevant=[dict(start=n[1]-target_start,duration=min(n[2],target_start+TARGET-n[1]),
                       pitch=n[4],velocity=n[5],program=assignment['source_program'])
                  for n in full if n[3]==ROLES.index(role) and n[1]<target_start+TARGET and n[1]+n[2]>target_start]
        provenance,reasons,stats=a.qualify(role,relevant,0,TARGET) if relevant else (None,['no_sounding_target_notes'],None)
        if unsupported: reasons=[*reasons,'unsupported_source_duration_above_8_beats']
        if provenance and provenance!=assignment['provenance']: raise ValueError('Changed role provenance')
        role_results[role]=dict(state=state,loss_mask=bool(mask and not unsupported),
            gate_passed=not reasons,rejection_reasons=reasons,source_program=assignment['source_program'],
            provenance=assignment['provenance'],target_statistics=stats,
            incoming_supported_notes=sum(n[3]==ROLES.index(role) and n[1]<target_start<n[1]+n[2] for n in full) if not unsupported else 0,
            genuine_rest=state=='present_resting' and not unsupported)
    projected.sort(key=lambda n:(n[1],n[3],n[4]))
    sliced,loss_start=slice_supported(projected,target_start)
    validate_container(sliced,loss_start)
    strings,cello=role_results['strings'],role_results['cello']
    if 'transfer' in (strings['provenance'],cello['provenance']):
        s,c=strings['target_statistics'],cello['target_statistics']
        register_ok=s is not None and c is not None and s['median_pitch']>c['median_pitch']
    else: register_ok=True
    passed=bool(not long and register_ok and all(r['gate_passed'] for r in role_results.values()))
    audible=[copy.deepcopy(n) for n in sliced if n[1]<loss_start+TARGET and n[1]+n[2]>loss_start]
    for n in audible: n[1]-=loss_start
    identity=hashlib.sha256((window['song']+':'+str(target_start)).encode()).hexdigest()[:16]
    return dict(id=identity,song=window['song'],group=window['group'],split=window['split'],category=window['category'],
        start_tick=target_start,assignment=window['assignment'],roles=role_results,
        unsupported_source_notes=long,register_constraint_passed=bool(register_ok),eligible=passed,
        prefix_ticks=loss_start,target_ticks=TARGET,
        loss_beat_mask=[beat*480>=loss_start for beat in range((loss_start+TARGET)//480)],
        # Ineligible examples retain diagnostics, not a capped pseudo-target.
        notes=sliced if passed else None,
        exact_fingerprint=p.fingerprint(audible),transposition_fingerprint=p.fingerprint(audible,True))


def group_gate(rows, minimum):
    counts={role:len({r['group'] for r in rows if r['split']=='validation' and r['eligible'] and r['roles'][role]['gate_passed']}) for role in ROLES}
    return dict(required_per_role=minimum,validation_groups_per_role=counts,
                passed=all(count>=minimum for count in counts.values()))


def summarize(rows,directory):
    kept,dedup=p.deduplicate(rows,'eligible')
    a.write(directory/'deduplication.json',dedup)
    a.write(directory/'eligible_ids.json',[r['id'] for r in kept])
    counts={}
    for split in ('train','validation','test'):
        original=[r for r in rows if r['split']==split]
        selected=[r for r in kept if r['split']==split]
        counts[split]=dict(input_windows=len(original),eligible_before_dedup=sum(r['eligible'] for r in original),
            windows_after_dedup=len(selected),groups_after_dedup=len({r['group'] for r in selected}),
            exact_windows_after_dedup=sum(r['category']=='exact_full_trio' for r in selected),
            transferred_windows_after_dedup=sum(r['category']=='transferred_full_trio' for r in selected),
            windows_excluded_for_long_sustains=sum(bool(r['unsupported_source_notes']) for r in original),
            role_states=dict(Counter(v['state'] for r in selected for v in r['roles'].values())))
    gates={trainer:group_gate(kept,minimum) for trainer,minimum in PROTOCOL['validation_requirements'].items()}
    eligibility_by_id={r['id']:r['eligible'] for r in rows}
    prior=p.read_json(p.OUT/'windows.json')
    restored=[dict(id=r['id'],song=r['song'],split=r['split'],group=r['group']) for r in prior if not r['passes_without_legacy_threshold'] and eligibility_by_id[r['id']]]
    lost_case=[r for r in rows if r['group']=='slakh:Track01757']
    a.write(directory/'Track01757_diagnosis.json',lost_case)
    return dict(input_windows=len(rows),splits=counts,gates=gates,restored_windows=restored,
        restored_groups={s:len({r['group'] for r in restored if r['split']==s}) for s in counts},
        lost_case_supported=False if any(r['unsupported_source_notes'] for r in lost_case) else None,
        source_corpus_unchanged=True,training_started=False,training_loader_integrated=False,
        supervised_training_allowed=False,reason='Data gates must pass and the new context/masking schema must be integrated and verified before training.')


def require_supervised_gate(directory,trainer='train_offline_ensemble'):
    """Reusable preflight; no weak count-only shortcut to training."""
    directory=Path(directory)
    manifest=p.read_json(directory/'summary.json')
    if trainer not in manifest['gates']: raise ValueError('Unknown trainer gate')
    if not manifest['gates'][trainer]['passed']:
        raise RuntimeError(f'{trainer}: insufficient independent validation compositions: '+str(manifest['gates'][trainer]['validation_groups_per_role']))
    if not manifest['training_loader_integrated']:
        raise RuntimeError('Supported-prefix/role-mask training loader is not integrated or verified')
    return manifest


def write_report(summary,directory):
    rows=['# Supported-boundary validation gate','','No training or changes to existing corpus, splits, model weights or reward gates.',
          'Unsupported source notes longer than eight beats are excluded; supported target-crossing notes remain at their original onset in the context prefix.',
          'The prefix has no target loss; no eight-note minimum or fabricated boundary attack is introduced.','',
          '| Split | Input windows | Eligible before dedup | Windows / groups after dedup | Excluded for long notes |',
          '|---|---:|---:|---:|---:|']
    for split,c in summary['splits'].items():
        rows.append(f'| {split} | {c["input_windows"]} | {c["eligible_before_dedup"]} | {c["windows_after_dedup"]} / {c["groups_after_dedup"]} | {c["windows_excluded_for_long_sustains"]} |')
    rows+=['','## Trainer gates','']
    for trainer,g in summary['gates'].items():
        rows.append(f'- {trainer}: requires {g["required_per_role"]} validation groups per role; observed {g["validation_groups_per_role"]}; passed={g["passed"]}.')
    rows+=['','Track01757 still contains 22-beat source notes and is excluded. Supported-prefix handling does not extend the eight-beat action space.',
           f'Previously failing windows restored by the changed boundary handling: {len(summary["restored_windows"])}. Restored groups by split: {summary["restored_groups"]}.',
           'Restoration counts refer to previously failing windows, not necessarily new unique composition groups.',
           '', '**Supervised training allowed: False.** The existing trainers do not yet consume this prefix/masking schema. No automatic run is launched.',
           'The ten-group requirement belongs to the offline trainer; the repair trainer has a stricter twelve-group requirement.',
           'Counts are for this fixed trio-only pool, not the untouched broader instrument-pair corpus. No test work is moved into validation.',
           'Adding further eligible works requires a separately versioned expansion, original source/licence provenance and deduplication against existing splits.']
    (directory/'REPORT.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();directory=args.output.resolve();directory.mkdir(parents=True,exist_ok=True)
    path=directory/'protocol.json'
    if path.exists() and p.read_json(path)!=PROTOCOL: raise ValueError('Protocol change requires new version')
    if not path.exists():a.write(path,PROTOCOL)
    before=a.digest(ROOT/'assets/composition_v1/corpus.json.gz')
    rows=[];windows=p.selected_windows()
    for index,window in enumerate(windows):
        rows.append(audit_window(window))
        if (index+1)%50==0:print(f'Supported-boundary audited {index+1}/{len(windows)} windows; training disabled',flush=True)
    a.write(directory/'windows.json',rows)
    summary=summarize(rows,directory)
    summary['source_corpus_unchanged']=before==a.digest(ROOT/'assets/composition_v1/corpus.json.gz')
    if not summary['source_corpus_unchanged']:raise ValueError('Original corpus changed')
    summary['input_audit_sha256']=a.digest(a.OUT/'windows.jsonl.gz')
    summary['protocol_sha256']=a.digest(path)
    a.write(directory/'summary.json',summary)
    write_report(summary,directory)
    print((directory/'REPORT.md').read_text(encoding='utf-8'),flush=True)


if __name__=='__main__':main()
