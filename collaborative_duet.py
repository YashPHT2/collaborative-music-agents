"""Two role-conditioned inference agents composing a piano duet from shared history.

Both roles use frozen SkyTNT weights. There are no supplied melody notes, joint
RL updates, or claims of novelty/quality. The melody plans a bar from past music;
accompaniment responds to that bar's melody one onset at a time. In the following
bar the melody can hear the accompaniment that was actually committed.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import time

from accompany import EventSampler, MIDIModel, ROOT, TPB, resolve_overlaps, torch, DynamicCache, write_result
from evaluate_baseline import validate_audio

BAR = TPB*4
ROLES = {
    'melody': dict(channel=0,track=1,low=67,high=84,velocity_low=64,velocity_high=100,
                   max_duration=32,cap=16,monophonic=True),
    'accompaniment': dict(channel=1,track=2,low=36,high=66,velocity_low=40,velocity_high=88,
                          max_duration=64,cap=32,monophonic=False),
}


def signature(notes):
    return sorted(tuple(int(v) for v in note[1:]) for note in notes)


def history_hash(notes):
    return hashlib.sha256(json.dumps(signature(notes)).encode()).hexdigest()


def visible_history(ledger,role,bar_start,shared=True):
    channel=ROLES[role]['channel']
    return sorted((copy.deepcopy(n) for n in ledger if n[1]<bar_start and (shared or n[3]==channel)),
                  key=lambda n:(n[1],n[3],n[4]))


def role_seed(seed,role,bar):
    return seed+1009*bar+(0 if role=='melody' else 100003)


class RoleSampler(EventSampler):
    def __init__(self,model,role,seed,bpm,temperature,history):
        super().__init__(model,seed,bpm,temperature)
        self.role=role
        self.settings=ROLES[role]
        self.observed=copy.deepcopy(history)
        for note in history:
            self.append_note(note,ROLES['melody' if note[3]==0 else 'accompaniment']['track'])

    @torch.inference_mode()
    def sample_between(self,earliest,latest):
        """Sample a note in [earliest,latest), or WAIT, without time travel."""
        earliest=max(earliest,self.last_tick)
        if earliest>=latest:
            return None
        settings=self.settings
        x=torch.tensor([self.rows[self.cached_length:]],dtype=torch.long,device=self.model.device)
        hidden=self.model(x,cache=self.cache)[:,-1]
        self.cached_length=len(self.rows)
        row=None
        token_cache=DynamicCache()
        parameters=self.tok.events['note']
        for index in range(len(parameters)+1):
            if index==0:
                allowed=[self.tok.event_ids['note'],self.tok.eos_id]
            else:
                name=parameters[index-1]
                allowed=self.tok.parameter_ids[name]
                if name=='track':
                    allowed=[allowed[settings['track']]]
                elif name=='channel':
                    allowed=[allowed[settings['channel']]]
                elif name=='pitch':
                    allowed=allowed[settings['low']:settings['high']+1]
                elif name=='velocity':
                    allowed=allowed[settings['velocity_low']:settings['velocity_high']+1]
                elif name=='time1':
                    minimum=earliest//TPB-self.last_tick//TPB
                    maximum=(latest-30)//TPB-self.last_tick//TPB
                    allowed=allowed[max(0,minimum):min(len(allowed)-1,maximum)+1]
                elif name=='time2':
                    delta=row[0,1+parameters.index('time1')].item()-self.tok.parameter_ids['time1'][0]
                    beat=(self.last_tick//TPB+delta)*TPB
                    minimum=max(0,(earliest-beat+29)//30)
                    maximum=min(15,(latest-30-beat)//30)
                    allowed=allowed[minimum:maximum+1]
                elif name=='duration':
                    delta=row[0,1+parameters.index('time1')].item()-self.tok.parameter_ids['time1'][0]
                    substep=row[0,1+parameters.index('time2')].item()-self.tok.parameter_ids['time2'][0]
                    onset=(self.last_tick//TPB+delta)*TPB+substep*30
                    # Notes may cross incoming melody onsets, but not the bar.
                    bar_end=((earliest//BAR)+1)*BAR
                    maximum=min(settings['max_duration'],(bar_end-onset)//30)
                    allowed=allowed[1:maximum+1]
            if not allowed:
                raise RuntimeError(f'Empty {self.role} token mask at index {index}')
            logits=self.model.forward_token(hidden if index==0 else None,
                                            None if row is None else row[:,-1:],cache=token_cache)[:,-1]
            mask=torch.full_like(logits,float('-inf'))
            mask[:,allowed]=0
            probabilities=torch.softmax((logits+mask)/self.temperature,dim=-1)
            token=self.model.sample_top_p_k(probabilities,0.98,20,generator=self.generator).reshape(1,1)
            row=token if row is None else torch.cat([row,token],dim=1)
            if index==0 and token.item()==self.tok.eos_id:
                return None
        event=self.tok.tokens2event(row[0].tolist())
        _,delta,substep,_,channel,pitch,velocity,duration=event
        tick=(self.last_tick//TPB+delta)*TPB+substep*30
        assert earliest<=tick<latest
        return ['note',tick,duration*30,channel,pitch,velocity]

    def commit(self,note):
        self.append_note(note,self.settings['track'])


def compose_bar(model,role,history,start,end,seed,bpm,temperature,incoming=None):
    sampler=RoleSampler(model,role,seed,bpm,temperature,history)
    settings=ROLES[role]
    notes=[]
    audit=dict(role=role,start_tick=start,end_tick=end,history_events=len(history),
               partner_history_events=sum(n[3]!=settings['channel'] for n in history),
               history_hash=history_hash(history),seed=seed,incoming_events=0,
               incoming_policy='observe each melody onset before filling the next interval',
               eos_waits=0,density_cap_reached=False)
    cursor=start
    # Empty incoming means free generation. Otherwise each note opens a new
    # observation window. A pre-onset rest window exposes no future pitch.
    incoming=sorted(incoming or [],key=lambda n:(n[1],n[4]))
    windows=[]
    if incoming:
        if incoming[0][1]>start:
            windows.append((start,incoming[0][1],None))
        for i,note in enumerate(incoming):
            windows.append((note[1],incoming[i+1][1] if i+1<len(incoming) else end,note))
    else:
        windows=[(start,end,None)]
    for left,right,observation in windows:
        if observation is not None:
            sampler.append_note(observation,1)
            audit['incoming_events']+=1
        cursor=max(cursor,left)
        while len(notes)<settings['cap'] and cursor<right:
            candidate=sampler.sample_between(cursor,right)
            if candidate is None:
                audit['eos_waits']+=1
                break
            sampler.commit(candidate)
            notes.append(candidate)
            cursor=candidate[1]+candidate[2] if settings['monophonic'] else candidate[1]
        if len(notes)>=settings['cap']:
            audit['density_cap_reached']=True
            break
    clean=resolve_overlaps(notes)
    audit.update(raw_events=len(notes),exported_events=len(clean),output_hash=history_hash(clean))
    return clean,audit


def compose_duet(model,seed,bars,bpm,temperature,shared=True):
    ledger=[]
    trace=[]
    for bar in range(bars):
        start,end=bar*BAR,(bar+1)*BAR
        history=visible_history(ledger,'melody',start,shared)
        lead,lead_audit=compose_bar(model,'melody',history,start,end,role_seed(seed,'melody',bar),bpm,temperature)
        history=visible_history(ledger,'accompaniment',start,shared)
        accompaniment,accomp_audit=compose_bar(model,'accompaniment',history,start,end,
                                              role_seed(seed,'accompaniment',bar),bpm,temperature,
                                              incoming=lead if shared else None)
        # Both roles observe exactly the exported notes in subsequent bars.
        ledger.extend(lead+accompaniment)
        trace.extend([lead_audit,accomp_audit])
        print(f"{'Shared' if shared else 'Independent'} bar {bar+1}/{bars}: melody {len(lead)}, accompaniment {len(accompaniment)}",flush=True)
    return sorted(ledger,key=lambda n:(n[1],n[3],n[4])),trace


def score(ledger,bpm,channel=None):
    conductor=[['track_name',0,b'Conductor'],['set_tempo',0,int(60_000_000/bpm)],
               ['time_signature',0,4,2,24,8],['key_signature',0,0,0]]
    tracks=[conductor]
    for name,settings in ROLES.items():
        if channel is not None and settings['channel']!=channel:
            continue
        tracks.append([['track_name',0,('AI '+name).encode()],['patch_change',0,settings['channel'],0]]+
                      [copy.deepcopy(n) for n in ledger if n[3]==settings['channel']])
    return [TPB,*tracks]


def validate_export(path,ledger,end,bpm):
    import mido
    midi=mido.MidiFile(path)
    tick=0
    active={}
    decoded=[]
    for event in mido.merge_tracks(midi.tracks):
        tick+=event.time
        if event.type=='note_on' and event.velocity>0:
            key=(event.channel,event.note)
            assert key not in active,f'Same-pitch overlap: {key}'
            active[key]=(tick,event.velocity)
        elif event.type=='note_off' or (event.type=='note_on' and event.velocity==0):
            start,velocity=active.pop((event.channel,event.note))
            assert tick>start
            decoded.append(['note',start,tick-start,event.channel,event.note,velocity])
    assert not active
    assert signature(decoded)==signature(ledger)
    for note in ledger:
        settings=ROLES['melody' if note[3]==0 else 'accompaniment']
        assert settings['low']<=note[4]<=settings['high']
        assert 0<=note[1]<note[1]+note[2]<=end
    lead=sorted([n for n in ledger if n[3]==0],key=lambda n:n[1])
    assert all(a[1]+a[2]<=b[1] for a,b in zip(lead,lead[1:])), 'Melody is not monophonic'
    return dict(note_events_preserved=True,balanced_note_events=True,note_count=len(ledger),
                duration_seconds=round(midi.length,3),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def self_repetition(ledger):
    result={}
    for role,settings in ROLES.items():
        pitches=[n[4] for n in sorted(ledger,key=lambda n:n[1]) if n[3]==settings['channel']]
        grams=Counter(tuple(pitches[i:i+4]) for i in range(max(0,len(pitches)-3)))
        result[role]=dict(notes=len(pitches),distinct_pitches=len(set(pitches)),
                          most_frequent_four_pitch_pattern=max(grams.values(),default=0))
    return result


def sensitivity_probe(model,ledger,trace,seed,bars,bpm,temperature):
    # Counterfactual: replay an existing melody decision with identical RNG,
    # while transposing only the immediately preceding accompaniment bar.
    for bar in range(min(bars-1,4),0,-1):
        start,end=bar*BAR,(bar+1)*BAR
        history=visible_history(ledger,'melody',start,True)
        edited=copy.deepcopy(history)
        changed=0
        for note in edited:
            if note[3]==1 and start-BAR<=note[1]<start:
                note[4]+=2
                changed+=1
        if not changed:
            continue
        original,a=compose_bar(model,'melody',history,start,end,role_seed(seed,'melody',bar),bpm,temperature)
        actual=[n for n in ledger if n[3]==0 and start<=n[1]<end]
        assert signature(original)==signature(actual),'Replay is not reproducible'
        altered,b=compose_bar(model,'melody',edited,start,end,role_seed(seed,'melody',bar),bpm,temperature)
        return dict(bar=bar+1,intervention='Transpose preceding accompaniment bar +2 semitones in observation only',
                    changed_partner_events=changed,same_seed=True,replay_matches_original=True,
                    contexts_differ=a['history_hash']!=b['history_hash'],
                    output_changed=signature(original)!=signature(altered),original_notes=original,
                    counterfactual_notes=altered,
                    interpretation='Context-sensitivity probe, not evidence of improved musical coordination.')
    return dict(performed=False,reason='No preceding accompaniment notes were available')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=2027)
    parser.add_argument('--bars',type=int,default=8)
    parser.add_argument('--bpm',type=int,default=100)
    parser.add_argument('--temperature',type=float,default=0.9)
    parser.add_argument('--no-audio',action='store_true')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    if not 2<=args.bars<=16 or not 30<=args.bpm<=240 or args.temperature<=0 or args.seed<0:
        parser.error('Use 2..16 bars, 30..240 BPM, positive temperature, and a nonnegative seed.')
    device=args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    model=MIDIModel.from_pretrained(ROOT/'models/skytnt-tv2o-medium',dtype=dtype,local_files_only=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    root=ROOT/'outputs'/'collaborative_duet'/f'seed{args.seed}'
    root.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    joined,shared_trace=compose_duet(model,args.seed,args.bars,args.bpm,args.temperature,True)
    independent,independent_trace=compose_duet(model,args.seed,args.bars,args.bpm,args.temperature,False)
    for role,settings in ROLES.items():
        if not any(n[3]==settings['channel'] for n in joined):
            raise RuntimeError(f'The {role} agent produced no notes; output retained for debugging only.')
    checks={}
    for name,notes,channel in [('shared_duet',joined,None),('independent_duet',independent,None),
                               ('melody_only',joined,0),('accompaniment_only',joined,1)]:
        path=write_result(root,name,score(notes,args.bpm,channel),not args.no_audio)
        exported=notes if channel is None else [n for n in notes if n[3]==channel]
        checks[name]=validate_export(path,exported,args.bars*BAR,args.bpm)
        if not args.no_audio:
            checks[name]['audio']=validate_audio(path.with_suffix('.wav'))
    probe=sensitivity_probe(model,joined,shared_trace,args.seed,args.bars,args.bpm,args.temperature)
    manifest=dict(seed=args.seed,bars=args.bars,bpm=args.bpm,temperature=args.temperature,
                  model='skytnt/midi-model-tv2o-medium',revision='0f8f265d4330f4e46527ac2313200254c5757f5f',
                  model_source_revision='f504d5cb58f769ab0f2909c679238f6621034573',
                  device=device,dtype=str(dtype),supplied_melody=False,trained=False,joint_rl=False,
                  independent_policy_weights=False,roles=ROLES,
                  architecture='Two role-conditioned inference agents with independent per-bar RNG, shared frozen backbone, and fresh per-role KV caches.',
                  shared_context='Melody sees both exported parts before current bar. Accompaniment sees both prior parts and each current melody onset causally. Current-bar accompaniment reaches melody next bar.',
                  independent_control='Each agent sees only its own prior bars; no current or previous partner notes. Same role masks, seed derivation and bar boundaries.',
                  schedule_caveat='Bar planning precedes accompaniment. Scheduler knows next melody onset time, but not its pitch. This is scheduled composition, not unconstrained live simultaneous improvisation.',
                  self_repetition=self_repetition(joined),
                  shared_vs_independent_notes_differ=signature(joined)!=signature(independent),
                  elapsed_seconds=round(time.perf_counter()-started,3),originality_verified=False,
                  musical_quality_improvement_established=False)
    for name,data in [('manifest.json',manifest),('shared_events.json',joined),('independent_events.json',independent),
                      ('context_trace.json',dict(shared=shared_trace,independent=independent_trace)),
                      ('sensitivity_probe.json',probe),('verification.json',checks)]:
        (root/name).write_text(json.dumps(data,indent=2)+'\n')
    report=['# First shared-context two-agent duet','',
            'Both melody and accompaniment are sampled from pretrained weights. No scripted melody is supplied. '
            'The two roles have separate random streams, register/track/channel masks, and context reconstruction. '
            'They share one frozen backbone; they are not independently trained musicians or a jointly trained MARL system.',
            '',f"Seed {args.seed}, {args.bars} bars, {args.bpm} BPM. Melody notes: {manifest['self_repetition']['melody']['notes']}; accompaniment notes: {manifest['self_repetition']['accompaniment']['notes']}.",
            '', '## Interaction','',manifest['shared_context'], '',manifest['schedule_caveat'],
            '', 'At each bar, the melody agent proposes a monophonic phrase. The accompaniment fills intervals '
            'between that phrase’s observed onsets. Both exported parts enter the next bar’s shared ledger. '
            'Input notes are sorted chronologically; no current-bar partner notes are shown early to the melody. '
            'When an agent emits EOS it waits; bar boundaries, register masks and note caps are explicit constraints. '
            'Repeated same-pitch overlaps are cleaned before the shared ledger is updated.',
            '', '## Controls','',manifest['independent_control'],
            '',f"Shared vs independent output differs: {manifest['shared_vs_independent_notes_differ']}.",
            '',f"Counterfactual melody response changed: {probe.get('output_changed','probe unavailable')}. "
            'See sensitivity_probe.json. A response to changed context does not imply a good response.',
            '', '## Files','',
            '- shared_duet: both agents with shared context.',
            '- independent_duet: partner observations removed.',
            '- melody_only / accompaniment_only: exact isolated parts from the shared performance.',
            '- context_trace.json: role histories, partner counts, input/output hashes and seeds.',
            '- shared_events.json / independent_events.json: absolute MIDI-note events.',
            '- verification.json: MIDI/audio checks; separate channels preserve role identity.',
            '', '## Limits','',
            'This demonstrates the composition loop and context exchange. It does not demonstrate learned cooperation, '
            'an advantage over a centralized generator, reliable music quality, or originality. No comparison against '
            'the full training corpus was performed. Local repetition statistics are not plagiarism checks. '
            'The independent control also removes current melody observations from accompaniment, so its comparison '
            'does not isolate bidirectional feedback alone. More seeds, a one-way conditioning control, and a '
            'centralized generator comparison are required before claiming a coordination benefit.',
            '', 'Next training step: expose role-specific trainable policies/adapters in this environment, '
            'initially freeze the melody policy, and evaluate accompaniment training before joint updates. '
            'Keep the previous PPO pilot separate; it has not been promoted into these agents.']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(root),notes=manifest['self_repetition'],
                          shared_vs_independent_notes_differ=manifest['shared_vs_independent_notes_differ'],
                          sensitivity_probe_changed=probe.get('output_changed')),indent=2),flush=True)


if __name__=='__main__':
    main()
