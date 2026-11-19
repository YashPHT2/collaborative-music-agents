"""Phrase-conditioned learned musicians with explicit arrangement scaffolding."""
import copy,csv,hashlib,io,json,math,subprocess,time,zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import mido,numpy as np,torch
from band import (ROOT,ROLES,Brief,BandSampler,TeamCritic,state_features,load_model,
                  ensemble_reward,update,fingerprint,export_performance)
from band_reward import ensemble_reward_v2
from train_band import make_policies,save_checkpoint

OUT=ROOT/'outputs/adaptive_band_v2'
CHECKPOINT=ROOT/'outputs/band_reward_v2/latest.pt'
DRUM_PITCHES=[36,38,42,46,45,49]

def fit_drum_prior():
    """Fit categorical pulse/stroke statistics, not a Transformer, on GMD train only."""
    archive=ROOT/'assets/groove-v1.0.0-midionly.zip'
    expected='651cbc524ffb891be1a3e46d89dc82a1cecb09a57c748c7b45b844c4841dcc1e'
    if not archive.exists():
        import urllib.request
        urllib.request.urlretrieve('https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip',archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
    counts=np.ones((16,6),dtype=float); velocities=[]; files=0; hits=0
    mapping={36:36,38:38,40:38,37:38,42:42,22:42,44:42,46:46,26:46,
             45:45,47:45,43:45,48:45,50:45,58:45,49:49,55:49,57:49,52:49}
    with zipfile.ZipFile(archive) as z:
        rows=csv.DictReader(io.StringIO(z.read('groove/info.csv').decode()))
        for row in rows:
            if row['split']!='train' or row['beat_type']!='beat' or row['time_signature']!='4-4': continue
            if row['style'].split('/')[0] not in ['pop','rock','soul']: continue
            midi=mido.MidiFile(file=io.BytesIO(z.read('groove/'+row['midi_filename'])))
            tick=0; files+=1
            for e in mido.merge_tracks(midi.tracks):
                tick+=e.time
                if e.type=='note_on' and e.velocity>0 and e.note in mapping:
                    slot=round(tick/midi.ticks_per_beat*4)%16
                    counts[slot,DRUM_PITCHES.index(mapping[e.note])]+=1
                    velocities.append(e.velocity); hits+=1
    assert files>0 and hits>100
    probs=counts/counts.sum(axis=1,keepdims=True)
    meta=dict(dataset='Groove MIDI Dataset v1.0.0',url='https://magenta.tensorflow.org/datasets/groove',
              license='CC BY 4.0',citation='Gillick et al., Learning to Groove with Inverse Sequence Transformations, ICML 2019',
              split='train only',styles=['pop','rock','soul'],files=files,hits=hits,
              method='Categorical stroke probabilities by sixteenth position; no whole performance is replayed.',
              archive_sha256=expected,probabilities=probs.tolist())
    (OUT/'drum_prior.json').write_text(json.dumps(meta,indent=2))
    return probs

def infer_intent(lead,tonic,minor=True):
    """An explicit musical parser broadcasts the leader's self-generated bar."""
    histogram=Counter()
    for n in lead: histogram[n[4]%12]+=n[2]
    options=[]
    for root in range(12):
        for third in [3,4]:
            pcs=[root,(root+third)%12,(root+7)%12]
            fit=sum(histogram[p] for p in pcs)/max(1,sum(histogram.values()))
            fit+=.12*(root==tonic)
            options.append((fit,root,third))
    _,root,third=max(options)
    return dict(root=root,third=third,chord=[root,(root+third)%12,(root+7)%12],
                tonic=tonic,minor=minor,activity=min(1,len(lead)/6),leader_events=len(lead),
                onsets=sorted({n[1]%1920 for n in lead}))

class PhraseSampler(BandSampler):
    def __init__(self,model,role,seed,brief,beat,history,prior,intent=None):
        super().__init__(model,role,seed,brief,beat,history)
        self.settings=copy.deepcopy(self.settings)
        self.settings['cap']={'piano':1,'bass':1,'drums':2}[role]
        self.settings['polyphony']={'piano':1,'bass':1,'drums':2}[role]
        self.prior=prior; self.intent=intent or dict(tonic=brief.tonic,activity=.35,chord=[brief.tonic],root=brief.tonic)
        self.exploration=.1  # Same mixture in training, generation, and counterfactual.

    def onsets(self):
        return [v for v in super().onsets() if v in [0,8]]

    def constrain_values(self,kind,values,decoded):
        if kind=='velocity':
            lo,hi={'piano':(56,84),'bass':(44,68),'drums':(28,56)}[self.role]
            return [v for v in range(lo,hi+1,4)]
        if kind=='duration':
            permitted={'piano':[8,16,24,32],'bass':[16,24,32],'drums':[4]}[self.role]
            # Phrase boundaries release the old harmony before the next announced bar.
            remaining=(4-self.beat%4)*16-decoded['time2']
            legal=[v for v in values if v<=remaining]
            return [v for v in legal if v in permitted] or [min(legal)]
        if kind=='pitch':
            if self.role=='piano':
                scale={(self.brief.tonic+p)%12 for p in [0,2,3,5,7,8,10]}
                allowed=[p for p in values if 67<=p<=84 and p%12 in scale]
                own=[n for n in getattr(self,'observed',[]) if n[3]==0]
                if not own:
                    chord={(self.brief.tonic+p)%12 for p in [0,3,7]}
                    allowed=[p for p in allowed if p%12 in chord]
                if self.beat>=self.brief.bars*4-2:
                    return [p for p in allowed if p%12==self.brief.tonic] or allowed
                return allowed
            if self.role=='bass':
                return [p for p in values if 33<=p<=48 and p%12 in self.intent['chord']]
        return values

    def adjust_logits(self,kind,values,logits,decoded):
        result=logits.clone()
        if kind=='event':
            # EOS is a meaningful beat-rest decision, not an end-of-song accident.
            result[1]=torch.logsumexp(result,dim=0)+math.log(.20 if self.role=='drums' else .08)
        if kind=='pitch' and self.role=='piano':
            # In this solo register the ensemble model can overconcentrate at the
            # lower boundary. Temper its confidence before applying the learned actor.
            result=(result-result.mean())/result.std(unbiased=False).clamp_min(.75)
            own=[n for n in self.observed if n[3]==0]
            last=own[-1][4] if own else 72
            result+=torch.tensor([-.12*abs(p-last)-(2. if p==last else 0) for p in values],device=result.device)
            if self.beat>=self.brief.bars*4-2:
                result+=torch.tensor([2.5 if p%12==self.brief.tonic else 0 for p in values],device=result.device)
        if kind=='pitch' and self.role=='bass':
            result+=torch.tensor([1.8*(p%12==self.intent['root'])-.04*abs(p-40) for p in values],device=result.device)
        if self.role=='drums' and kind=='pitch':
            slot=(self.beat%4)*4+decoded['time2']//4
            learned=np.log(self.prior[slot]+1e-6)
            result+=torch.tensor([2*learned[DRUM_PITCHES.index(p)] for p in values],device=result.device)
            if self.intent['activity']<.6:
                result+=torch.tensor([-2. if p in [38,45,46,49] else 0 for p in values],device=result.device)
        return result

def compose_bar(model,actors,critic,brief,bar,ledger,prior,seed,lead_override=None,old_intent=None):
    start=bar*1920; lead=[]; steps=[]
    # Leader proposes a bar; this becomes an explicitly announced plan.
    for beat in range(bar*4,bar*4+4):
        state=state_features(brief,beat,ledger+lead)
        with torch.no_grad(): value=float(critic(state))
        history=sorted(ledger+lead,key=lambda n:(n[1],n[3],n[4]))
        sampler=PhraseSampler(model,'piano',seed+beat*1009,brief,beat,history,prior)
        if lead_override is None: notes,records=sampler.propose(actors['piano'],state)
        else: notes,records=[copy.deepcopy(n) for n in lead_override if n[1]//480==beat],[]
        lead.extend(notes); steps.append(dict(state=state,value=value,reward=0.,records={'piano':records},components={}))
    intent=infer_intent(lead,brief.tonic)
    message=old_intent or intent
    incoming=copy.deepcopy(ledger)
    for note in lead: incoming.append(note)
    generated=[]
    for local,beat in enumerate(range(bar*4,bar*4+4)):
        past=[n for n in incoming+generated if n[1]<beat*480]
        state=state_features(brief,beat,past)
        # Exact announced notes are musical context for same-bar accompaniment.
        current=[n for n in lead if n[1]//480==beat]
        if old_intent is not None:
            # Control uses the prior tonal/activity brief, with the same lead rendered.
            current=[]
            past=[n for n in past if n[3]!=0 or n[1]<6*1920]
        for role in ['bass','drums']:
            rb=copy.deepcopy(brief); rb.tonic=message['tonic']
            observed=sorted(past+current,key=lambda n:(n[1],n[3],n[4]))
            follower_state=state_features(rb,beat,past)
            # Simultaneous follower proposals see the same committed past.
            sampler=PhraseSampler(model,role,seed+beat*1009+list(ROLES).index(role)*100003,rb,beat,observed,prior,message)
            notes,records=sampler.propose(actors[role],follower_state)
            generated.extend(notes); steps[local]['records'][role]=records
    notes=sorted(lead+generated,key=lambda n:(n[1],n[3],n[4]))
    reward,parts=ensemble_reward_v2(ledger+notes,start,start+1920,brief)
    steps[-1]['reward']=reward; steps[-1]['components']=parts
    return notes,steps,intent

def compose(model,actors,critic,prior,seed=813,training=False,change=True,control=False,leader_notes=None):
    bars=4 if training else 12
    ledger=[]; steps=[]; trace=[]; themes=[]; last_intent=None
    for bar in range(bars):
        tonic=9+(2 if change and bar>=(2 if training else 6) else 0)
        brief=Brief(bars,100,tonic,True,'band-heuristic-v2')
        override=None
        # Exact reprises of a generated two-bar motif are an arranged scaffold.
        if not training and bar in [2,3,6,7,8,9]:
            source=themes[bar%2]
            shift=2 if bar>=6 else 0
            override=[['note',bar*1920+n[1],n[2],0,n[4]+shift,n[5]+(4 if bar>=6 else 0)] for n in source]
            if bar>=6:
                developed=[]
                for i,n in enumerate(override):
                    if i%2==0 and n[2]>=480:
                        half=(n[2]//240)*120
                        developed.extend([['note',n[1],half,*n[3:]],['note',n[1]+half,n[2]-half,*n[3:]]])
                    else: developed.append(n)
                override=developed
        if leader_notes is not None:
            override=[copy.deepcopy(n) for n in leader_notes if n[1]//1920==bar]
        notes,ss,intent=compose_bar(model,actors,critic,brief,bar,ledger,prior,seed,override,
                                  last_intent if control and bar>=6 else None)
        if bar<2:
            themes.append([['note',n[1]-bar*1920,n[2],*n[3:]] for n in notes if n[3]==0])
        if bar==5: last_intent=copy.deepcopy(intent)
        ledger.extend(notes); ledger.sort(key=lambda n:(n[1],n[3],n[4])); steps.extend(ss)
        trace.append(dict(bar=bar+1,announced_intent=intent,followers_intent=last_intent if control and bar>=6 else intent,
                          reprise=override is not None,role_notes={r:sum(n[3]==s['channel'] for n in notes) for r,s in ROLES.items()}))
    return dict(notes=ledger,steps=steps,trace=trace,brief=asdict(Brief(bars,100,9,True,'band-heuristic-v2')),seed=seed)

def render_exact(root,name,episode):
    checks=export_performance(root,name,episode)
    from evaluate_baseline import validate_audio
    for stem in ['mix',*ROLES]:
        wav=root/f'{name}_{stem}.wav'; temp=root/f'{name}_{stem}_30.wav'
        filters='apad,atrim=duration=30,afade=t=out:st=29.5:d=0.5'
        if stem=='mix': filters+=',loudnorm=I=-18:TP=-1.5:LRA=11'
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(wav),'-af',
            filters,'-ar','44100','-c:a','pcm_s16le',str(temp)],check=True,capture_output=True)
        temp.replace(wav)
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(wav),'-q:a','2',str(wav.with_suffix('.mp3'))],check=True,capture_output=True)
        checks[stem]['audio']=validate_audio(wav)
    return checks

def main():
    OUT.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); torch.manual_seed(20261021)
    prior=fit_drum_prior(); payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    actors=make_policies(); critic=TeamCritic()
    for r,a in actors.items(): a.load_state_dict(payload['actors'][r])
    opts={r:torch.optim.Adam(a.parameters(),lr=1e-4) for r,a in actors.items()}; co=torch.optim.Adam(critic.parameters(),lr=3e-4)
    model=load_model(); before={r:fingerprint(a) for r,a in actors.items()}; base=fingerprint(model); logs=[]
    completed=0
    if (OUT/'latest.pt').exists():
        saved=torch.load(OUT/'latest.pt',map_location='cpu',weights_only=False)
        for r,a in actors.items():
            a.load_state_dict(saved['actors'][r]); opts[r].load_state_dict(saved['optimizers'][r])
        critic.load_state_dict(saved['critic']); co.load_state_dict(saved['critic_optimizer'])
        logs=saved['logs']; completed=saved['completed']
        print(f'Resuming after {completed} phrase updates',flush=True)
    for u in range(completed,2):
        eps=[]
        for e in range(4):
            ep=compose(model,actors,critic,prior,seed=41000001+u*4+e,training=True)
            eps.append(ep); print(f'Phrase training {u+1}/2 episode {e+1}/4',flush=True)
        learning=update(actors,critic,opts,co,eps); logs.append(learning)
        save_checkpoint(OUT/'latest.pt',actors,critic,opts,co,payload['initial'],u+1,20261021,logs,0)
    performance=compose(model,actors,critic,prior)
    control=compose(model,actors,critic,prior,control=True,leader_notes=[n for n in performance['notes'] if n[3]==0])
    # Causal control: same seed and arrangement, identical pre-change performance.
    assert [n for n in performance['notes'] if n[1]<11520]==[n for n in control['notes'] if n[1]<11520]
    assert [n for n in performance['notes'] if n[3]==0]==[n for n in control['notes'] if n[3]==0]
    followers={r:[n for n in performance['notes'] if n[3]==s['channel'] and n[1]>=11520]!=[n for n in control['notes'] if n[3]==s['channel'] and n[1]>=11520] for r,s in ROLES.items() if r!='piano'}
    assert all(followers.values()),'A follower did not respond to the changed intent'
    checks=render_exact(OUT,'performance',performance)
    # Keep control MIDI/events for objective reproduction; render its full mix too.
    control_checks=render_exact(OUT,'held_intent_control',control)
    result=dict(music_seconds=28.8,audio_seconds=30,change_seconds=14.4,
        learned_actors_changed={r:before[r]!=fingerprint(a) for r,a in actors.items()},backbone_unchanged=base==fingerprint(model),
        same_prechange_music=True,same_piano_in_control=True,followers_changed_with_intent=followers,
        all_notes_generated_by='SkyTNT with separately trained residual actors and musical masks; selected piano phrases reprised by controller',
        arrangement_scaffold='Piano-generated two-bar motif reprises with arranged rhythmic development, scale/register masks, explicit leader intent parser and scheduled key change +2 semitones at bar 7.',
        training='8 additional four-bar joint episodes under new phrase/sparse-role constraints; GMD categorical drum prior fitted on train split.',
        independently_finetuned_transformers=False,learned_quality_evaluator=False,
        quality_requires_listening=True,export_checks=checks,control_checks=control_checks)
    assert all(result['learned_actors_changed'].values()) and result['backbone_unchanged']
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)); (OUT/'training_log.json').write_text(json.dumps(logs,indent=2))
    (OUT/'REPORT.md').write_text('# Adaptive band performance\n\n'
        '28.8 seconds of music plus release, trimmed to exactly 30 seconds. The piano changes key at 14.4 seconds. '
        'Its generated opening motif is reprised by an explicit arrangement controller. Bass and drums sample their own parts from learned policies conditioned on the piano’s announced bar intent.\n\n'
        'The parser, structure, scale masks, quieter velocity ranges and density budgets are engineered scaffolding. '
        'Observed differences from the held-intent control demonstrate conditioning on the intent, not learned superiority, emergent communication or a quality advantage. '
        'Piano and bass use pretrained musical knowledge plus residual RL updates; only the categorical drum prior is fitted on new human data. '
        'No learned quality evaluator has been trained from the handful of listening votes.\n\n'
        'Groove MIDI Dataset: Gillick, Roberts, Engel, Eck, Bamman (ICML 2019), CC BY 4.0. '
        'https://magenta.tensorflow.org/datasets/groove\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['export_checks','control_checks']},indent=2),flush=True)

if __name__=='__main__': main()
