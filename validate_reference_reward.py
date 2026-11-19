"""Predeclared held-out reference reward validation. No policy training.

Source excerpts are disjoint from profile/calibration excerpts. Perturbations
are assumed degradations for an automated stress test, NOT human taste labels.
"""
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter,sosfiltfilt,correlate,find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_music_references import FILES

ROOT=Path(__file__).resolve().parent/'outputs'/'reward_validation_v1'
SR=22050;HOP=512;SECONDS=12
SPLITS={
    'flowing_strings':dict(calibration=[0,32,64],heldout=[16,48,80]),
    'spacious_cinematic':dict(calibration=[8,56,104,152,200,248],heldout=[32,80,128,176,224,272])}
BROKEN=['phrase_shuffle','band_desync','pitch_clash','short_loop']
CONTROLS=['gain_minus_6db','mild_room','transpose_2_semitones']
GROUP_WEIGHTS=dict(rhythm=.35,tonal_motion=.30,development=.30,timbre=.05)
PROTOCOL=dict(version=1,seconds=SECONDS,sr=SR,splits=SPLITS,broken=BROKEN,controls=CONTROLS,
    group_weights=GROUP_WEIGHTS,minimum_overall_pairwise_win_rate=.80,
    minimum_win_rate_per_reference=.75,minimum_win_rate_per_reference_corruption=2/3,
    maximum_control_p90_absolute_score_change=.15,
    sanity='Original must score above silence and RMS-matched noise for every held-out excerpt',
    decision='Failures block use of this reward for RL. Automated pass still requires human verification and generated-MIDI renderer checks.',
    label_limit='Synthetically changed does not necessarily mean musically worse. Labels require listener review.',
    leakage_limit='Disjoint excerpts, but only two recordings: this is within-recording validation, not unseen-piece or listener-preference generalization.')


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(path):
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-vn','-ac','2','-ar',str(SR),'-f','f32le','pipe:1'])
    return np.frombuffer(raw,dtype='<f4').reshape(-1,2).copy()


def q(x):return np.quantile(x,[.1,.5,.9]).tolist()


def features(audio):
    y=audio.mean(axis=1).astype(np.float32)
    rms=float(np.sqrt(np.mean(y*y)))
    # Gain normalization is intentional; absolute mastering loudness is not taste.
    y=y/max(rms,1e-7)
    S=np.abs(librosa.stft(y,n_fft=2048,hop_length=HOP))
    power=S*S;times=librosa.frames_to_time(np.arange(S.shape[1]),sr=SR,hop_length=HOP)
    chroma=librosa.feature.chroma_stft(S=power,sr=SR,n_fft=2048,hop_length=HOP)
    chroma=gaussian_filter1d(chroma,2,axis=1)
    chroma/=np.maximum(np.linalg.norm(chroma,axis=0,keepdims=True),1e-9)
    onset=librosa.onset.onset_strength(S=librosa.power_to_db(power,ref=np.max),sr=SR,hop_length=HOP)
    onset=onset/(np.percentile(onset,95)+1e-9)
    peaks=librosa.onset.onset_detect(onset_envelope=onset,sr=SR,hop_length=HOP)
    onset_times=librosa.frames_to_time(peaks,sr=SR,hop_length=HOP);ioi=np.diff(onset_times)
    centered=onset-onset.mean();ac=correlate(centered,centered,mode='full',method='fft')[len(onset)-1:]
    ac/=max(float(ac[0]),1e-9)
    lags=[max(1,round(seconds*SR/HOP)) for seconds in [.25,.5,.75,1,1.5,2,3,4]]
    rhythm=[len(peaks)/SECONDS,float(np.std(ioi)/(np.mean(ioi)+1e-9)) if len(ioi) else 0.,
            *(q(ioi) if len(ioi) else [0.,0.,0.])]
    rhythm.extend(float(ac[min(len(ac)-1,lag)]) for lag in lags)
    change=np.linalg.norm(np.diff(chroma,axis=1),axis=0)
    mass=chroma/np.maximum(chroma.sum(axis=0,keepdims=True),1e-9)
    entropy=-(mass*np.log(mass+1e-9)).sum(axis=0)/np.log(12)
    tonal=[*q(change),*q(entropy)]
    # Transposition-invariant pitch-class recurrence, not exact melody copying.
    for lag in lags:
        lag=min(lag,chroma.shape[1]-1)
        tonal.append(float(np.mean(np.sum(chroma[:,lag:]*chroma[:,:-lag],axis=0))))
    frame_rms=librosa.feature.rms(S=S)[0];db=20*np.log10(frame_rms+1e-9)
    energy=np.array([np.median(db[(times>=i*SECONDS/6)&(times<(i+1)*SECONDS/6)]) for i in range(6)])
    energy=(energy-np.median(energy))/12
    activity=np.array([np.mean(onset[(times>=i*SECONDS/6)&(times<(i+1)*SECONDS/6)]) for i in range(6)])
    activity=activity-activity.mean()
    development=[*energy.tolist(),*activity.tolist(),float(np.std(db)/12)]
    half=chroma.shape[1]//2
    development.append(float(np.mean(np.sum(chroma[:,:half]*chroma[:,half:half*2],axis=0))))
    mel=librosa.feature.melspectrogram(S=power,sr=SR,n_mels=24)
    mel=mel.sum(axis=1);mel=mel/(mel.sum()+1e-9)
    timbre=np.sqrt(mel).tolist()
    # Static similarity baseline deliberately cannot describe phrase order.
    static=[*np.sort(chroma.mean(axis=1)).tolist(),*timbre]
    result={k:np.asarray(v,dtype=float) for k,v in dict(rhythm=rhythm,tonal_motion=tonal,development=development,timbre=timbre,static=static).items()}
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite features')
    return result,rms


def calibrate(rows):
    result={}
    for group in rows[0]:
        x=np.stack([r[group] for r in rows]);center=np.median(x,axis=0)
        scale=np.maximum((np.percentile(x,75,axis=0)-np.percentile(x,25,axis=0))/1.349,.10+.10*np.abs(center))
        result[group]=dict(center=center.tolist(),scale=scale.tolist())
    return result


def score(profile,feat,rms,structured=True):
    components={}
    for group,parameters in profile.items():
        z=(feat[group]-parameters['center'])/parameters['scale']
        components[group]=float(np.exp(-.5*np.mean(np.minimum(z*z,9))))
    value=sum(GROUP_WEIGHTS[k]*components[k] for k in GROUP_WEIGHTS) if structured else components['static']
    if rms<1e-6:value=0.
    return float(value),components


def smooth_edges(x,samples=220):
    x=x.copy();n=min(samples,len(x)//4)
    if n:
        envelope=np.sin(np.linspace(0,np.pi/2,n))[:,None]**2
        x[:n]*=envelope;x[-n:]*=envelope[::-1]
    return x


def perturb(audio,kind,seed):
    rng=np.random.default_rng(seed);n=len(audio)
    if kind=='original':return audio.copy()
    if kind=='gain_minus_6db':return audio*.501187
    if kind=='mild_room':
        result=audio.copy()
        for seconds,amplitude in [(.061,.10),(.113,.05)]:
            lag=int(seconds*SR);result[lag:]+=audio[:-lag]*amplitude
        return result/1.15
    if kind=='transpose_2_semitones':return librosa.effects.pitch_shift(audio.T,sr=SR,n_steps=2).T
    if kind=='phrase_shuffle':
        chunks=np.array_split(audio,4);order=[2,0,3,1]
        return np.concatenate([smooth_edges(chunks[i]) for i in order])
    if kind=='short_loop':
        chunk=smooth_edges(audio[:int(1.5*SR)])
        return np.tile(chunk,(int(np.ceil(n/len(chunk))),1))[:n]
    if kind=='band_desync':
        low=sosfiltfilt(butter(4,400,fs=SR,output='sos'),audio,axis=0);high=audio-low
        result=low.copy()
        # Frequency bands are not instrument stems. Delays vary between sections.
        for section,(first,last) in enumerate(zip(np.linspace(0,n,5,dtype=int)[:-1],np.linspace(0,n,5,dtype=int)[1:])):
            shift=int([.09,.23,.13,.31][section]*SR)
            shifted=np.roll(high[first:last],shift,axis=0)
            result[first:last]+=smooth_edges(shifted)
        return result
    if kind=='pitch_clash':
        shifted=librosa.effects.pitch_shift(audio.T,sr=SR,n_steps=6).T
        return .65*audio+.35*shifted
    if kind=='silence':return np.zeros_like(audio)
    if kind=='noise':
        noise=rng.standard_normal(audio.shape).astype(np.float32)
        return noise*(np.sqrt(np.mean(audio*audio))/(np.sqrt(np.mean(noise*noise))+1e-9))
    raise ValueError(kind)


def interval_overlap(a,b,length=SECONDS):return max(a,b)<min(a+length,b+length)


def main():
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'audio').mkdir(exist_ok=True)
    protocol_path=ROOT/'protocol.json'
    if protocol_path.exists():raise ValueError('Preserve existing validation results; use a new experiment version to rerun')
    write(protocol_path,PROTOCOL)
    rows=[];profiles={};manifest=[]
    for name,path in FILES.items():
        audio=decode(path);split=SPLITS[name]
        assert not any(interval_overlap(a,b) for a in split['calibration'] for b in split['heldout'])
        assert max(split['heldout']+split['calibration'])+SECONDS <= len(audio)/SR
        calibration=[]
        print('Calibrating disjoint reference profile:',name,flush=True)
        for start in split['calibration']:
            feat,rms=features(audio[start*SR:(start+SECONDS)*SR]);calibration.append(feat)
        profile=calibrate(calibration);profiles[name]=profile
        write(ROOT/'profiles.json',profiles)
        manifest.append(dict(reference=name,source_sha256=digest(path),calibration=split['calibration'],heldout=split['heldout']))
        # Profile is fixed before reading any held-out feature vectors or labels.
        for index,start in enumerate(split['heldout']):
            original=audio[start*SR:(start+SECONDS)*SR]
            for kind in ['original',*BROKEN,*CONTROLS,'silence','noise']:
                variant=perturb(original,kind,seed=987001+index)
                # Common headroom prevents clipping artefacts becoming the label.
                # Gain differences remain in the gain control by design.
                peak=float(np.max(np.abs(variant)))
                if peak>.98:variant=variant*(.98/peak)
                file=f'{name}_{start}_{kind}.flac';sf.write(ROOT/'audio'/file,variant,SR,subtype='PCM_24')
                feat,rms=features(variant)
                candidate,components=score(profile,feat,rms);baseline,_=score(profile,feat,rms,False)
                row=dict(reference=name,start_seconds=start,kind=kind,audio='audio/'+file,
                         candidate_score=candidate,static_baseline_score=baseline,components=components,rms=rms)
                rows.append(row)
            write(ROOT/'scores.json',rows)
            print('Scored held-out section',name,start,flush=True)
    write(ROOT/'manifest.json',dict(sources=manifest,protocol_sha256=digest(protocol_path),policy_training_started=False,
          calibration='Robust centers/scales only; no preference-model fitting or weights optimized on held-out outcomes',
          source_stems_available=False,source_transcriptions_available=False))
    report_results(rows)


def report_results(rows):
    originals={(r['reference'],r['start_seconds']):r for r in rows if r['kind']=='original'}
    result={}
    for label,column in [('structured_candidate','candidate_score'),('static_baseline','static_baseline_score')]:
        deltas=[originals[(r['reference'],r['start_seconds'])][column]-r[column] for r in rows if r['kind'] in BROKEN]
        per_reference={};per_type={}
        for reference in FILES:
            current=[r for r in rows if r['reference']==reference and r['kind'] in BROKEN]
            per_reference[reference]=float(np.mean([originals[(reference,r['start_seconds'])][column]>r[column] for r in current]))
            per_type[reference]={kind:float(np.mean([originals[(reference,r['start_seconds'])][column]>r[column] for r in current if r['kind']==kind])) for kind in BROKEN}
        control_changes={kind:[abs(originals[(r['reference'],r['start_seconds'])][column]-r[column]) for r in rows if r['kind']==kind] for kind in CONTROLS}
        sanity=all(originals[(r['reference'],r['start_seconds'])][column]>r[column] for r in rows if r['kind'] in ['silence','noise'])
        # Bootstrap sections, not corruptions: variants of one section are dependent.
        groups=list(originals);rng=np.random.default_rng(987001);boots=[]
        by_group={key:[originals[key][column]>r[column] for r in rows if (r['reference'],r['start_seconds'])==key and r['kind'] in BROKEN] for key in groups}
        for _ in range(1000):
            selected=rng.integers(0,len(groups),len(groups));boots.append(float(np.mean([win for i in selected for win in by_group[groups[i]]])))
        win_rate=float(np.mean(np.array(deltas)>0));control_p90={k:float(np.percentile(v,90)) for k,v in control_changes.items()}
        gate=win_rate>=.8 and min(per_reference.values())>=.75 and all(v>=2/3 for values in per_type.values() for v in values.values()) and all(v<=.15 for v in control_p90.values()) and sanity
        result[label]=dict(pairwise_win_rate=win_rate,mean_score_margin=float(np.mean(deltas)),per_reference=per_reference,per_corruption=per_type,
                           control_p90_absolute_score_change=control_p90,sanity_passed=sanity,
                           section_bootstrap_95_win_rate_interval=np.quantile(boots,[.025,.975]).tolist(),automated_gate_passed=bool(gate))
    write(ROOT/'results.json',result)
    write(ROOT/'acceptance.json',dict(rl_training_allowed=False,automated_gate_passed=result['structured_candidate']['automated_gate_passed'],
          human_corruption_label_review='pending',generated_midi_renderer_validation='not performed',
          interpretation='Even an automated pass is only a synthetic stress test on two recordings; it does not validate taste or generation quality.'))
    fig,ax=plt.subplots(figsize=(11,5));labels=[];values=[]
    for reference in FILES:
        for kind in BROKEN:
            current=[r for r in rows if r['reference']==reference and r['kind']==kind]
            labels.append(reference+'\n'+kind);values.append(float(np.mean([originals[(reference,r['start_seconds'])]['candidate_score']-r['candidate_score'] for r in current])))
    ax.bar(np.arange(len(values)),values,color=['#278c72' if v>0 else '#bd4f4f' for v in values]);ax.axhline(0,color='black',lw=1)
    ax.set_xticks(np.arange(len(labels)),labels,rotation=35,ha='right',fontsize=8);ax.set_ylabel('Original minus altered score');ax.set_title('Held-out reward stress test: positive margins favour original');fig.tight_layout();fig.savefig(ROOT/'score_margins.png',dpi=140);plt.close(fig)
    report=['# Reference reward validation','',
      'No policy training or preference-model fitting was performed. Profiles and thresholds were fixed before held-out scoring.',
      'Calibration and test excerpts are disjoint within each recording; the two recordings are NOT independent unseen-piece validation.','',
      '| Reward | Original beats disruption | Section-bootstrap 95% interval | Automated gate |',
      '|---|---:|---|---|']
    for name,r in result.items():report.append(f"| {name} | {r['pairwise_win_rate']:.1%} | {r['section_bootstrap_95_win_rate_interval']} | {r['automated_gate_passed']} |")
    report+=['','![Held-out score margins](score_margins.png)','',
      '## What this can establish','',
      'Whether a fixed reference-feature score distinguishes untouched held-out audio from the specified synthetic disruptions and tolerates nuisance production changes.',
      'It cannot establish general musical taste, reliable melody quality, stem-level cooperation, or reward transfer from mastered recordings to SoundFont MIDI renderings.','',
      'Phrase order is shuffled in coarse three-second chunks, not reviewed score-aligned phrases. Band desynchronization offsets frequency bands, not instrument stems. Pitch clash overlays a shifted full mix, not an isolated harmony instrument. Short loops can be artistically valid: all assumed-negative labels need listening review.',
      'Silence/noise are sanity checks only, excluded from the main win rate. Profile score is transposition-aware only through invariant summaries, not through exact melody matching.','',
      '## Frozen decision','',
      'RL remains blocked pending human label review and renderer-transfer checks, regardless of the automated gate. If this version fails, preserve the result and test any revised reward as a new version with new held-out material.','',
      '## Files','',
      '- protocol.json: predeclared split, feature weights and thresholds',
      '- profiles.json: calibration-only robust feature centers/scales',
      '- scores.json / results.json: per-section component scores and gate outcomes',
      '- acceptance.json: explicit training block',
      '- listen.html and audio/: original, disruptions and production controls for review']
    (ROOT/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    make_page(rows,result)
    print(json.dumps(result,indent=2),flush=True)


def make_page(rows,result):
    pairs=[dict(reference=r['reference'],start=r['start_seconds']) for r in rows if r['kind']=='original']
    identity=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reward validation — no training</title><style>body{font:18px system-ui;background:#101724;color:#e8edf5;max-width:1000px;margin:40px auto;padding:20px}p{line-height:1.6}section{background:#1c293d;padding:22px;margin:20px 0;border-radius:12px}.players{display:grid;grid-template-columns:1fr 1fr;gap:20px}audio{width:100%}select,button{font:inherit;padding:8px;margin:5px}a{color:#9bd2ff}@media(max-width:650px){.players{grid-template-columns:1fr}}</style><h1>Can the reward detect broken music?</h1><p>No training was run. These are held-out sections from your recordings, compared with controlled audio changes. Musical degradation labels are assumptions for you to review.</p><p id="outcome"></p><section><label>Reference section <select id="section"></select></label><label>Change <select id="kind"></select></label><div class="players"><div><h2>Original</h2><audio id="original" controls></audio></div><div><h2>Changed version</h2><audio id="altered" controls></audio></div></div><p id="score"></p><p>Is this change musically worse than the original?</p><button data-vote="worse">Yes, worse</button><button data-vote="similar">Similar / acceptable</button><button data-vote="better">Changed version better</button><button data-vote="unsure">Unsure</button></section><button id="download">Download label reviews</button><p id="status"></p><p><a href="REPORT.md">Full report</a> · <a href="results.json">Results</a> · <a href="protocol.json">Predeclared protocol</a> · <a href="acceptance.json">Training gate</a></p><script>
const rows=__ROWS__,sections=__SECTIONS__,results=__RESULTS__,hash='__HASH__',votes=JSON.parse(localStorage.getItem('reward-review-'+hash)||'{}'),section=document.getElementById('section'),kind=document.getElementById('kind');sections.forEach((s,i)=>section.add(new Option(s.reference+' — '+s.start+'s',i)));['phrase_shuffle','band_desync','pitch_clash','short_loop','gain_minus_6db','mild_room','transpose_2_semitones','silence','noise'].forEach(k=>kind.add(new Option(k.replaceAll('_',' '),k)));function load(){const s=sections[+section.value],a=rows.find(r=>r.reference===s.reference&&r.start_seconds===s.start&&r.kind==='original'),b=rows.find(r=>r.reference===s.reference&&r.start_seconds===s.start&&r.kind===kind.value);document.getElementById('original').src=a.audio;document.getElementById('altered').src=b.audio;document.getElementById('score').textContent='Structured reward: original '+a.candidate_score.toFixed(3)+' / changed '+b.candidate_score.toFixed(3)+' · Static baseline: '+a.static_baseline_score.toFixed(3)+' / '+b.static_baseline_score.toFixed(3)}section.onchange=load;kind.onchange=load;load();document.querySelectorAll('audio').forEach(a=>a.onplay=()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()}));document.querySelectorAll('[data-vote]').forEach(b=>b.onclick=()=>{const s=sections[+section.value];votes[s.reference+'|'+s.start+'|'+kind.value]=b.dataset.vote;localStorage.setItem('reward-review-'+hash,JSON.stringify(votes));document.getElementById('status').textContent=Object.keys(votes).length+' changes reviewed'});document.getElementById('download').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({panel_sha256:hash,votes},null,2)],{type:'application/json'}));a.download='reward-label-reviews.json';a.click()};const r=results.structured_candidate;document.getElementById('outcome').textContent='Original ranked higher in '+(r.pairwise_win_rate*100).toFixed(1)+'% of disruption tests. Automated gate: '+(r.automated_gate_passed?'PASS':'FAIL')+'. RL training remains blocked pending listening review and renderer checks.';</script></html>'''
    page=page.replace('__ROWS__',json.dumps(rows)).replace('__SECTIONS__',json.dumps(pairs)).replace('__RESULTS__',json.dumps(result)).replace('__HASH__',identity)
    (ROOT/'listen.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
