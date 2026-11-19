"""Detailed reference measurements and explicitly uncertain musical estimates.

No training, reward fitting, note transcription or instrument-label invention.
Raw feature arrays are retained so later rewards can be audited and recomputed.
"""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_music_references import FILES

ROOT=Path(__file__).resolve().parent/'outputs'/'reference_details_v2'
SR=22050
HOP=512
NAMES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def distribution(x):
    values=np.asarray(x).reshape(-1);values=values[np.isfinite(values)]
    return dict(mean=float(np.mean(values)),p10=float(np.percentile(values,10)),median=float(np.median(values)),p90=float(np.percentile(values,90))) if len(values) else None


def loudness(path):
    result=subprocess.run(['ffmpeg','-hide_banner','-i',str(path),'-vn','-af','ebur128=peak=true','-f','null','-'],capture_output=True,text=True,check=True)
    summary=result.stderr[result.stderr.rfind('Summary:'):]
    def get(pattern):
        match=re.search(pattern,summary)
        return float(match.group(1)) if match else None
    return dict(integrated_lufs=get(r'I:\s+(-?[\d.]+) LUFS'),loudness_range_lu=get(r'LRA:\s+([\d.]+) LU'),
                true_peak_dbfs=get(r'Peak:\s+(-?[\d.]+) dBFS'),method='FFmpeg ebur128, full stereo source')


def tonal_candidates(chroma):
    base_major=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    base_minor=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    average=chroma.mean(axis=1);scores=[]
    for root in range(12):
        for kind,template in [('major',base_major),('minor',base_minor)]:
            score=float(np.corrcoef(average,np.roll(template,root))[0,1])
            scores.append(dict(label=NAMES[root]+' '+kind,correlation=score if np.isfinite(score) else 0.))
    return sorted(scores,key=lambda r:r['correlation'],reverse=True)[:4]


def chord_candidates(vector):
    templates=[]
    for root in range(12):
        for quality,intervals in [('major',[0,4,7]),('minor',[0,3,7]),('sus2',[0,2,7]),('sus4',[0,5,7])]:
            template=np.zeros(12);template[[(root+i)%12 for i in intervals]]=1
            similarity=float(vector@template/(np.linalg.norm(vector)*np.linalg.norm(template)+1e-9))
            templates.append(dict(label=NAMES[root]+' '+quality,similarity=similarity))
    return sorted(templates,key=lambda r:r['similarity'],reverse=True)[:3]


def analyze(path,name):
    out=ROOT/name;out.mkdir(parents=True,exist_ok=True)
    print('Decoding',name,flush=True)
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-vn','-ac','2','-ar',str(SR),'-f','f32le','pipe:1'])
    stereo=np.frombuffer(raw,dtype='<f4').reshape(-1,2).T.copy();y=stereo.mean(axis=0);duration=len(y)/SR
    D=librosa.stft(y,n_fft=2048,hop_length=HOP);S=np.abs(D);power=S*S
    print('Harmonic/percussive and spectral features',name,flush=True)
    harmonic,percussive=librosa.decompose.hpss(D)
    H,P=np.abs(harmonic),np.abs(percussive)
    mel=librosa.feature.melspectrogram(S=power,sr=SR,n_mels=64,fmin=40,fmax=10000)
    logmel=librosa.power_to_db(mel,ref=np.max)
    mfcc=librosa.feature.mfcc(S=logmel,n_mfcc=20)
    features=dict(rms=librosa.feature.rms(S=S),centroid=librosa.feature.spectral_centroid(S=S,sr=SR),
                  bandwidth=librosa.feature.spectral_bandwidth(S=S,sr=SR),rolloff=librosa.feature.spectral_rolloff(S=S,sr=SR,roll_percent=.85),
                  flatness=librosa.feature.spectral_flatness(S=S),contrast=librosa.feature.spectral_contrast(S=S,sr=SR),
                  zcr=librosa.feature.zero_crossing_rate(y,frame_length=2048,hop_length=HOP),
                  mfcc=mfcc,mfcc_delta=librosa.feature.delta(mfcc),mfcc_delta2=librosa.feature.delta(mfcc,order=2))
    times=librosa.frames_to_time(np.arange(S.shape[1]),sr=SR,hop_length=HOP)
    hpower=H*H;ppower=P*P
    harmonic_fraction=hpower.sum(axis=0)/(hpower.sum(axis=0)+ppower.sum(axis=0)+1e-12)
    chroma=librosa.feature.chroma_stft(S=hpower,sr=SR,n_fft=2048,hop_length=HOP)
    # CENS-like smoothing/normalization: stable tonal colour, not exact notes.
    cens=gaussian_filter1d(chroma,2,axis=1);cens/=np.maximum(np.linalg.norm(cens,axis=0,keepdims=True),1e-9)
    onset=librosa.onset.onset_strength(S=librosa.power_to_db(power,ref=np.max),sr=SR,hop_length=HOP)
    onset_frames=librosa.onset.onset_detect(onset_envelope=onset,sr=SR,hop_length=HOP,units='frames')
    onset_times=librosa.frames_to_time(onset_frames,sr=SR,hop_length=HOP)
    print('Pulse estimates and beat grid',name,flush=True)
    tempogram=librosa.feature.tempogram(onset_envelope=onset,sr=SR,hop_length=HOP,win_length=384)
    dynamic_tempo=librosa.feature.tempo(onset_envelope=onset,sr=SR,hop_length=HOP,aggregate=None,max_tempo=240)
    tempo,beats=librosa.beat.beat_track(onset_envelope=onset,sr=SR,hop_length=HOP,bpm=dynamic_tempo,units='time')
    # Track the strongest harmonic spectral peak only. In a mix it may be an
    # overtone or accompaniment; it is never labelled as the lead melody.
    frequencies,magnitudes=librosa.piptrack(S=H,sr=SR,n_fft=2048,hop_length=HOP,fmin=65,fmax=2000,threshold=.15)
    strongest=magnitudes.argmax(axis=0);peak=frequencies[strongest,np.arange(frequencies.shape[1])]
    peak_strength=magnitudes[strongest,np.arange(frequencies.shape[1])]
    peak_midi=np.where(peak>0,69+12*np.log2(np.maximum(peak,1e-9)/440),0)
    np.savez_compressed(out/'frame_features.npz',times=times,chroma=chroma,smoothed_chroma=cens,logmel=logmel,
                        onset_strength=onset,tempogram=tempogram,dynamic_tempo_bpm=dynamic_tempo,
                        harmonic_energy_fraction=harmonic_fraction,strongest_harmonic_peak_hz=peak,
                        strongest_harmonic_peak_midi=peak_midi,strongest_peak_magnitude=peak_strength,**features)
    seconds=np.arange(int(np.ceil(duration)));rows=[];chroma_sec=[];mfcc_sec=[];chords=[]
    for second in seconds:
        mask=(times>=second)&(times<second+1);left=int(second*SR);right=min(len(y),int((second+1)*SR));segment=stereo[:,left:right]
        mid=(segment[0]+segment[1])/2;side=(segment[0]-segment[1])/2
        width=float(np.sqrt(np.mean(side**2))/(np.sqrt(np.mean(mid**2))+1e-9))
        correlation=float(np.corrcoef(segment)[0,1]) if np.std(segment[0])>1e-8 and np.std(segment[1])>1e-8 else 0.
        v=chroma[:,mask].mean(axis=1);chroma_sec.append(v);mfcc_sec.append(mfcc[:,mask].mean(axis=1))
        rms=float(np.sqrt(np.mean(mid**2)))
        row=dict(start_seconds=int(second),rms_dbfs=float(20*np.log10(rms+1e-10)),peak_dbfs=float(20*np.log10(np.max(np.abs(segment))+1e-10)),
                 crest_factor_db=float(20*np.log10((np.max(np.abs(segment))+1e-10)/(rms+1e-10))),
                 stereo_side_mid_ratio=width,stereo_correlation=correlation,
                 harmonic_energy_fraction=float(np.mean(harmonic_fraction[mask])),
                 onset_count=int(np.sum((onset_times>=second)&(onset_times<second+1))),
                 estimated_tempo_bpm=float(np.median(dynamic_tempo[mask])))
        for key in ['centroid','bandwidth','rolloff','flatness','zcr']:
            row[key]=float(features[key][:,mask].mean())
        rows.append(row)
        chords.append(dict(start_seconds=int(second),candidates=chord_candidates(v) if row['rms_dbfs']>=-60 else [],
                           status='unverified_template_candidates' if row['rms_dbfs']>=-60 else 'below_-60_dbfs_no_harmony_label'))
    with (out/'per_second.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    cs=np.asarray(chroma_sec).T;norm=cs/np.maximum(np.linalg.norm(cs,axis=0,keepdims=True),1e-9);similarity=norm.T@norm
    novelty=np.zeros(len(seconds));radius=4
    for i in range(radius,len(seconds)-radius):
        a,b=norm[:,i-radius:i].mean(axis=1),norm[:,i:i+radius].mean(axis=1)
        novelty[i]=1-float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
    boundaries,_=find_peaks(novelty,distance=8,prominence=max(.01,float(np.std(novelty))))
    repetitions=[];size=8
    for a in range(0,len(seconds)-size,4):
        for b in range(a+size*2,len(seconds)-size,4):
            sim=float(np.mean(np.diag(similarity[a:a+size,b:b+size])))
            repetitions.append(dict(first_seconds=a,second_seconds=b,duration_seconds=size,pitch_class_similarity=sim))
    repetitions=sorted(repetitions,key=lambda r:r['pitch_class_similarity'],reverse=True)[:15]
    audible=np.array([r['rms_dbfs']>-60 for r in rows])
    summary=dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),duration_seconds=duration,
                 analysis_sample_rate=SR,frame_hop_samples=HOP,frame_resolution_ms=1000*HOP/SR,
                 loudness=loudness(path),key_candidates=tonal_candidates(chroma),
                 onset_count=len(onset_times),estimated_tempo_bpm=distribution(dynamic_tempo),
                 harmonic_energy_fraction=distribution(harmonic_fraction),
                 stereo_width=distribution([r['stereo_side_mid_ratio'] for r in rows if r['rms_dbfs']>-60]),
                 relative_dynamic_range_db=float(np.percentile([r['rms_dbfs'] for r in rows if r['rms_dbfs']>-60],95)-np.percentile([r['rms_dbfs'] for r in rows if r['rms_dbfs']>-60],5)),
                 candidate_section_change_seconds=boundaries.tolist(),repeated_tonal_passage_candidates=repetitions,
                 confidence=dict(loudness='direct signal measurement',spectral_features='direct signal measurement',
                                 tempo_and_beats='algorithmic estimates, half/double-time ambiguity; not the authored tempo map',
                                 chords_and_key='template candidates from mixed audio; not ground-truth harmony',
                                 predominant_peak='strongest harmonic partial; not a lead melody transcription',
                                 sections_and_repetition='tonal similarity candidates; not verified compositional phrases'),
                 not_extracted=['Exact notes for each instrument','Reliable instrument identities or stems','Authored meter and tempo changes',
                                'Exact melody and chord voicings','Bow articulation and performance technique labels','A validated human taste reward'])
    write_json(out/'summary.json',summary);write_json(out/'chord_candidates.json',chords)
    write_json(out/'timing_estimates.json',dict(onset_times_seconds=onset_times.tolist(),estimated_beat_times_seconds=np.asarray(beats).tolist(),
                 warning='Onset/beat detection estimates; orchestral attacks and tempo changes can be missed or doubled'))
    np.savez_compressed(out/'section_features.npz',seconds=seconds,chroma=cs,mfcc=np.array(mfcc_sec).T,self_similarity=similarity,novelty=novelty)
    fig,axes=plt.subplots(5,1,figsize=(13,15),gridspec_kw=dict(height_ratios=[1,1,1,1,2]))
    axes[0].plot(seconds,[r['rms_dbfs'] for r in rows]);axes[0].set_ylabel('RMS dBFS')
    axes[1].plot(times,onset,lw=.5);axes[1].set_ylabel('Onset strength')
    axes[2].plot(times,dynamic_tempo,lw=.8);axes[2].set_ylabel('Estimated pulse BPM\n(metrically ambiguous)')
    axes[3].imshow(cs,origin='lower',aspect='auto',extent=[0,duration,0,12],cmap='magma');axes[3].set_yticks(np.arange(12)+.5,NAMES);axes[3].set_ylabel('Pitch-class energy')
    axes[4].imshow(similarity,origin='lower',aspect='auto',extent=[0,duration,0,duration],vmin=0,vmax=1,cmap='magma');axes[4].set_ylabel('Time (seconds)')
    for ax in axes:ax.set_xlabel('Time (seconds)')
    fig.suptitle(name+' — measurements and uncertain estimates; no note/instrument transcription',fontsize=11);fig.tight_layout()
    fig.savefig(out/'overview.png',dpi=120);plt.close(fig)
    print('Finished',name,flush=True)
    return summary


def main():
    ROOT.mkdir(parents=True,exist_ok=True);summaries={}
    for name,path in FILES.items():summaries[name]=analyze(path,name)
    write_json(ROOT/'manifest.json',dict(version=2,librosa_version=librosa.__version__,numpy_version=np.__version__,
                 inputs={k:dict(path=v['source'],sha256=v['sha256']) for k,v in summaries.items()},
                 training_started=False,reward_fitting_started=False,frames='Approximately 23.2ms feature hops; analysis windows overlap and do not imply equivalent musical-label precision'))
    lines=['# Detailed reference extraction','',
           'Both user-provided audio files were analysed locally. No new training or reward fitting was performed.',
           'Raw arrays are retained at approximately 23.2ms hops, with one-second tables and longer-range recurrence analysis.',
           'Frame spacing is not note-timing accuracy. These are signal features plus uncertain musical estimates.','',
           '| Reference | Duration | Integrated LUFS | Loudness range LU | Detected onsets |',
           '|---|---:|---:|---:|---:|']
    for name,s in summaries.items():
        lines.append(f"| {name} | {s['duration_seconds']:.2f}s | {s['loudness']['integrated_lufs']} | {s['loudness']['loudness_range_lu']} | {s['onset_count']} |")
    lines+=['','## What was extracted','',
            '- Dynamics: RMS, peak, crest factor, integrated loudness, loudness range and time evolution.',
            '- Rhythm: onset times/strength, local tempogram, tempo candidates and estimated beat times.',
            '- Tonal content: chroma, smoothed chroma, ranked key and chord-template candidates.',
            '- Development: harmonic change novelty, section-change candidates, recurrence and repeated tonal-window candidates.',
            '- Timbre/texture: 64-band log-mel spectra, 20 MFCCs and their first/second derivatives, spectral contrast, centroid, bandwidth, rolloff, flatness and zero-crossing rate.',
            '- Layering/spatial behaviour: harmonic/percussive energy fractions, stereo width and channel correlation.',
            '- Pitch detail: strongest harmonic spectral peak tracks, explicitly not labelled as the lead melody.','',
            '## Limits that matter before designing rewards','',
            'Instrument stems, exact polyphonic notes, chord voicings, meter, articulations and expressive playing labels are NOT recovered by this extractor. HPSS separates harmonic/percussive components, not organ/strings/cello.',
            'Tempo may lock to half/double time; chord/key templates are ambiguous in an orchestral mix. Repeated tonal colour is not proof of repeated melody. The slowed/reverb reference is analysed exactly as supplied.',
            'No score here establishes musical enjoyment or is a validated reward. Raw timbre similarity would also confound professional orchestral production with the current SoundFont renderer.','',
            '## Feature files','']
    for name in summaries:
        lines += [f'### {name}','',f'![Feature overview]({name}/overview.png)','',
                  f'- [Measured summary and confidence]({name}/summary.json)',f'- [Per-second measurements]({name}/per_second.csv)',
                  f'- [Onset and beat estimates]({name}/timing_estimates.json)',f'- [Chord candidates]({name}/chord_candidates.json)',
                  f'- [All frame features]({name}/frame_features.npz)',f'- [Section/recurrence arrays]({name}/section_features.npz)','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Detailed report ready:',ROOT/'REPORT.md',flush=True)


if __name__=='__main__':main()
