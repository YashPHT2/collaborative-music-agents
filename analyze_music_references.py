"""Signal-level reference analysis. Not instrument separation or note transcription."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from scipy.signal import stft,find_peaks,correlate
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'outputs'/'reference_audio_analysis_v1'
FILES={
    'flowing_strings':Path(r'C:\Users\Yash\Downloads\spotifydown.com - Can You Hear The Music - Ludwig Göransson.mp3'),
    'spacious_cinematic':Path(r'C:\Users\Yash\Downloads\Hans Zimmer - S.T.A.Y (Interstellar Main Theme)(slowed  reverb).mp3')}


def analyze(path,name):
    sr=22050;hop=256
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-vn','-ac','1','-ar',str(sr),'-f','f32le','pipe:1'])
    x=np.frombuffer(raw,dtype='<f4');duration=len(x)/sr
    f,t,z=stft(x,sr,nperseg=2048,noverlap=2048-hop)
    mag=np.abs(z).astype(np.float32)
    # Frequency-bin whitening reduces dominance of broad spectral colour. This
    # is approximate pitch-class energy; harmonics and reverb still bias it.
    whitened=mag/np.maximum(np.mean(mag,axis=1,keepdims=True),1e-5)
    bins=np.flatnonzero((f>=65)&(f<=4000))
    pitch=np.rint(69+12*np.log2(f[bins]/440)).astype(int)%12
    chroma=np.zeros((12,len(t)),dtype=np.float32)
    for pc in range(12):chroma[pc]=whitened[bins[pitch==pc]].sum(axis=0)
    flux=np.maximum(np.diff(np.log1p(mag*100),axis=1),0).mean(axis=0)
    flux=np.r_[0,flux];flux/=max(float(np.percentile(flux,95)),1e-8)
    seconds=np.arange(int(np.ceil(duration)));pitch_seconds=[];energy=[];onsets=[];pulses=[]
    for second in seconds:
        frames=(t>=second)&(t<second+1)
        v=chroma[:,frames].mean(axis=1);v/=max(float(np.linalg.norm(v)),1e-8);pitch_seconds.append(v)
        signal=x[second*sr:min(len(x),(second+1)*sr)]
        energy.append(float(20*np.log10(np.sqrt(np.mean(signal**2))+1e-10)))
        onsets.append(float(np.mean(flux[frames])))
    pitch_seconds=np.array(pitch_seconds);similarity=pitch_seconds@pitch_seconds.T
    novelty=np.zeros(len(seconds))
    for i in range(4,len(seconds)-4):
        a=pitch_seconds[i-4:i].mean(0);b=pitch_seconds[i:i+4].mean(0)
        novelty[i]=1-float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
    peaks,_=find_peaks(novelty,prominence=.03,distance=8)
    # Local autocorrelation peaks are ambiguous metrical pulse candidates,
    # explicitly not an authoritative beat grid or a list of tempo changes.
    rate=sr/hop
    for start in range(0,max(1,int(duration)-7),4):
        segment=flux[(t>=start)&(t<start+8)]
        centered=segment-gaussian_filter1d(segment,rate*.25)
        ac=correlate(centered,centered,mode='full',method='fft')[len(segment)-1:]
        lag_low,lag_high=int(rate*60/240),min(len(ac)-1,int(rate*60/40))
        choices,_=find_peaks(ac[lag_low:lag_high])
        ordered=sorted((int(lag_low+i) for i in choices),key=lambda i:ac[i],reverse=True)[:3]
        pulses.append(dict(start_seconds=start,candidate_bpm=[round(60*rate/i,1) for i in ordered]))
    data=dict(name=name,source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),duration_seconds=duration,
              seconds=seconds.tolist(),rms_dbfs=energy,onset_strength=onsets,approximate_pitch_class_energy=pitch_seconds.tolist(),
              candidate_harmonic_change_seconds=peaks.tolist(),local_pulse_candidates=pulses,
              limitations=['No instrument identification, separated stems, reliable melody or exact chord labels.',
                           'Autocorrelation pulse candidates have half/double-time ambiguity and are not measured tempo annotations.',
                           'Similarity reflects broad pitch-class colour, not proof of a repeated melodic motif.',
                           'Analysis describes these exact files, including slowed/reverb processing. It is not a training score.'])
    (ROOT/f'{name}.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(3,1,figsize=(12,10),gridspec_kw=dict(height_ratios=[1,1,2]))
    axes[0].plot(seconds,energy,color='#126b8a');axes[0].set_ylabel('RMS dBFS');axes[0].set_title(path.stem)
    axes[1].plot(seconds,onsets,color='#9d5a25');axes[1].set_ylabel('Onset strength\n(relative)')
    image=axes[2].imshow(similarity,origin='lower',extent=[0,duration,0,duration],aspect='auto',vmin=0,vmax=1,cmap='magma')
    axes[2].set_ylabel('Time (seconds)');fig.colorbar(image,ax=axes[2],label='Pitch-class similarity')
    for ax in axes:ax.set_xlabel('Time (seconds)')
    fig.suptitle('Signal measurements — no automatic instrument or melody labels',fontsize=11)
    fig.tight_layout();fig.savefig(ROOT/f'{name}.png',dpi=130);plt.close(fig)
    return dict(name=name,duration_seconds=round(duration,2),quiet_to_loud_span_db=round(float(np.percentile(energy,90)-np.percentile(energy,10)),1),
                first_10_seconds_rms_dbfs=round(float(np.mean(energy[:10])),1),
                strongest_10_second_region_start=int(np.argmax(np.convolve(energy,np.ones(10)/10,mode='valid'))))


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    summary=[analyze(path,name) for name,path in FILES.items()]
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report='''# Reference audio analysis

These profiles come directly from the two user-provided MP3 files. They are
signal measurements, not a claim that the system has listened, transcribed an
orchestra, or learned the composers' musical judgement.

The plots show loudness development, transient activity and broad pitch-class
recurrence. The JSON files also contain local pulse candidates, with explicit
half/double-time ambiguity. No exact tempo map, melody, chord sequence or
instrument labels should be inferred without review.

## Turning the recordings into supervision

1. Obtain multi-instrument note transcriptions, then review melody, bass,
   rhythm alignment and phrase boundaries. Keep uncertain notes masked.
2. Build one shared beat/phrase/harmonic representation from the reviewed score.
   Train all role parts against that same representation and aligned targets.
3. Use the recordings' relative dynamics and activity envelopes as arrangement
   targets. Calibrate these against the same renderer before using audio losses;
   loudness and reverb are not measures of good composition.
4. Mix reference-derived examples with broader ensemble material. Evaluate on
   separate pieces and check for copied phrases: two recordings cannot establish
   general composing ability.
5. Add joint RL only after reconstruction and original phrase generation work.
   Compare teacher-scored structure and human preferences alongside transparent
   engineering rewards. Never call a closer audio envelope better music by itself.

## Outputs

![Flowing strings reference](flowing_strings.png)

![Spacious cinematic reference](spacious_cinematic.png)

The slowed/reverb file is analysed as provided. Its timing and decay differ
from an unprocessed original; this pipeline does not undo those effects.
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
