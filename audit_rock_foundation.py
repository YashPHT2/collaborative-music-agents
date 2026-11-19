"""Fixed source/continuation note audit plus sound-palette and solo controls."""
import collections,copy,gzip,html,json,math,subprocess
from pathlib import Path
import numpy as np,soundfile as sf
import generate_rock_comparison as g

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/rock_foundation_audit_v1';INPUT=ROOT/'outputs/rock_prompt_set_v1/prompt_01';RATE=44100

def crop(notes):
    if any(n[1]<3840<n[1]+n[2] for n in notes):raise ValueError('Context sustain crosses audit boundary')
    out=[]
    for n in notes:
        if 3840<=n[1]<11520:
            n=copy.deepcopy(n);n[1]-=3840
            if n[1]+n[2]>7680:raise ValueError('Note crosses target end')
            out.append(n)
    return out

def role_metrics(notes,role,bounds):
    s=g.rock.ROLES[role];own=sorted([n for n in notes if n[3]==s['id']],key=lambda n:(n[1],n[4]))
    pitches=[n[4] for n in own];onsets=collections.Counter(n[1] for n in own);active=np.zeros(256,dtype=bool)
    for n in own:active[n[1]//30:min(256,(n[1]+n[2])//30)]=True
    return dict(notes=len(own),distinct_pitches=len(set(pitches)),pitch_min=min(pitches) if pitches else None,pitch_max=max(pitches) if pitches else None,
        pitch_median=float(np.median(pitches)) if pitches else None,mean_duration_beats=float(np.mean([n[2]/480 for n in own])) if own else None,
        onsets_per_beat=len(own)/16,distinct_onset_times=len(onsets),largest_simultaneous_attack=max(onsets.values(),default=0),sounding_fraction=float(active.mean()),
        outside_training_pitch_support=sum(not bounds['min']<=p<=bounds['max'] for p in pitches),training_pitch_bounds=bounds,
        drum_piece_counts=dict(collections.Counter(pitches)) if role=='drums' else None)

def page(report,assets):
    rows=''
    for role in g.rock.ROLES:
        a=report['metrics']['source'][role];b=report['metrics']['ai'][role]
        rows+=f'<tr><td>{role.replace("_"," ")}</td><td>{a["notes"]} / {b["notes"]}</td><td>{a["pitch_min"]}–{a["pitch_max"]} / {b["pitch_min"]}–{b["pitch_max"]}</td><td>{a["mean_duration_beats"]:.2f} / {b["mean_duration_beats"]:.2f}</td><td>{b["outside_training_pitch_support"]}</td></tr>'
    solos=''
    for role in g.rock.ROLES:
        solos+=f'<details><summary>{role.replace("_"," ")}</summary><p>Source part</p><audio controls preload="none" src="source_current_{role}.mp3"></audio><p>AI part</p><audio controls preload="none" src="ai_current_{role}.mp3"></audio></details>'
    script='''for(const a of document.querySelectorAll('audio'))a.onplay=()=>{for(const b of document.querySelectorAll('audio'))if(a!==b)b.pause()};document.getElementById('palette').onchange=e=>{for(const a of document.querySelectorAll('audio[data-version]')){a.pause();a.src=a.dataset.version+'_'+e.target.value+'_full.mp3';a.load()}};'''
    (OUT/'player_check.js').write_text(script)
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rock foundation audit</title><style>body{font:18px/1.6 system-ui;max-width:1000px;margin:30px auto;padding:20px;background:#14212b;color:#edf5ef}section,details{background:#233746;padding:20px;margin:18px 0;border-radius:12px}audio{width:100%}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:10px;border-bottom:1px solid #647581;text-align:left}select{font:inherit;padding:10px}a{color:#a6e7cb}</style><h1>Where is the rock music going wrong?</h1><p>One fixed example from the first prompt in the previous set. The two-bar opening is removed. Source is the next four bars of dataset MIDI; AI is step128’s generated continuation after that opening. Each clip is 14 seconds at 80 BPM, including the release tail.</p><p>The source is the selected five-role training projection, rather than the complete original arrangement. Other source instruments were omitted by the rock loader.</p><section><h2>Full band: same notes, different instrument sounds</h2><label>Guitar sounds <select id="palette"><option value="current">Current: overdrive + distortion</option><option value="source_palette">Source palette: steel-string + clean electric</option></select></label><p>Source continuation</p><audio controls preload="metadata" data-version="source" src="source_current_full.mp3"></audio><p>AI continuation</p><audio controls preload="metadata" data-version="ai" src="ai_current_full.mp3"></audio><p>The switch changes guitar programs only. Note pitches, timing, durations and velocities are identical within each version. Both palettes use the same sound bank; this does not test the quality of all possible renderers.</p></section><section><h2>Concrete findings</h2><p>The sampler permits pitches 0–127 for every role. In this example, AI bass reaches 82, while all bass training notes span 25–57. Seven of its fifteen bass notes lie outside that training range. The source bass spans 28–36.</p><p>The original source guitar programs are 25 and 27 (zero-based General MIDI), but our standard role mapping uses 29 and 30. Original MIDI verification confirmed steel-string and clean electric parts were being rendered as overdrive and distortion.</p><p>The source entry is titled “'''+html.escape(report['source_title'])+'''”. Its metadata labels rock, without verified happy-mood labels. This audit does not infer its mood from the title.</p><div style="overflow:auto"><table><thead><tr><th>Part</th><th>Notes: source / AI</th><th>Pitch range: source / AI</th><th>Mean beats/note: source / AI</th><th>AI notes outside train range</th></tr></thead><tbody>'''+rows+'''</tbody></table></div><p>These are descriptive measurements, not quality rewards or proof of a good/bad composition.</p></section><section><h2>Instrument solos</h2><p>These use the current guitar palette. Each solo keeps its full-band version’s gain so relative balance is preserved; solos are not individually made equally loud.</p>'''+solos+'''</section><p>No model training or GPU generation was performed. This is one-example diagnosis. Notes and drum-channel identity passed MIDI round-trip checks. The full-band mixes are loudness matched and every encoded file is checked for clipping.</p><script>'''+script+'</script></html>'
    (OUT/'listen.html').write_text(document,encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True);g.rock.configure();g.BARS=4;g.BPM=80
    protocol=json.loads((INPUT/'protocol.json').read_text());assert protocol['bpm']==80
    remote=json.loads((INPUT/'remote_verification.json').read_text());ai_path=INPUT/'step128_prompted.json';assert g.sha(ai_path)==remote['clips']['step128_prompted']['sha256']
    ai=json.loads(ai_path.read_text());assert sorted(ai['notes'])==sorted(ai['reference_prefix']+[x['note'] for x in ai['decisions']])
    source=json.loads((INPUT/'reference_window.json').read_text());assert source['id']==protocol['prompt_source_window']
    episodes={'source':dict(notes=crop(source['notes'])),'ai':dict(notes=crop(ai['notes']))}
    with gzip.open(ROOT/'outputs/rock_training_v1/rock_examples.json.gz','rt') as f:windows=json.load(f)
    original=next(w for w in windows if w['id']==source['id'])
    assert source['notes']==original['notes'] and source['bpm']==original['bpm'] and source['group']==original['group']
    pitches=collections.defaultdict(list)
    for w in windows:
        if w['split']=='train':
            for n in w['notes']:pitches[n[3]].append(n[4])
    bounds={r:dict(min=min(pitches[s['id']]),max=max(pitches[s['id']])) for r,s in g.rock.ROLES.items()}
    instruments=json.loads((OUT/'source_instruments.json').read_text());assert instruments['source']['id'] in {r['source'] for r in json.loads((ROOT/'outputs/rock_training_v1/rock_manifest.json').read_text())['sources'] if r['group']==protocol['prompt_group']}
    report=dict(schema='rock-foundation-audit-v1',source_window=source['id'],source_group=source['group'],source_title=instruments['source']['title'],
        source_sha256=g.sha(INPUT/'reference_window.json'),ai_sha256=g.sha(ai_path),training_data_sha256=g.sha(ROOT/'outputs/rock_training_v1/rock_examples.json.gz'),
        metrics={name:{r:role_metrics(ep['notes'],r,bounds[r]) for r in g.rock.ROLES} for name,ep in episodes.items()},
        current_sampler_pitch_support={r:[min(s['pitches']),max(s['pitches'])] for r,s in g.rock.ROLES.items()},
        palettes=dict(current={r:g.rock.ROLES[r]['patch'] for r in instruments['patches']},source_palette=instruments['patches']),
        no_training=True,no_gpu_generation=True,notes_reused_without_edits=True,
        limitation='One prompted continuation, not an aggregate quality finding. Training pitch bounds are observed data support, not universal physical instrument limits.')
    raws={};fulls={};all_peak_db={};metrics_lufs={}
    for version,ep in episodes.items():
        for palette in ['current','source_palette']:
            directory=OUT/(version+'_'+palette);directory.mkdir(exist_ok=True)
            old={r:g.rock.ROLES[r]['patch'] for r in instruments['patches']}
            try:
                if palette=='source_palette':
                    for r,patch in instruments['patches'].items():g.rock.ROLES[r]['patch']=patch
                full_midi=g.export(directory,ep)
            finally:
                for r,patch in old.items():g.rock.ROLES[r]['patch']=patch
            variants={'full':full_midi}
            if palette=='current':variants.update({r:directory/(r+'.mid') for r in g.rock.ROLES})
            peaks=[]
            for part,midi in variants.items():
                name=f'{version}_{palette}_{part}';raw=directory/(part+'_raw.wav')
                if not raw.exists():g.render(midi,raw)
                audio,sr=sf.read(raw,always_2d=True);assert sr==RATE
                if np.max(np.abs(audio))>=32767/32768:raise ValueError('Native render reached the PCM limit; rerender with more headroom before normalization')
                fixed=np.zeros((14*RATE,2));n=min(len(audio),len(fixed));fixed[:n]=audio[:n];fixed[-RATE:]*=np.linspace(1,0,RATE)[:,None]
                path=directory/(part+'_fixed.wav');sf.write(path,fixed,RATE,subtype='FLOAT');raws[name]=path
                integrated,peak=g.loudness(path);peaks.append(peak)
                if part=='full':fulls[version,palette]=name;metrics_lufs[version,palette]=integrated
                print('Rendered',name,flush=True)
            all_peak_db[version,palette]=max(peaks)
    target=min([-22.]+[metrics_lufs[k]-2-all_peak_db[k] for k in fulls])
    gains={};assets={}
    # Full mixes determine gain; the matching solos inherit that exact gain.
    for key,name in fulls.items():
        gain=target-metrics_lufs[key];wav=OUT/(name+'.wav')
        for attempt in range(3):
            subprocess.run(['ffmpeg','-y','-v','error','-i',str(raws[name]),'-af',f'volume={gain:.9f}dB','-c:a','pcm_s24le',str(wav)],check=True)
            measured,_=g.loudness(wav)
            if abs(measured-target)<=.03:break
            gain+=target-measured
        gains[key]=gain
    for name,raw in raws.items():
        version,palette_part=name.split('_',1);palette='source_palette' if palette_part.startswith('source_palette_') else 'current'
        gain=gains[version,palette];wav=OUT/(name+'.wav');mp3=OUT/(name+'.mp3')
        if name not in fulls.values():subprocess.run(['ffmpeg','-y','-v','error','-i',str(raw),'-af',f'volume={gain:.9f}dB','-c:a','pcm_s24le',str(wav)],check=True)
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-q:a','2',str(mp3)],check=True)
        decoded=OUT/(name+'_decoded.wav');subprocess.run(['ffmpeg','-y','-v','error','-i',str(mp3),str(decoded)],check=True)
        audio,sr=sf.read(decoded);decoded.unlink();assert np.isfinite(audio).all() and np.max(np.abs(audio))<1
        assets[name]=dict(sha256=g.sha(mp3),seconds=len(audio)/sr,decoded_peak=float(np.max(np.abs(audio))),gain_db=gain,midi_action_roundtrip=True)
    report['assets']=assets;report['full_mix_target_lufs']=target;report['full_mix_lufs']={name:g.loudness(OUT/(name+'.wav'))[0] for name in fulls.values()}
    g.write_json(OUT/'audit.json',report);page(report,assets);print('FOUNDATION AUDIT READY',flush=True)

if __name__=='__main__':main()
