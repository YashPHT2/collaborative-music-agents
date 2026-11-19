"""Generate a fresh, genuinely label-hidden, offline ensemble calibration ballot."""
import base64,copy,hashlib,json,random,subprocess,zipfile
from pathlib import Path
import torch
from band import ROOT,Brief,TeamCritic,load_model,rollout,ROLES,ensemble_reward
from band_reward import ensemble_reward_v2
from train_band import make_policies
from accompany import write_result
from evaluate_baseline import validate_audio

def clip(notes,start):
    end=start+7680
    return sorted([['note',max(0,n[1]-start),min(end,n[1]+n[2])-max(start,n[1]),*n[3:]]
                   for n in notes if n[1]<end and n[1]+n[2]>start],key=lambda n:(n[1],n[3],n[4]))

def alter(notes,kind):
    changed=copy.deepcopy(notes)
    for n in changed:
        if kind=='bass_displacement' and n[3]==1:
            n[1]+=240; n[2]=min(n[2],7680-n[1])
        if kind=='piano_bar_order' and n[3]==0:
            bar=n[1]//1920; n[1]=[2,0,3,1][bar]*1920+n[1]%1920
            n[2]=min(n[2],1920-n[1]%1920)
    changed=[n for n in changed if n[1]<7680 and n[2]>0]
    # Trim a same-pitch sustain before the next onset after the diagnostic edit.
    changed.sort(key=lambda n:(n[1],n[3],n[4]))
    for i,n in enumerate(changed):
        following=next((p for p in changed[i+1:] if p[3:5]==n[3:5]),None)
        if following: n[2]=min(n[2],following[1]-n[1])
    return [n for n in changed if n[2]>0]

def render_clip(root,key,notes,bpm=100):
    path=root/f'{key}.mp3'
    if path.exists(): return path
    tracks=[[['set_tempo',0,int(60_000_000/bpm)],['time_signature',0,4,2,24,8]]]
    for role,s in ROLES.items():
        tracks.append([['track_name',0,role.encode()],['patch_change',0,s['channel'],s['patch']],*[n for n in notes if n[3]==s['channel']]])
    midi=write_result(root,key,[480,*tracks],True)
    normalized=root/f'{key}_normalized.wav'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(midi.with_suffix('.wav')),
        '-af','apad,atrim=duration=10.6,afade=t=out:st=10.3:d=0.3,loudnorm=I=-18:TP=-1.5:LRA=11',
        '-ar','44100','-ac','2','-map_metadata','-1',str(normalized)],check=True,capture_output=True)
    check=validate_audio(normalized)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(normalized),'-map_metadata','-1','-q:a','2',str(path)],check=True,capture_output=True)
    (root/f'{key}_audio.json').write_text(json.dumps(check))
    return path

def main():
    torch.set_num_threads(4)
    root=ROOT/'outputs'/'band_team_ballot_v2'; root.mkdir(parents=True,exist_ok=True)
    assets=root/'organizer_audio'; assets.mkdir(exist_ok=True)
    if (root/'Band_Team_Ballot_v2.zip').exists(): raise ValueError('Ballot exists; preserve IDs and votes')
    model=load_model(); critic=TeamCritic()
    methods={}
    for name,path in [('previous',ROOT/'outputs/band_joint_v2/latest.pt'),('reward_v2',ROOT/'outputs/band_reward_v2/latest.pt')]:
        ck=torch.load(path,map_location='cpu',weights_only=False); actors=make_policies()
        for r,a in actors.items(): a.load_state_dict(ck['actors'][r])
        methods[name]=actors
    candidates=[]; audit=[]; scores=[]
    for source in range(5):
        seed=920001+source; tonic=[9,0,2,5,7][source]; minor=bool(source%2==0)
        b=Brief(8,100,tonic,minor)
        performances={}
        for method,actors in methods.items():
            cache=assets/f'{seed}_{method}_events.json'
            if cache.exists(): notes=json.loads(cache.read_text())
            else:
                notes=rollout(model,actors,critic,b,seed)['notes']; cache.write_text(json.dumps(notes))
            performances[method]=notes
        for half in range(2):
            passage=f'p{source*2+half+1:02}'
            ns={m:clip(n,half*7680) for m,n in performances.items()}
            files={m:render_clip(assets,f'{passage}_{m}',n) for m,n in ns.items()}
            candidates.append(dict(passage=passage,kind='reward_training',methods=['previous','reward_v2'],files=list(files.values())))
            for kind in ['bass_displacement','piano_bar_order']:
                changed=alter(ns['previous'],kind)
                if changed==ns['previous']: raise ValueError('Identical diagnostic')
                path=render_clip(assets,f'{passage}_{kind}',changed)
                candidates.append(dict(passage=passage,kind=kind,methods=['unaltered',kind],files=[files['previous'],path]))
            for method,notes in ns.items():
                for version,fn in [('v1',ensemble_reward),('v2',ensemble_reward_v2)]:
                    values=[fn(notes,start,start+1920,Brief(4,100,tonic,minor))[0] for start in range(0,7680,1920)]
                    scores.append(dict(passage=passage,method=method,reward_version=version,mean=sum(values)/4))
            print(f'Prepared passage {passage}: 3 comparisons',flush=True)
    rng=random.Random(20261019); rng.shuffle(candidates)
    placements=[False]*15+[True]*15; rng.shuffle(placements)
    pairs=[]; encoded={}
    for i,(case,swap) in enumerate(zip(candidates,placements),1):
        order=[1,0] if swap else [0,1]; pid=f'B{i:02}'
        public=dict(id=pid,A=None,B=None)
        mapping={}
        for label,j in zip(['A','B'],order):
            path=case['files'][j]; digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest not in encoded: encoded[digest]='data:audio/mpeg;base64,'+base64.b64encode(path.read_bytes()).decode()
            public[label]=digest
            mapping[label]=dict(method=case['methods'][j],sha256=digest,path=str(path.relative_to(root)))
        if public['A']==public['B']: raise ValueError('Identical A/B audio')
        pairs.append(public); audit.append(dict(id=pid,passage=case['passage'],kind=case['kind'],mapping=mapping))
    (root/'organizer_manifest.json').write_text(json.dumps(dict(ballot_id='band-team-v2',pairs=audit,
        independent_source_seeds=5,passages=10,training_comparisons=10,diagnostic_comparisons=20,
        caveat='Three comparisons per passage are correlated. Edited versions are diagnostic hypotheses, not known worse music.'),indent=2))
    (root/'reward_audit.json').write_text(json.dumps(scores,indent=2))
    page=PAGE.replace('__PAIRS__',json.dumps(pairs)).replace('__AUDIO__',json.dumps(encoded))
    (root/'OPEN_ME.html').write_text(page,encoding='utf-8')
    readme='''THREE-MUSICIAN BLIND BALLOT v2
Extract the ZIP on a laptop and open OPEN_ME.html in Chrome, Edge or Firefox.
Audio is embedded; no internet, server or installation is needed.
There are 30 comparisons of roughly ten-second full-band excerpts, in three sets
of ten. You may take breaks. Replay both versions at the same comfortable volume.
Rate enjoyment, coherence, and whether the instruments work together separately.
Tie and Neither good are valid. Optional comments are particularly helpful.
Add a name/alias and download JSON votes, or copy text for WhatsApp. Export before
closing; local saving may be unavailable in some browsers. Nothing is submitted.
Please vote independently before discussing. A/B labels are specific to this ballot.
Some passages recur in separate comparisons; these are not 30 independent songs.
The methods and reward scores are deliberately excluded from this participant ZIP.
'''
    (root/'READ_ME.txt').write_text(readme,encoding='utf-8')
    message='Please try our three-musician blind ballot. Download the ZIP to a laptop, extract it, and open OPEN_ME.html. No installation or internet required. There are 30 short A/B comparisons in three sets; breaks are fine. Rate enjoyment, coherence, and how well piano/bass/drums work together. Tie or Neither good is valid. Please vote independently, then send the downloaded JSON or copied vote text in WhatsApp.\n'
    (root/'WHATSAPP_MESSAGE.txt').write_text(message,encoding='utf-8')
    archive=root/'Band_Team_Ballot_v2.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in ['OPEN_ME.html','READ_ME.txt']: z.write(root/file,'Band_Team_Ballot_v2/'+file)
    print(json.dumps(dict(archive=str(archive),pairs=len(pairs),audio_assets=len(encoded),megabytes=archive.stat().st_size/1024**2),indent=2))

PAGE='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Band listening ballot</title>
<style>body{font:17px system-ui;background:#111927;color:#edf3fc;max-width:860px;margin:auto;padding:24px}section{background:#213049;padding:20px;margin:22px 0;border-radius:12px}audio{width:100%}select,input,button,textarea{font:inherit;margin:8px 0;padding:8px;max-width:100%;box-sizing:border-box}label{display:block}textarea{width:100%}.muted{color:#b5c5db}button{cursor:pointer}fieldset{border:0;padding:0}#text{width:100%}</style>
<h1>Listen to the band</h1><p>30 short A/B comparisons · three sets of ten · take breaks whenever needed.</p><p>Judge the full piano, bass and drum performance. Rate independently before discussing. Choose Tie if equally good or indistinguishable, and Neither good if both are poor.</p>
<label>Name or alias (optional)<input id="name" maxlength="60"></label><p id="status" role="status"></p>
<button id="download">Download votes</button> <button id="copy">Show text for WhatsApp</button><textarea id="text" hidden readonly rows="10"></textarea><main id="pairs"></main>
<script>
const pairs=__PAIRS__,audio=__AUDIO__,key='band-team-v2',metrics=['enjoyment','coherence','together'];let votes={},name='';
try{const saved=JSON.parse(localStorage.getItem(key)||'{}');votes=saved.votes||{};name=saved.name||''}catch{}
const person=document.getElementById('name');person.value=name;function save(){try{localStorage.setItem(key,JSON.stringify({votes,name:person.value}))}catch{}status()}
person.oninput=save;function status(){const completed=pairs.filter(p=>metrics.every(m=>votes[p.id]?.[m])).length;document.getElementById('status').textContent=completed+'/30 fully rated. Export before closing.'}
for(let i=0;i<pairs.length;i++){const p=pairs[i];if(i%10===0){const h=document.createElement('h2');h.textContent='Set '+(i/10+1);document.getElementById('pairs').append(h)}const card=document.createElement('section');const h=document.createElement('h3');h.textContent='Comparison '+p.id;card.append(h);
for(const l of ['A','B']){const label=document.createElement('p');label.textContent=l;const a=document.createElement('audio');a.controls=true;a.preload='none';a.src=audio[p[l]];a.onplay=()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()});card.append(label,a)}
for(const metric of metrics){const label=document.createElement('label');label.textContent={enjoyment:'Which do you enjoy more?',coherence:'Which flows more coherently?',together:'Which has the instruments working better together?'}[metric];const s=document.createElement('select');s.setAttribute('aria-label',p.id+' '+metric);for(const v of ['','A','B','Tie','Neither good']){const o=document.createElement('option');o.value=v;o.textContent=v||'Choose a rating';s.append(o)}s.value=votes[p.id]?.[metric]||'';s.onchange=()=>{votes[p.id]??={};votes[p.id][metric]=s.value;save()};label.append(s);card.append(label)}
const note=document.createElement('textarea');note.placeholder='Optional: what worked or sounded wrong?';note.setAttribute('aria-label',p.id+' comments');note.value=votes[p.id]?.comment||'';note.maxLength=1500;note.oninput=()=>{votes[p.id]??={};votes[p.id].comment=note.value;save()};card.append(note);document.getElementById('pairs').append(card)}
function result(){return{ballot_id:'band-team-v2',participant:person.value.trim(),created_at:new Date().toISOString(),ratings:pairs.map(p=>({id:p.id,...(votes[p.id]||{})}))}}
document.getElementById('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(result(),null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='band-team-v2-'+(person.value.trim().replace(/[^a-zA-Z0-9_-]/g,'_')||'anonymous')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
document.getElementById('copy').onclick=()=>{const r=result();const t=document.getElementById('text');t.hidden=false;t.value=['Band team ballot v2','Voter: '+(r.participant||'Anonymous'),...r.ratings.map(v=>v.id+': enjoyment='+(v.enjoyment||'unrated')+'; coherence='+(v.coherence||'unrated')+'; together='+(v.together||'unrated')+'; comment='+(v.comment||''))].join('\\n');t.focus();t.select()};status();
</script></html>'''

if __name__=='__main__': main()
