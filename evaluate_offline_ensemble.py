"""Matched eight-bar listening panel, provenance and partner interventions."""
import argparse, html, json, random, gzip, time, hashlib
from pathlib import Path
import torch
import offline_ensemble as oe
import note_trio as nt
from partner_response import ensemble_score

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=nt.ROOT/'outputs'/'offline_ensemble_v1')
    p.add_argument('--seeds',type=int,nargs='+',default=[9619101,9619102,9619103,9619104]);args=p.parse_args()
    root=args.output;items=[];probes=[];torch.set_num_threads(4)
    if (root/'listen.html').exists() and (root/'panel_identity.json').exists():
        print('Preserving the existing rendered panel and its ballot identity.',flush=True);return
    progress=torch.load(root/'progress.pt',map_location='cpu',weights_only=False)
    prior=json.loads((root/'evaluation_time.json').read_text()).get('seconds',0) if (root/'evaluation_time.json').exists() else 0
    remaining=progress['config']['hours']*3600-progress['state']['elapsed']-prior;started=time.monotonic()
    def budget():
        (root/'evaluation_time.json').write_text(json.dumps(dict(seconds=prior+time.monotonic()-started)))
        if time.monotonic()-started>=remaining:raise TimeoutError('Campaign compute limit reached; partial evaluation artifacts retained.')
    budget()
    import composition_data as cd
    windows=oe.candidates(cd.build(),8)
    cases_by_role={}
    for role in oe.ROLES:
        selected={}
        for w in windows:
            if w['split']=='test' and role in w['available_roles'] and len(w['available_roles'])>=2:selected.setdefault(w['group'],w)
        cases_by_role[role]=list(selected.values())[:10]
    baseline_scores={}
    for mode,checkpoint in [('baseline','cinematic.pt'),('planned','planned.pt'),('revised','revised.pt')]:
        if not (root/checkpoint).exists():continue
        model,adapters,agents,planner,conditions=oe.load(root/checkpoint)
        for seed in args.seeds:
            budget()
            name=f'{mode}_{seed}';ep=oe.compose(model,adapters,agents,planner,conditions,8,seed,mode)
            checks=nt.export(root,name,ep,stems=seed==args.seeds[0])
            (root/f'{name}_plans.json').write_text(json.dumps(ep['plans']))
            items.append(dict(id=name,mode=mode,seed=seed,audio=f'{name}_mix.mp3',notes=len(ep['notes']),checks=checks,
                audio_sha256=hashlib.sha256((root/f'{name}_mix.mp3').read_bytes()).hexdigest()))
        if mode=='baseline':
            for role in oe.ROLES:
                for index,w in enumerate(cases_by_role[role]):
                    budget()
                    ledger,_=oe.fixed_response(model,adapters,agents[role],planner,conditions,w,role,9620000+index,'correct')
                    baseline_scores[role+'|'+w['song']]=ensemble_score(ledger,5760,w['available_roles'])[0]
            (root/'baseline_response_scores.json').write_text(json.dumps(baseline_scores,indent=2))
        if mode=='revised':
            # Engineering probes only. They never choose or label a checkpoint as best.
            for context in ['correct','missing','mismatched']:
                ep=oe.compose(model,adapters,agents,planner,conditions,8,args.seeds[0],mode,partner_mode=context)
                probes.append(dict(context=context,seed=args.seeds[0],engineering_score=ensemble_score(ep['notes'],8*1920)[0]))
            response=[]
            for role in oe.ROLES:
                for index,w in enumerate(cases_by_role[role]):
                    case=dict(role=role,song=w['song'],group=w['group'],split='test',seed=9620000+index,scores={},
                        baseline_correct=baseline_scores.get(role+'|'+w['song']))
                    for context in ['correct','missing','mismatched']:
                        budget()
                        ledger,notes=oe.fixed_response(model,adapters,agents[role],planner,conditions,w,role,case['seed'],context)
                        case['scores'][context]=ensemble_score(ledger,5760,w['available_roles'])[0]
                    response.append(case)
                    (root/'response_probes.json').write_text(json.dumps(response,indent=2))
                    print('Fixed partner probe',role,index+1,case['scores'],flush=True)
        del model,adapters,agents,planner,conditions
        if torch.cuda.is_available():torch.cuda.empty_cache()
    if not items:raise ValueError('No trained stages available')
    random.Random(9619).shuffle(items)
    panel=[dict(clip=i['id'],audio=i['audio']) for i in items]
    panel_hash=hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()
    (root/'panel_identity.json').write_text(json.dumps(dict(sha256=panel_hash)))
    (root/'panel_key.json').write_text(json.dumps(items,indent=2))
    (root/'evaluation.json').write_text(json.dumps(dict(partner_probes=probes,quality_gate='pending listener ratings',cooperation_gate='pending composition-level per-role probes'),indent=2))
    page='''<!doctype html><html><meta charset="utf-8"><title>Ensemble listening</title><style>body{font:18px system-ui;background:#101724;color:#e8edf5;max-width:780px;margin:40px auto;padding:20px}audio{width:100%}button,select,input{font:inherit;margin:8px;padding:8px}label{display:block}</style><h1>Ensemble listening</h1><p>Rate each short piece on its own. 1 = poor; 5 = excellent. These are agent-generated notes. Quality and cooperation are under evaluation.</p><label>Your name or nickname <input id="voter" value="Anonymous"></label><label>Clip <select id="clip"></select></label><audio id="audio" controls></audio><div id="scores"></div><button id="save">Save rating</button><button id="download">Download ratings</button><p id="status"></p><script>const clips=PANEL,ratings=JSON.parse(localStorage.getItem('ensemble-ratings-v1')||'{}'),select=document.getElementById('clip'),audio=document.getElementById('audio');clips.forEach((c,i)=>select.add(new Option('Piece '+(i+1),c.clip)));['enjoyment','development','ensemble_fit'].forEach(k=>document.getElementById('scores').insertAdjacentHTML('beforeend',`<label>${k.replace('_',' ')} <select id="${k}">${[1,2,3,4,5].map(n=>`<option>${n}</option>`).join('')}</select></label>`));function load(){audio.src=clips.find(c=>c.clip===select.value).audio;['enjoyment','development','ensemble_fit'].forEach(k=>document.getElementById(k).value=ratings[select.value]?.[k]||3)}select.onchange=load;load();document.getElementById('save').onclick=()=>{ratings[select.value]=Object.fromEntries(['enjoyment','development','ensemble_fit'].map(k=>[k,+document.getElementById(k).value]));localStorage.setItem('ensemble-ratings-v1',JSON.stringify(ratings));document.getElementById('status').textContent=Object.keys(ratings).length+'/'+clips.length+' rated'};document.getElementById('download').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({version:1,voter:document.getElementById('voter').value,ratings},null,2)],{type:'application/json'}));a.download='ensemble-ratings.json';a.click()};</script></html>'''
    page=page.replace('PANEL',json.dumps(panel)).replace('ensemble-ratings-v1','ensemble-ratings-'+panel_hash)
    page=page.replace('version:1,voter:', 'version:1,panel_sha256:'+json.dumps(panel_hash)+',voter:')
    (root/'listen.html').write_text(page,encoding='utf-8')
    budget()
    print(f'Created {len(items)} clips; listener evaluation pending.',flush=True)

if __name__=='__main__':main()
