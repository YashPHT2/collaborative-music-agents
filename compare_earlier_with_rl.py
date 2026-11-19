"""Earlier v2 model versus its own PPO branch: same seeds and same sampler."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
import offline_ensemble as oe
import note_trio as nt
from sequence_critic import SequenceCritic
from joint_ensemble_rl import OUT as OTHER,rollout,replay_check,metrics
from train_ensemble_repair import write_json

ROOT=nt.ROOT/'outputs'/'earlier_model_joint_rl_v1'
SEEDS=[9619102,9619101,9619103]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT);args=parser.parse_args();root=args.output
    state=json.loads((root/'training_log.json').read_text())
    if not state.get('complete') and not (state['step']==state['total'] and (root/'latest.pt').exists()):
        raise ValueError('All RL training updates must finish first')
    candidate_file='selected.pt' if state.get('selected_step',0)>0 else 'latest.pt'
    candidate_step=state.get('selected_step',0) or state['step']
    checkpoint=torch.load(root/candidate_file,map_location='cpu',weights_only=False)
    source=Path(checkpoint['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=checkpoint['source_sha256']:raise ValueError('Source changed')
    torch.set_num_threads(4)
    model,adapters,agents,planner,conditions=oe.load(source)
    critic=SequenceCritic();critic.load_state_dict(checkpoint['critic']);items=[];results=[]
    # The reference and candidate use the exact same v2 sampler and seed formula.
    for mode in ('earlier','rl'):
        if mode=='rl':
            for role in oe.ROLES:agents[role].load_state_dict(checkpoint['policies'][role])
        for seed in SEEDS:
            ep=rollout(model,adapters,agents,planner,conditions,critic,8,seed)
            replay=replay_check(agents,ep)
            if mode=='earlier':
                reference=nt.ROOT/'outputs'/'offline_ensemble_v1'/f'revised_{seed}_events.json'
                if ep['notes']!=json.loads(reference.read_text()):raise ValueError('Earlier performance did not reproduce exactly')
            name=f'{mode}_{seed}'
            checks=nt.export(root,name,ep,stems=seed==SEEDS[0],allow_silent_stems=True)
            results.append(dict(mode=mode,seed=seed,metrics=metrics([ep]),replay_error=replay,earlier_exact_reproduction=mode=='earlier'))
            items.append(dict(id=name,seed=seed,mode=mode,audio=f'{name}_mix.mp3',audio_sha256=hashlib.sha256((root/f'{name}_mix.mp3').read_bytes()).hexdigest(),checks=checks))
            print('Rendered',name,flush=True)
    pairs=[];rng=random.Random(986001)
    for seed in SEEDS:
        pair=[i for i in items if i['seed']==seed];rng.shuffle(pair)
        pairs.append(dict(seed=seed,A=pair[0]['id'],B=pair[1]['id']))
    identity=hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()
    write_json(root/'matched_key.json',dict(items=items,pairs=pairs,panel_sha256=identity,selected_update=state.get('selected_step',0),
                   compared_update=candidate_step,compared_checkpoint=candidate_file,validation_gate_passed=state.get('selected_step',0)>0))
    write_json(root/'matched_metrics.json',results)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Earlier model vs RL</title>
<style>body{font:18px system-ui;background:#101724;color:#e8edf5;max-width:1000px;margin:35px auto;padding:20px}p{line-height:1.6}.pair{background:#1c293d;padding:22px;border-radius:14px;margin:20px 0}.players{display:grid;grid-template-columns:1fr 1fr;gap:22px}audio{width:100%}button,select{font:inherit;padding:9px;margin:5px}a{color:#9bd2ff}@media(max-width:650px){.players{grid-template-columns:1fr}}</style>
<h1>Earlier model vs its RL branch</h1><p>Three matched pairs: same starting model, seed, instruments, tempo, length and generation method. The first pair uses the seed for your favourite earlier Piece 4. Choose whichever sounds better, or Same / Neither. A/B identities stay hidden until revealed.</p><p id="selection"></p><div id="pairs"></div><button id="reveal">Reveal versions</button><button id="download">Download votes</button><p id="status"></p><script>
const pairs=__PAIRS__,key=__KEY__,panel='__HASH__',storage='earlier-v-rl-'+panel,votes=JSON.parse(localStorage.getItem(storage)||'{}');document.getElementById('selection').textContent=__SELECTION__;
pairs.forEach((p,i)=>{const card=document.createElement('section');card.className='pair';const h=document.createElement('h2');h.textContent='Pair '+(i+1)+(i===0?' — favourite piece’s seed':'');card.append(h);const players=document.createElement('div');players.className='players';['A','B'].forEach(label=>{const box=document.createElement('div'),title=document.createElement('b');title.textContent=label;box.append(title);const audio=document.createElement('audio');audio.controls=true;audio.preload='metadata';audio.src=p[label]+'_mix.mp3';audio.onplay=()=>document.querySelectorAll('audio').forEach(other=>{if(other!==audio)other.pause()});box.append(audio);const play=document.createElement('button');play.textContent='Play '+label+' from start';play.onclick=()=>{audio.currentTime=0;audio.play().catch(()=>{})};box.append(play);players.append(box)});card.append(players);const choices=document.createElement('div');['A','B','Same','Neither'].forEach(choice=>{const b=document.createElement('button');b.textContent=choice;b.style.border=votes[p.seed]===choice?'3px solid #4ac9a0':'';b.onclick=()=>{votes[p.seed]=choice;localStorage.setItem(storage,JSON.stringify(votes));[...choices.children].forEach(child=>child.style.border=child===b?'3px solid #4ac9a0':'');document.getElementById('status').textContent=Object.keys(votes).length+'/3 pairs voted'};choices.append(b)});card.append(choices);const mapping=document.createElement('p');mapping.className='mapping';mapping.hidden=true;mapping.textContent=['A','B'].map(label=>label+' = '+(key.find(k=>k.id===p[label]).mode==='earlier'?'Earlier model':'RL branch')).join(' · ');card.append(mapping);document.getElementById('pairs').append(card)});
document.getElementById('reveal').onclick=()=>document.querySelectorAll('.mapping').forEach(e=>e.hidden=false);document.getElementById('download').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({panel_sha256:panel,votes},null,2)],{type:'application/json'}));a.download='earlier-vs-rl-votes.json';a.click()};</script></html>'''
    selection=f'Compared RL update: {candidate_step}. '+('This checkpoint passed engineering validation; listener preference remains untested.' if state.get('selected_step',0)>0 else 'No update passed the validation gate. This is the final trained candidate for listening, not an accepted improvement.')
    page=page.replace('__PAIRS__',json.dumps(pairs)).replace('__KEY__',json.dumps(items)).replace('__HASH__',identity).replace('__SELECTION__',json.dumps(selection))
    (root/'matched.html').write_text(page,encoding='utf-8')
    print('Matched comparison ready:',root/'matched.html',flush=True)


if __name__=='__main__':main()
