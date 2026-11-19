"""Fresh-seed short demo, matched old-model control and partner diagnostics."""
import json
import argparse
from pathlib import Path
import torch
import ensemble_repair as er
import note_trio as nt
from train_ensemble_repair import corpus,write_json
from partner_response import ensemble_score
from phrase_curriculum import melodic_features

ROOT=nt.ROOT/'outputs'/'phrase_ensemble_v1'
SEED=9860407


def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    state=json.loads((ROOT/'training_log.json').read_text())
    initial=torch.load(ROOT/'initial.pt',map_location='cpu',weights_only=False)
    if not state.get('complete') and not args.preview:raise ValueError('Training must finish before evaluating')
    selected=ROOT/'latest.pt'
    has_update=any(v['passed'] for v in state['selections'].values())
    if args.preview:
        if not state['selections'].get('participation',{}).get('passed'):raise ValueError('No validated phrase-stage checkpoint yet')
        selected=ROOT/'participation.pt';has_update=True
        ROOT=nt.ROOT/'outputs'/'phrase_preview_v1';ROOT.mkdir(parents=True,exist_ok=True)
    if not has_update:
        selected=ROOT/'revision_candidate.pt'
        if not selected.exists():selected=ROOT/'participation_candidate.pt'
    trained=torch.load(selected,map_location='cpu',weights_only=False)
    model,adapters,agents,planner,conditions=er.load(selected)
    results={};episodes={}
    mode=trained.get('composition_mode','planned') if has_update else 'revised'
    def restore(saved):
        adapters.load(saved['adapters']);planner.load_state_dict(saved['planner']);conditions.load_state_dict(saved['conditions'])
    for label,saved in [('earlier',initial),('phrase',trained)]:
        restore(saved)
        ep=er.compose(model,adapters,agents,planner,conditions,6,SEED,mode)
        ep['tempo_map']=[(b*1920,86+24*b/5) for b in range(6)]
        ep['room_effect']=True
        checks=nt.export(ROOT,label,ep,allow_silent_stems=True)
        write_json(ROOT/f'{label}_episode.json',ep)
        results[label]=dict(participation=er.participation(ep['notes'],6),audio=checks,
                            melody_analysis={r:melodic_features(ep['notes'],i) for i,r in enumerate(er.ROLES)})
        episodes[label]=ep
        print('Rendered',label,{r:p['notes'] for r,p in results[label]['participation'].items()},flush=True)
    _,groups,_,_=corpus();probes=[];restore(trained)
    for role in ([] if args.preview else er.ROLES):
        eligible={g:ws for g,ws in groups['test'][role].items() if any(len(w['available_roles'])>=2 for w in ws)}
        for index,group in enumerate(sorted(eligible)[:3]):
            w=next(w for w in eligible[group] if len(w['available_roles'])>=2)
            row=dict(role=role,group=group,scores={})
            for context in ['correct','missing','mismatched']:
                ledger,notes=er.fixed_response(model,adapters,agents,planner,conditions,w,role,986100+index,context)
                row['scores'][context]=ensemble_score(ledger,7680,w['available_roles'])[0]
            probes.append(row);write_json(ROOT/'partner_probes.json',probes)
            print('Partner diagnostic',role,index+1,flush=True)
    response={r:{c:sum(p['scores'][c] for p in probes if p['role']==r)/max(1,sum(p['role']==r for p in probes)) for c in ['correct','missing','mismatched']} for r in er.ROLES}
    changed=episodes['earlier']['notes']!=episodes['phrase']['notes']
    earlier_structure={tuple(n[1:5]) for n in episodes['earlier']['notes']}
    trained_structure={tuple(n[1:5]) for n in episodes['phrase']['notes']}
    overlap=len(earlier_structure&trained_structure)/max(1,len(earlier_structure|trained_structure))
    report=dict(checkpoint=str(selected),selected_training_passed=has_update,selections=state['selections'],seed=SEED,
                comparison_same_generation_mode=mode,note_sequences_differ=changed,structural_note_set_jaccard=overlap,results=results,partner_response=response,
                reference_audio_used_for_training=False,references='High-level goals: recurring melody, ensemble support, changing energy',
                human_quality='unassessed',preview=args.preview,cooperation='Pending: partner-response training still running' if args.preview else 'Three test compositions per role: descriptive diagnostics, not established improvement',
                explicit_production='Same tempo map, room echo and mastering for old and new notes; no melody or chord insertion')
    write_json(ROOT/'demo_verification.json',report)
    status='The checkpoint passed the imitation validation gate.' if has_update else 'No checkpoint passed the validation gate. This is an unselected trained candidate for listening.'
    if args.preview:status+=' Preview of the completed four-bar phrase stage; partner-response training continues separately.'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phrase-trained trio</title><style>body{font:19px system-ui;background:#101724;color:#e8edf5;max-width:900px;margin:40px auto;padding:20px}p{line-height:1.6}section{background:#1c293d;padding:24px;margin:20px 0;border-radius:14px}audio{width:100%}select,button{font:inherit;padding:8px;margin:6px}a{color:#9bd2ff}</style><h1>Phrase-trained trio — short demo</h1><p>New four-bar phrase training, melodic-contour loss and matched partner examples. Original organ, strings and cello notes from the agents. Same fresh seed and sound processing for both versions.</p><p>__STATUS__</p><section><h2>New phrase-trained model</h2><select id="part"><option value="mix">All three agents</option><option value="organ">Organ</option><option value="strings">Strings</option><option value="cello">Cello</option></select><audio id="new" controls src="phrase_mix.mp3"></audio></section><section><h2>Earlier model — same seed and settings</h2><audio id="old" controls src="earlier_mix.mp3"></audio></section><p>The recordings you shared provide musical direction. Neither recording nor its notes was used as training data. This demo does not establish similarity to a professional soundtrack or a listener preference gain. Tempo and room effects are explicit performance controls.</p><p><a href="phrase_mix.mid">Download new MIDI</a> · <a href="demo_verification.json">Results and provenance</a></p><script>const a=document.getElementById('new');document.getElementById('part').onchange=e=>{a.pause();a.src='phrase_'+e.target.value+'.mp3';a.load()};document.querySelectorAll('audio').forEach(x=>x.onplay=()=>document.querySelectorAll('audio').forEach(y=>{if(x!==y)y.pause()}));</script></html>'''
    (ROOT/'listen.html').write_text(page.replace('__STATUS__',status),encoding='utf-8')
    print('Phrase demo ready:',ROOT/'listen.html',flush=True)


if __name__=='__main__':main()
