"""Run a bounded PPO pilot and compare it with matched and original baselines."""
import argparse
import copy
import csv
import json
import random
import statistics
import time

from accompany import ROOT, MIDIModel, torch, EventSampler, accompany, assemble, write_result
from pilot_environment import synthetic_episode, rollout, window_reward
from ppo_adapter import ResidualPolicy, ppo_update
from music_metrics import diagnostics
from evaluate_baseline import validate_midi, validate_audio


def evaluate_proxy(lead,notes,end):
    rewards = [window_reward(n,notes,n[1],lead[i+1][1] if i+1<len(lead) else end)
               for i,n in enumerate(lead)]
    return statistics.mean(rewards)


def fingerprint(model):
    # Verify every base tensor remains unchanged; no gradients are enabled on it.
    return [(name,float(parameter.float().sum().item()),float(parameter.float().square().sum().item()))
            for name,parameter in model.named_parameters()]


def listening_page(root,rows):
    # Independent labels for this experiment. The old ballot labels do not apply.
    comparisons = []
    for case in sorted({r['case'] for r in rows}):
        a,b = ['matched_initial','ppo_adapter']
        if random.Random(case).random()<0.5:
            a,b=b,a
        comparisons.append(dict(case=case,a=f'{case}/{a}.mp3',b=f'{case}/{b}.mp3',mapping=dict(A=a,B=b)))
    data = json.dumps(comparisons)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,"><title>PPO pilot comparison</title><style>
body{font:16px/1.6 system-ui;background:#111827;color:#e5e7eb;max-width:900px;margin:32px auto;padding:20px}
section{background:#1f2937;padding:20px;margin:20px 0;border-radius:12px}audio{display:block;width:100%;margin:12px 0}
button{padding:10px;margin:5px;background:#334155;color:white;border:1px solid #64748b;border-radius:6px}
button[aria-pressed=true]{background:#047857}.mapping{color:#fbbf24}</style>
<h1>First PPO pilot: does it sound better?</h1><p>Compare the same melody with a matched untrained adapter and the PPO-trained adapter.
Both use the same frozen SkyTNT checkpoint. A change occurs at 9.6 seconds in the change cases.
Listen to the whole piece: the adapter can change both halves. This is a small technical pilot, not an established quality improvement.</p>
<p><label for="mode">Listen to: </label><select id="mode"><option value="duet">Full duet</option><option value="solo">Accompaniment only</option></select></p>
<button id="export">Download votes</button><button id="reveal">Reveal labels</button><div id="pairs"></div>
<script>const pairs=__DATA__;let votes={};
for(const pair of pairs){const card=document.createElement('section');const heading=document.createElement('h2');heading.textContent=({fresh_none:'Steady melody',fresh_pitch:'Key change',fresh_rhythm:'Rhythm change',fresh_both:'Key and rhythm change'})[pair.case];card.append(heading);
for(const label of ['A','B']){const name=document.createElement('b');name.textContent=label;card.append(name);
const audio=document.createElement('audio');audio.controls=true;audio.preload='metadata';audio.src=pair[label.toLowerCase()];audio.dataset.duet=pair[label.toLowerCase()];audio.setAttribute('aria-label',pair.case+' '+label);
audio.onplay=()=>document.querySelectorAll('audio').forEach(other=>{if(other!==audio)other.pause()});card.append(audio);
const skip=document.createElement('button');skip.textContent='Start at change';skip.onclick=async()=>{try{if(audio.readyState<1)await new Promise((resolve,reject)=>{audio.addEventListener('loadedmetadata',resolve,{once:true});audio.addEventListener('error',reject,{once:true});audio.load()});audio.currentTime=9.6;await audio.play()}catch{skip.textContent='Use audio controls'}};card.append(skip)}
const buttons=[];for(const choice of ['A','B','Tie','Neither good']){const button=document.createElement('button');button.textContent=choice;button.onclick=()=>{votes[pair.case]=choice;buttons.forEach(b=>b.setAttribute('aria-pressed',b===button))};buttons.push(button);card.append(button)}
const mapping=document.createElement('p');mapping.hidden=true;mapping.className='mapping';mapping.textContent='A: '+pair.mapping.A+' · B: '+pair.mapping.B;card.append(mapping);document.getElementById('pairs').append(card)}
document.getElementById('reveal').onclick=()=>document.querySelectorAll('.mapping').forEach(e=>e.hidden=false);
document.getElementById('mode').onchange=event=>document.querySelectorAll('audio').forEach(audio=>{audio.pause();audio.src=event.target.value==='solo'?audio.dataset.duet.replace('.mp3','_solo.mp3'):audio.dataset.duet;audio.load()});
document.getElementById('export').onclick=()=>{const result={experiment:'ppo-pilot-v1',created_at:new Date().toISOString(),ratings:pairs.map(p=>({case:p.case,vote:votes[p.case]||null,mapping:p.mapping}))};const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='ppo-pilot-votes.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></html>'''.replace('__DATA__',data)
    (root/'listen.html').write_text(page,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates',type=int,default=8)
    parser.add_argument('--episodes-per-update',type=int,default=4)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    args = parser.parse_args()
    if args.updates<1 or args.episodes_per_update<2 or args.updates*args.episodes_per_update>500:
        parser.error('Use positive updates, at least 2 episodes per update, at most 500 training episodes.')
    root = ROOT/'outputs'/'ppo_pilot_v1'
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    policy = ResidualPolicy()
    initial = copy.deepcopy(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(),lr=0.001)
    device = args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
    model = MIDIModel.from_pretrained(ROOT/'models'/'skytnt-tv2o-medium',dtype=dtype,local_files_only=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    original_fingerprint = fingerprint(model)
    update_rng = torch.Generator().manual_seed(args.seed)
    validation = [synthetic_episode(2000+i,condition=c) for i,c in enumerate(['none','pitch','rhythm','both'])]
    def validation_score():
        scores = []
        for i,(lead,_,_) in enumerate(validation):
            _,_,rewards,_ = rollout(model,policy,lead,800+i)
            scores.append(statistics.mean(rewards))
        return statistics.mean(scores)
    started = time.perf_counter()
    baseline_validation = validation_score()
    best_score,best_state,best_update = baseline_validation,copy.deepcopy(initial),0
    print(f'Initial validation proxy: {baseline_validation:.4f}',flush=True)
    training_log, corpus = [],[]
    for update in range(args.updates):
        episodes,rewards = [],[]
        for episode in range(args.episodes_per_update):
            index = update*args.episodes_per_update+episode
            corpus_seed = 1000+index
            # This bound keeps seed namespaces disjoint even for custom budgets.
            lead,harmony,condition = synthetic_episode(corpus_seed)
            corpus.append(dict(split='train',seed=corpus_seed,condition=condition,melody=lead,harmony=harmony))
            _,trajectory,episode_rewards,_ = rollout(model,policy,lead,10000+index,train=True)
            episodes.append(trajectory)
            rewards.append(statistics.mean(episode_rewards))
        stats = ppo_update(policy,optimizer,episodes,update_rng)
        record = dict(update=update+1,train_mean_proxy=statistics.mean(rewards),**stats)
        if (update+1)%2==0 or update+1==args.updates:
            score = validation_score()
            record['validation_proxy'] = score
            if score>best_score:
                best_score,best_state,best_update = score,copy.deepcopy(policy.state_dict()),update+1
        training_log.append(record)
        (root/'training_log.json').write_text(json.dumps(training_log,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    final_state = copy.deepcopy(policy.state_dict())
    torch.save(dict(state_dict=final_state,seed=args.seed,updates=args.updates),root/'adapter_final.pt')
    torch.save(dict(state_dict=best_state,seed=args.seed,selected_update=best_update),root/'adapter_selected.pt')
    policy.load_state_dict(best_state)
    reference_policy = ResidualPolicy()
    reference_policy.load_state_dict(initial)
    rows, verifications = [],{}
    for i,condition in enumerate(['none','pitch','rhythm','both']):
        lead,harmony,_ = synthetic_episode(3000+i,bars=8,condition=condition)
        corpus.append(dict(split='fresh_final_test',seed=3000+i,condition=condition,melody=lead,harmony=harmony))
        case=f'fresh_{condition}'
        directory=root/case
        directory.mkdir(exist_ok=True)
        print(f'Evaluating {case}',flush=True)
        seed=900+i
        original_notes,_ = accompany(EventSampler(model,seed,100,0.9),lead,verbose=False)
        from accompany import resolve_overlaps
        original_notes=resolve_overlaps(original_notes)
        initial_notes,_,_,_ = rollout(model,reference_policy,lead,seed)
        trained_notes,_,_,_ = rollout(model,policy,lead,seed)
        # Reranking baseline: same untrained adapter, three independent samples;
        # select using the pilot proxy. Costs 3x sampling, no training.
        candidates=[initial_notes]+[rollout(model,reference_policy,lead,seed+100*k)[0] for k in [1,2]]
        candidate_scores=[evaluate_proxy(lead,notes,15360) for notes in candidates]
        reranked=candidates[max(range(3),key=lambda k:candidate_scores[k])]
        for method,notes in [('original_sampler',original_notes),('matched_initial',initial_notes),
                             ('ppo_adapter',trained_notes),('rerank_3',reranked)]:
            path=write_result(directory,method,assemble(lead,notes,100),True)
            (directory/f'{method}_notes.json').write_text(json.dumps(notes)+'\n')
            verifications[f'{case}/{method}']=dict(**validate_midi(path,lead,notes),audio=validate_audio(path.with_suffix('.wav')))
            rows.append(dict(case=case,method=method,proxy=evaluate_proxy(lead,notes,15360),
                             **diagnostics(lead,notes,harmony,0,15360)))
    for i,(lead,harmony,condition) in enumerate(validation):
        corpus.append(dict(split='validation',seed=2000+i,condition=condition,melody=lead,harmony=harmony))
    unchanged=fingerprint(model)==original_fingerprint
    assert unchanged and all(p.grad is None for p in model.parameters()),'Frozen model changed'
    changed_parameters=sum(not torch.equal(initial[key],final_state[key]) for key in initial)
    assert changed_parameters>0,'Adapter did not train'
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest=dict(experiment='ppo-pilot-v1',training_seed=args.seed,training_updates=args.updates,
                  training_episodes=args.updates*args.episodes_per_update,
                  trainable_parameters=sum(p.numel() for p in policy.parameters()),
                  base_parameters=sum(p.numel() for p in model.parameters()),base_unchanged=unchanged,
                  changed_adapter_tensors=changed_parameters,selected_update=best_update,
                  initial_validation_proxy=baseline_validation,selected_validation_proxy=best_score,
                  device=device,dtype=str(dtype),elapsed_seconds=time.perf_counter()-started,
                  policy_initialization='Zero residual on frozen base distribution. Event tokens have fixed 2.5% uniform WAIT exploration; matched baseline uses same floor.',
                  algorithm=dict(name='PPO clipped surrogate',clip=0.2,gamma=0.99,gae_lambda=0.95,epochs=4,lr=0.001,reference_kl_coefficient=0.05,entropy_coefficient=0.005),
                  checkpoint='skytnt/midi-model-tv2o-medium',revision='0f8f265d4330f4e46527ac2313200254c5757f5f',
                  reward='0.5 coverage + 0.4 current-melody interval support -0.25 excess dominant-pitch share -0.2 excess density; empty window -0.6',
                  listener_votes_used_as_training_labels=False,scope='Single accompaniment output adapter, not MARL or LoRA/base-weight finetuning',
                  final_test_used_for_selection=False,human_quality_improvement_established=False)
    for name,data in [('manifest.json',manifest),('corpus.json',corpus),('verification.json',verifications)]:
        (root/name).write_text(json.dumps(data,indent=2)+'\n')
    listening_page(root,rows)
    lines=['# First PPO output-adapter pilot','',
           f"Trained {manifest['trainable_parameters']:,} adapter parameters over the frozen {manifest['base_parameters']:,}-parameter SkyTNT model.",
           f"{manifest['training_episodes']} four-bar training episodes; one training seed. Selected update {best_update} on four validation melodies.",
           '','Your earlier ballot: responding 2, frozen 2, ties 2, neither good 1. It does not establish a preferred method. '
           'Votes were recorded, not converted into invented preference labels. Theme B has now been inspected; fresh synthetic final-test melodies are used here.',
           '', '## Fresh final-test results','',
           '| Case | Method | Proxy | Rough intervals | Coverage | Notes/beat |',
           '|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['case']} | {row['method']} | {row['proxy']:.3f} | {row['rough_melody_interval_fraction']} | {row['sounding_coverage']} | {row['notes_per_beat']} |")
    means={method:statistics.mean(r['proxy'] for r in rows if r['method']==method) for method in {r['method'] for r in rows}}
    lines += ['',f'Mean heuristic proxy: {json.dumps(means,sort_keys=True)}','',
              '## What this establishes','',
              'PPO updates a tiny residual categorical policy for note-vs-wait and pitch decisions. It samples on-policy, '
              'stores actual masked sampling probabilities, computes complete-episode GAE, clips likelihood ratios, '
              'uses a value function, gradient clipping, and reference KL. The pretrained generator is verified unchanged. '
              'The other timing, velocity and duration tokens remain frozen-model choices. There are no trainable melody agents.',
              '', 'The original sampler and matched untrained adapter are both included. The matched adapter has the '
              'same event exploration floor and pitch support as PPO; this controls for sampler changes. Rerank-3 uses '
              'three untrained candidates and the same proxy, at three times the inference cost. It is not compute matched.',
              '', '## Limitations','',
              'The reward is hand-designed and unvalidated against listener preferences. It can discourage useful musical '
              'tension and rests. Repetition, silence and flooding fixtures are checked, but this does not rule out reward hacking. '
              'The critic discounts token decisions, so musical timing and decision counts are not equivalent. Scheduler '
              'boundary assistance and overlap cleanup are inherited from the previous baseline. Online train/validation '
              'rewards use the committed raw note history available at each input window; final-test proxy scores use '
              'the exported overlap-cleaned notes. Future duplicate-pitch attacks can shorten earlier notes during cleanup, '
              'so these scores need not match exactly. The reward sees only '
              'the current melody pitch; final chord/scale diagnostics use the scripted reference and can penalize valid alternatives.',
              '', 'Training/validation/final-test seed namespaces are disjoint, but all melodies share one small synthetic '
              'generator. This is an engineering generalization check, not external musical evaluation. Four final tests '
              'and one training seed cannot establish robustness or significance. All checkpoints, including the final '
              'update, are kept; update selection uses the proxy on validation only, never the final-test set.',
              '', '**A higher proxy is not evidence of better-sounding music.** Open `listen.html` and judge the new '
              'A/B comparisons before revealing labels. If neither sounds good, change the objective or environment '
              'rather than extending training blindly. No claim of human-quality improvement is made.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    from prepare_pilot_listening import prepare
    prepare(root)
    print(json.dumps(manifest,indent=2),flush=True)
    print(root/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
