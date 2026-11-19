"""Bounded reward-v2 training continuation, preserving the previous checkpoints."""
import copy,json,time
import torch
from band import ROOT,ROLES,Brief,TeamCritic,load_model,rollout,update,fingerprint
from train_band import make_policies,save_checkpoint

def main():
    torch.set_num_threads(4)
    root=ROOT/'outputs'/'band_reward_v2'; root.mkdir(parents=True,exist_ok=True)
    if (root/'latest.pt').exists(): raise ValueError('Experiment exists; do not overwrite')
    source=ROOT/'outputs'/'band_joint_v2'/'latest.pt'
    payload=torch.load(source,map_location='cpu',weights_only=False)
    actors=make_policies(); critic=TeamCritic()
    optimizers={r:torch.optim.Adam(a.parameters(),lr=1e-4) for r,a in actors.items()}
    co=torch.optim.Adam(critic.parameters(),lr=3e-4)
    for r,a in actors.items():
        a.load_state_dict(payload['actors'][r]); optimizers[r].load_state_dict(payload['optimizers'][r])
    # Refit the value estimator for the new reward scale; preserve actor momentum.
    torch.manual_seed(20261018)
    critic=TeamCritic(); co=torch.optim.Adam(critic.parameters(),lr=3e-4)
    model=load_model(); base=fingerprint(model); before={r:fingerprint(a) for r,a in actors.items()}
    logs=[]; started=time.perf_counter()
    for u in range(4):
        episodes=[]
        for e in range(4):
            ep=rollout(model,actors,critic,Brief(4,100,[0,9,5,2][e],bool(e%2),'band-heuristic-v2'),30000001+u*4+e)
            episodes.append(ep)
            print(f'Reward v2 update {u+1}/4 episode {e+1}/4',flush=True)
        learning=update(actors,critic,optimizers,co,episodes)
        logs.append(dict(update=15+u,episodes=4,reward_version='band-heuristic-v2',
                         mean_return=sum(sum(s['reward'] for s in ep['steps']) for ep in episodes)/4,learning=learning))
        save_checkpoint(root/'latest.pt',actors,critic,optimizers,co,payload['initial'],15+u,payload['seed'],payload['logs']+logs,time.perf_counter()-started)
        (root/'training_log.json').write_text(json.dumps(logs,indent=2))
        print(json.dumps(logs[-1]),flush=True)
    result=dict(reward_version='band-heuristic-v2',source=str(source),additional_episodes=16,total_episodes=72,
                actors_changed={r:before[r]!=fingerprint(a) for r,a in actors.items()},backbone_unchanged=base==fingerprint(model),
                learned_reward_model=False,human_preference_improvement_established=False)
    assert all(result['actors_changed'].values()) and result['backbone_unchanged']
    (root/'manifest.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
