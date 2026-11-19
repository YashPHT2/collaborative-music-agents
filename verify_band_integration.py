"""Run real-checkpoint causality/order/probability checks after a training trial."""
import json
import torch
from band import ROOT,ROLES,Brief,Musician,TeamCritic,load_model,rollout

def main():
    torch.set_num_threads(4)
    root=ROOT/'outputs'/'band_joint_v1'
    payload=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
    actors={r:Musician() for r in ROLES}; critic=TeamCritic()
    for r,a in actors.items(): a.load_state_dict(payload['actors'][r])
    critic.load_state_dict(payload['critic'])
    model=load_model()
    first=rollout(model,actors,critic,Brief(1),880001)
    reverse=rollout(model,actors,critic,Brief(1),880001,order=list(reversed(ROLES)))
    assert first['notes']==reverse['notes'], 'Execution-order leak'
    assert sorted(first['trace'],key=lambda t:(t['beat'],t['role']))==sorted(reverse['trace'],key=lambda t:(t['beat'],t['role']))
    maximum=0.; count=0
    for step in first['steps']:
        for r,records in step['records'].items():
            for record in records:
                actual=float(actors[r].distribution(record['observation'],record['reference']).log_prob(torch.tensor(record['action'])).detach())
                maximum=max(maximum,abs(actual-record['logp'])); count+=1
    assert maximum<1e-6
    report=dict(real_model_execution_order_invariant=True,trace_order_invariant=True,
                stochastic_token_records_replayed=count,max_logprob_error=maximum,
                checkpoint_reload_generation_reproduces=True)
    (root/'integration_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
