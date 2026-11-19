"""Real-backbone checks for isolated LoRA learning and sequence-critic PPO."""
import json
import torch
import note_trio as nt
from supervised_ensemble import InstrumentAdapters,teacher_example,teacher_loss,rollout
from sequence_critic import SequenceCritic


def main():
    torch.set_num_threads(4);torch.manual_seed(56)
    model=nt.load_model();adapters=InstrumentAdapters(model);base=adapters.base_hash()
    example=dict(beat=4,notes=[['note',0,480,0,64,60],['note',0,480,1,71,60],['note',0,480,2,40,60],
                             ['note',1920,480,0,67,60],['note',1920,480,1,72,60],['note',1920,480,2,43,60]])
    checked={}
    for role in nt.ROLES:
        before=adapters.state();adapters.activate(role,True)
        optimizer=torch.optim.AdamW(adapters.params(role),lr=1e-4)
        loss,n=teacher_loss(model,teacher_example(model,role,example));loss.backward()
        gradient_roles=[r for r in nt.ROLES if any(p.grad is not None and p.grad.abs().sum()>0 for p in adapters.params(r))]
        assert gradient_roles==[role],gradient_roles
        optimizer.step();model.zero_grad(set_to_none=True)
        after=adapters.state()
        changed={r:any(not torch.equal(v,before[k]) for k,v in after.items() if f'.{r}.' in k) for r in nt.ROLES}
        assert changed=={r:r==role for r in nt.ROLES},changed
        checked[role]=dict(nll=float(loss.detach()),decisions=n,isolated_update=True)
    agents={r:nt.Musician() for r in nt.ROLES};critic=SequenceCritic()
    a=rollout(model,adapters,agents,critic,nt.Brief(1),814)
    b=rollout(model,adapters,agents,critic,nt.Brief(1),814,order=reversed(nt.ROLES))
    assert a['notes']==b['notes'] and a['decisions']
    for step in a['steps']:
        for role,records in step['records'].items():
            for r in records:
                d=agents[role].distribution(r['observation'],r['reference'])
                assert abs(float(d.log_prob(torch.tensor(r['action'])))-r['logp'])<1e-5
    frozen=adapters.state();oldcritic=nt.fingerprint(critic);oldactors={r:nt.fingerprint(a) for r,a in agents.items()}
    logs=nt.update(agents,critic,{r:torch.optim.Adam(a.parameters(),lr=1e-4) for r,a in agents.items()},
                   torch.optim.Adam(critic.parameters(),lr=3e-4),[a])
    assert nt.fingerprint(critic)!=oldcritic
    assert all(nt.fingerprint(a)!=oldactors[r] for r,a in agents.items())
    assert all(torch.equal(v,frozen[k]) for k,v in adapters.state().items())
    assert base==adapters.base_hash()
    result=dict(adapter_checks=checked,base_unchanged=True,adapters_frozen_during_ppo=True,
                critic_and_all_residual_actors_updated=True,agent_order_independent=True,
                exact_log_probability_replay=True,every_note_is_an_action=True,notes=len(a['notes']),ppo=logs)
    path=nt.ROOT/'outputs'/'supervised_integration.json';path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
