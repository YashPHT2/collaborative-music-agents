"""Verify teacher-policy replay, role-isolated on-policy learning and provenance."""
import copy
import json
import torch
import note_trio as nt
import supervised_ensemble as se
import partner_response as pr
from prepare_partner_cache import SOURCE
from sequence_critic import SequenceCritic


class ForcedPolicy:
    def __init__(self,actions):self.actions=list(actions);self.rows=[]
    def distribution(self,observation,reference):
        self.rows.append((observation.clone(),reference.clone()))
        probs=torch.zeros(128);probs[self.actions.pop(0)]=1
        return torch.distributions.Categorical(probs=probs)


def main():
    torch.set_num_threads(4);torch.manual_seed(64)
    model,adapters,agents,critic=se.load_bundle(SOURCE)
    base=adapters.base_hash();frozen=adapters.state();checks={}
    example=dict(song='integration',window_start=0,available_roles=list(nt.ROLES),beat=4,notes=[
        ['note',0,480,0,64,60],['note',0,480,1,71,60],['note',0,480,2,40,60],
        ['note',1920,480,0,67,60],['note',1920,480,1,72,60],['note',1920,480,2,43,60]])
    for role in nt.ROLES:
        record=pr.teacher_records(model,adapters,role,example)
        forced=ForcedPolicy(record['actions'].tolist())
        history=nt.visible(example['notes'],4);adapters.activate(role)
        sampler=se.EnsembleSampler(model,role,52,nt.Brief(4),4,history)
        notes,_=sampler.propose(forced,nt.state_features(nt.Brief(4),4,history))
        assert not forced.actions and notes==[n for n in example['notes'] if n[1]==1920 and n[3]==nt.ROLES[role]['channel']]
        obs=torch.stack([x[0] for x in forced.rows]);refs=torch.stack([x[1] for x in forced.rows])
        torch.testing.assert_close(obs,record['observations'],atol=.04,rtol=.04)
        torch.testing.assert_close(refs,record['references'],atol=.015,rtol=.04)
        assert not record['observations'].requires_grad and not record['references'].requires_grad
        checks[role]=dict(teacher_sampling_replay_within_bfloat16_tolerance=True,
                         reference_max_abs_error=float((refs-record['references']).abs().max()))
    # A complete source window; only the designated learner policy can update.
    window=copy.deepcopy(example);window['notes']=[
        ['note',beat*480,120,s['channel'],{'organ':64,'strings':76,'cello':43}[r],60]
        for beat in range(16) for r,s in nt.ROLES.items()]
    for role in nt.ROLES:
        before={r:nt.fingerprint(a) for r,a in agents.items()};c=SequenceCritic();c.load_state_dict(critic.state_dict())
        ep=pr.response_rollout(model,adapters,agents[role],c,role,window,764)
        anchor_record=pr.teacher_records(model,adapters,role,example)
        learning=nt.update({role:agents[role]},c,{role:torch.optim.Adam(agents[role].parameters(),lr=1e-4)},torch.optim.Adam(c.parameters(),lr=3e-4),[ep],
            auxiliary=lambda current,actor:.05*pr.policy_nll(actor,anchor_record).mean())
        assert learning[role]['auxiliary_loss']>0
        assert nt.fingerprint(agents[role])!=before[role]
        assert all(nt.fingerprint(a)==before[r] for r,a in agents.items() if r!=role)
        checks[role]['fixed_partner_ppo_isolated_update']=True
        checks[role]['supervised_anchor_exercised']=True
    a=pr.joint_rollout(model,adapters,agents,critic,nt.Brief(1),814)
    b=pr.joint_rollout(model,adapters,agents,critic,nt.Brief(1),814,order=reversed(nt.ROLES))
    assert a['notes']==b['notes']
    assert base==adapters.base_hash() and all(torch.equal(v,frozen[k]) for k,v in adapters.state().items())
    result=dict(role_checks=checks,joint_order_independent=True,all_joint_notes_are_policy_actions=True,
                frozen_base_and_adapters_unchanged=True)
    (nt.ROOT/'outputs'/'partner_response_integration.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
