"""Real model integration: simultaneity, action provenance and likelihood replay."""
import torch
import note_trio as nt


def main():
    torch.set_num_threads(4); torch.manual_seed(56)
    model=nt.load_model(); agents={r:nt.Musician() for r in nt.ROLES}; critic=nt.TeamCritic()
    a=nt.rollout(model,agents,critic,nt.Brief(1),814)
    b=nt.rollout(model,agents,critic,nt.Brief(1),814,order=list(reversed(nt.ROLES)))
    assert a['notes']==b['notes'] and a['decisions']!=[]
    assert sorted(a['notes'])==sorted(d['note'] for d in a['decisions'])
    seen={r:set() for r in nt.ROLES}
    for step in a['steps']:
        for role,records in step['records'].items():
            for r in records:
                d=agents[role].distribution(r['observation'],r['reference'])
                assert abs(float(d.log_prob(torch.tensor(r['action'])))-r['logp'])<1e-5
                seen[role].add(int(r['observation'][1056:1064].argmax()))
    for role in nt.ROLES:
        assert {0,2,5,6,7}<=seen[role], (role,seen[role])
    print({'real_model_integration':'passed','identical_under_reversed_agent_order':True,
           'every_note_is_a_policy_action':True,'exact_log_probability_replay':True,
           'note_attributes_sampled_per_role':{r:sorted(v) for r,v in seen.items()},'notes':len(a['notes'])})


if __name__=='__main__': main()
