import copy
import unittest
from unittest.mock import patch
import torch
import note_trio as nt
import partner_response as pr
from sequence_critic import SequenceCritic


class FakeSampler:
    calls=[]
    def __init__(self,model,role,seed,brief,beat,history):
        self.role=role;self.beat=beat;self.history=copy.deepcopy(history)
        self.calls.append(dict(role=role,beat=beat,history=copy.deepcopy(history)))
    def propose(self,actor,state):
        s=nt.ROLES[self.role]
        pitch={'organ':60,'strings':72,'cello':48}[self.role]
        return [['note',self.beat*480,120,s['channel'],pitch,60]],[]


class FakeAdapters:
    def activate(self,role):pass


class ResponseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);FakeSampler.calls=[]
        self.example=dict(song='a',window_start=0,available_roles=list(nt.ROLES),notes=[
            ['note',beat*480,120,s['channel'],{'organ':64,'strings':76,'cello':43}[r],60]
            for beat in range(16) for r,s in nt.ROLES.items()])
        self.donor=copy.deepcopy(self.example);self.donor['song']='b'
        for n in self.donor['notes']:n[4]+=1

    def test_role_rotation_fixed_partners_and_causal_ledger(self):
        for role in nt.ROLES:
            FakeSampler.calls=[]
            with patch.object(pr.se,'EnsembleSampler',FakeSampler):
                ep=pr.response_rollout(None,FakeAdapters(),None,SequenceCritic(),role,self.example,42)
            ch=nt.ROLES[role]['channel']
            fixed=[n for n in ep['notes'] if n[3]!=ch]
            self.assertEqual(fixed,[n for n in sorted(self.example['notes'],key=lambda n:(n[1],n[3],n[4])) if n[3]!=ch])
            self.assertEqual(len(ep['decisions']),12)
            self.assertTrue(all(set(s['records'])=={role} for s in ep['steps']))
            for call in FakeSampler.calls:self.assertTrue(all(n[1]<call['beat']*480 for n in call['history']))
            self.assertEqual(set(ep['teacher_partner_roles']),set(nt.ROLES)-{role})

    def test_missing_observation_keeps_real_evaluation_partners(self):
        with patch.object(pr.se,'EnsembleSampler',FakeSampler):
            ep=pr.response_rollout(None,FakeAdapters(),None,SequenceCritic(),'cello',self.example,42,mode='missing')
        self.assertTrue(all(n[3]==2 for call in FakeSampler.calls for n in call['history']))
        self.assertEqual(len(ep['dataset_fixed_partners']),24)
        self.assertTrue(all(t['observed_partner_notes']==0 for t in ep['trace']))

    def test_shuffled_changes_only_past_partner_observation(self):
        h=pr.observation_history(self.example['notes'],6,'cello','shuffled',self.donor,self.example['available_roles'])
        own=[n for n in h if n[3]==2]
        self.assertEqual(own,[n for n in nt.visible(self.example['notes'],6) if n[3]==2])
        self.assertTrue(all(n[1]<6*480 for n in h))
        self.assertEqual([n for n in h if n[3]!=2],[n for n in nt.visible(self.donor['notes'],6) if n[3]!=2])

    def test_missing_source_role_is_never_fabricated(self):
        self.example['available_roles']=['organ','cello']
        self.example['notes']=[n for n in self.example['notes'] if n[3]!=1]
        with patch.object(pr.se,'EnsembleSampler',FakeSampler):
            ep=pr.response_rollout(None,FakeAdapters(),None,SequenceCritic(),'cello',self.example,42)
        self.assertEqual(ep['teacher_partner_roles'],['organ'])
        self.assertFalse(any(n[3]==1 for n in ep['notes']))

    def test_joint_order_and_action_provenance(self):
        agents={r:None for r in nt.ROLES};critic=SequenceCritic()
        with patch.object(pr.se,'EnsembleSampler',FakeSampler):
            a=pr.joint_rollout(None,FakeAdapters(),agents,critic,nt.Brief(1),42)
            b=pr.joint_rollout(None,FakeAdapters(),agents,critic,nt.Brief(1),42,order=reversed(nt.ROLES))
        self.assertEqual(a['notes'],b['notes'])
        self.assertEqual(sorted(a['notes']),sorted(d['note'] for d in a['decisions']))
        self.assertTrue(all(t['origin']=='joint_policy' for t in a['decisions']))

    def test_reward_causality_and_available_roles(self):
        score=pr.ensemble_score(self.example['notes'],1920,['organ','cello'])
        changed=copy.deepcopy(self.example['notes'])
        for n in changed:
            if n[1]>=1920 or n[3]==1:n[4]=100
        self.assertEqual(score,pr.ensemble_score(changed,1920,['organ','cello']))
        self.assertEqual(pr.ensemble_score([],1920)[0],-1)

    def test_donor_requires_different_song_and_partner_roles(self):
        with self.assertRaises(ValueError):pr.donor_for(self.example,'cello',[self.example])
        self.assertEqual(pr.donor_for(self.example,'cello',[self.example,self.donor])['song'],'b')


if __name__=='__main__':unittest.main()
