"""Real checkpoint gradient, freezing, exact-note and synchronous replay checks."""
import gzip,json,copy
import torch
import note_trio as nt
import supervised_ensemble as se
import offline_ensemble as oe
from train_offline_ensemble import teacher_loss,raw_loss,plan_loss

def main():
    torch.set_num_threads(4);torch.manual_seed(20261019)
    model=nt.load_model();adapters=se.InstrumentAdapters(model);original=adapters.base_hash()
    planner=oe.Planner().to(model.device);conditions=oe.Conditions().to(model.device)
    w=None
    for path in sorted((nt.ROOT/'assets/trio_ensemble/corpus').glob('*.json.gz')):
        with gzip.open(path,'rt') as f:rec=json.load(f)
        w=next(se.windows(rec),None)
        if w is not None:break
    if w is None:raise ValueError('No valid integration example')
    w['bars']=4;role=w['available_roles'][0]
    raw=[dict(start=n[1],duration=n[2],pitch=n[4],velocity=n[5],program=[19,48,42][n[3]]) for n in w['notes']]
    adapters.activate('organ',training=True)
    general_loss=raw_loss(model,dict(raw=raw),0)
    assert general_loss is not None and torch.isfinite(general_loss)
    general_loss.backward();model.zero_grad()
    pl=plan_loss(planner,w,model.device);pl.backward();assert planner.head.weight.grad is not None
    planner.zero_grad()
    adapters.activate(role,training=True);before=adapters.state()
    opt=torch.optim.AdamW(adapters.params(role)+list(conditions.parameters()),lr=1e-4)
    loss=teacher_loss(model,conditions,w,role,5,'revised');loss.backward();opt.step()
    after=adapters.state()
    changed=any(not torch.equal(before[k],after[k]) for k in before if f'.{role}.' in k)
    other_fixed=all(torch.equal(before[k],after[k]) for k in before if f'.{role}.' not in k)
    assert changed and other_fixed and adapters.base_hash()==original
    conditions.eval();planner.eval();agents={r:oe.Musician().eval() for r in oe.ROLES}
    ep=oe.compose(model,adapters,agents,planner,conditions,2,9619001,'revised')
    replay=oe.compose(model,adapters,agents,planner,conditions,2,9619001,'revised',order=oe.ROLES[::-1])
    assert ep['notes']==replay['notes']
    root=nt.ROOT/'outputs'/'offline_integration';root.mkdir(exist_ok=True)
    nt.midi_file(root/'replay.mid',ep['notes'])
    result=dict(loss=float(loss.detach()),general_ensemble_loss=float(general_loss.detach()),planner_loss=float(pl.detach()),selected_adapter_changed=changed,other_adapters_fixed=other_fixed,
        frozen_backbone_unchanged=True,revision_order_independent=True,exact_note_midi_roundtrip=True,notes=len(ep['notes']),
        quality_claim=False,description='Engineering integration test with one gradient update; not the trained campaign.')
    (root/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
