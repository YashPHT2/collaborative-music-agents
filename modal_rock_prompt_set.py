"""Generate six step-128 context probes on one A100, without retraining."""
import modal
from modal_rock_pilot import image,ROOT,volume
app=modal.App('music-rock-prompt-set-v1')
probe_image=(image.add_local_file(ROOT/'generate_rock_comparison.py','/project/generate_rock_comparison.py')
    .add_local_file(ROOT/'modal_rock_pilot.py','/project/modal_rock_pilot.py')
    .add_local_file(ROOT/'rock_prompt_diagnostic.py','/project/rock_prompt_diagnostic.py'))
for i in range(1,4):
    key=f'prompt_{i:02d}';probe_image=probe_image.add_local_file(ROOT/'outputs/rock_prompt_set_v1'/key/'protocol.json',f'/project/prompt_set/{key}.json')

@app.function(image=probe_image,gpu='A100-40GB',cpu=4,memory=16384,timeout=1800,volumes={'/results':volume},max_containers=1)
def generate():
    import json
    from pathlib import Path
    import torch
    import rock_prompt_diagnostic as d
    d.g.rock.configure();torch.set_num_threads(4);torch.manual_seed(9976300)
    out=Path('/results/rock_prompt_set_v1');out.mkdir(exist_ok=True)
    model=d.g.nt.load_model();adapters=d.g.l.Adapters(model);before=adapters.base_hash()
    source=Path('/results/rock_supervised_pilot_v1/progress.pt');saved=torch.load(source,map_location='cpu',weights_only=False)
    assert saved['state']['step']==128 and saved['base_sha256']==before and saved['roles']==d.g.rock.ROLES
    adapters.load(saved['adapters']);planner=d.g.l.Planner().to(model.device);planner.load_state_dict(saved['planner']);planner.eval()
    conditions=d.g.l.Conditions().to(model.device);conditions.load_state_dict(saved['conditions']);conditions.eval()
    actors={r:d.g.oe.Musician().eval() for r in d.g.rock.ROLES};checks={}
    for i in range(1,4):
        key=f'prompt_{i:02d}';input_path=Path(f'/project/prompt_set/{key}.json');p=json.loads(input_path.read_text());d.check_prefix(p['prefix'])
        assert d.g.sha(source)==p['checkpoint_sha256']['step128']
        directory=out/key;directory.mkdir(exist_ok=True);results={}
        for condition in ['unprompted','prompted']:
            label='step128_'+condition;path=directory/(label+'.json')
            if path.exists():ep=json.loads(path.read_text())
            else:
                print(key,condition,flush=True);ep=d.compose(model,adapters,planner,conditions,actors,p,condition);d.g.write_json(path,ep);volume.commit()
            assert ep['seed']==p['seed']
            results[label]=dict(sha256=d.g.sha(path),notes=len(ep['notes']),generated_continuation_notes=ep['generated_continuation_notes'])
        verified=dict(protocol_sha256=d.g.sha(input_path),clips=results,frozen_backbone_unchanged=True,no_training=True,gpu=torch.cuda.get_device_name())
        d.g.write_json(directory/'verification.json',verified);volume.commit();checks[key]=results
    assert adapters.base_hash()==before
    d.g.write_json(out/'verification.json',dict(clips=checks,frozen_backbone_unchanged=True,no_training=True,checkpoint_sha256=d.g.sha(source)))
    volume.commit();return checks

@app.local_entrypoint()
def main():print(generate.remote())
