"""Run four context probes on A100 with fixed inputs; no optimizer updates."""
import modal
from modal_rock_pilot import image,ROOT,volume
app=modal.App('music-rock-prompt-diagnostic-v1')
probe_image=(image.add_local_file(ROOT/'generate_rock_comparison.py','/project/generate_rock_comparison.py')
    .add_local_file(ROOT/'modal_rock_pilot.py','/project/modal_rock_pilot.py')
    .add_local_file(ROOT/'rock_prompt_diagnostic.py','/project/rock_prompt_diagnostic.py')
    .add_local_file(ROOT/'outputs/rock_prompt_diagnostic_v1/protocol.json','/project/prompt_protocol.json'))

@app.function(image=probe_image,gpu='A100-40GB',cpu=4,memory=16384,timeout=1800,volumes={'/results':volume},max_containers=1)
def generate():
    import json,copy
    from pathlib import Path
    import torch
    import rock_prompt_diagnostic as d
    d.g.rock.configure();torch.set_num_threads(4);torch.manual_seed(d.SEED)
    p=json.loads(Path('/project/prompt_protocol.json').read_text());d.check_prefix(p['prefix'])
    out=Path('/results/rock_prompt_diagnostic_v1');out.mkdir(exist_ok=True)
    model=d.g.nt.load_model();adapters=d.g.l.Adapters(model);before=adapters.base_hash()
    planner=d.g.l.Planner().to(model.device);conditions=d.g.l.Conditions().to(model.device);actors={r:d.g.oe.Musician().eval() for r in d.g.rock.ROLES}
    results={}
    for label,name in d.CHECKPOINTS.items():
        source=Path('/results/rock_supervised_pilot_v1')/name
        assert d.g.sha(source)==p['checkpoint_sha256'][label]
        saved=torch.load(source,map_location='cpu',weights_only=False);assert saved['base_sha256']==before
        adapters.load(saved['adapters']);planner.load_state_dict(saved['planner']);conditions.load_state_dict(saved['conditions']);planner.eval();conditions.eval()
        for condition in ['unprompted','prompted']:
            key=label+'_'+condition;path=out/(key+'.json')
            if path.exists():episode=json.loads(path.read_text())
            else:
                episode=d.compose(model,adapters,planner,conditions,actors,p,condition);d.g.write_json(path,episode);volume.commit()
            results[key]=dict(sha256=d.g.sha(path),notes=len(episode['notes']),generated_continuation_notes=episode['generated_continuation_notes'])
        del saved
    assert adapters.base_hash()==before
    d.g.write_json(out/'verification.json',dict(protocol_sha256=d.g.sha(Path('/project/prompt_protocol.json')),clips=results,frozen_backbone_unchanged=True,no_training=True,gpu=torch.cuda.get_device_name()))
    volume.commit();return results

@app.local_entrypoint()
def main():print(generate.remote())
