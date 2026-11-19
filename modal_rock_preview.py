"""Generate matched rock notes on A100; render downloaded MIDI locally."""
import modal
from modal_rock_pilot import image,ROOT,volume
app=modal.App('music-rock-listening-v1')
preview_image=(image.add_local_file(ROOT/'generate_rock_comparison.py','/project/generate_rock_comparison.py')
    .add_local_file(ROOT/'modal_rock_pilot.py','/project/modal_rock_pilot.py'))

@app.function(image=preview_image,gpu='A100-40GB',cpu=4,memory=16384,timeout=1800,volumes={'/results':volume},max_containers=1)
def generate():
    import json,gzip
    from pathlib import Path
    import numpy as np,torch
    import generate_rock_comparison as g
    g.rock.configure();torch.set_num_threads(4);torch.manual_seed(g.SEED)
    out=Path('/results/rock_listening_v1');out.mkdir(exist_ok=True)
    sources={'A':'selected.pt','B':'initial.pt','C':'progress.pt'}
    with gzip.open('/project/outputs/rock_training_v1/rock_examples.json.gz','rt') as f:windows=json.load(f)
    goal=np.mean([g.l.style([n for n in w['notes'] if n[3]!=3],w['available_roles']).numpy() for w in windows if w['split']=='train'],axis=0)
    model=g.nt.load_model();adapters=g.l.Adapters(model);before=adapters.base_hash()
    planner=g.l.Planner().to(model.device);conditions=g.l.Conditions().to(model.device)
    actors={r:g.oe.Musician().eval() for r in g.rock.ROLES};checks={}
    for variant,name in sources.items():
        saved=torch.load(Path('/results/rock_supervised_pilot_v1')/name,map_location='cpu',weights_only=False)
        assert saved['base_sha256']==before and saved['roles']==g.rock.ROLES
        path=out/(variant+'_episode.json')
        if path.exists():episode=json.loads(path.read_text())
        else:
            adapters.load(saved['adapters']);planner.load_state_dict(saved['planner']);conditions.load_state_dict(saved['conditions']);planner.eval();conditions.eval()
            episode=g.compose(model,adapters,planner,conditions,actors,torch.tensor(goal,device=model.device),variant)
            g.write_json(path,episode);volume.commit()
        checks[variant]=dict(step=saved['state']['step'],notes=len(episode['notes']),source_sha256=g.sha(Path('/results/rock_supervised_pilot_v1')/name))
        del saved
    assert adapters.base_hash()==before
    g.write_json(out/'notes_verified.json',dict(seed=g.SEED,clips=checks,frozen_backbone_unchanged=True,gpu=torch.cuda.get_device_name()))
    volume.commit();return checks

@app.local_entrypoint()
def main():
    print(generate.remote())
