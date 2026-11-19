"""Equal-update final-candidate listening ablation; no model promotion."""
import modal
from modal_rock_pilot import image,ROOT,volume
app=modal.App('music-rock-history-final-listening-v1')
preview_image=image
for name in ['modal_rock_pilot.py','generate_rock_comparison.py','rock_prompt_diagnostic.py']:
    preview_image=preview_image.add_local_file(ROOT/name,'/project/'+name)

@app.function(image=preview_image,gpu='A100-40GB',cpu=4,memory=16384,timeout=1800,volumes={'/results':volume},max_containers=1)
def generate():
    import hashlib,json
    from pathlib import Path
    import torch
    import rock_prompt_diagnostic as d
    d.g.rock.configure();torch.set_num_threads(4);torch.manual_seed(9976500)
    experiment=Path('/results/rock_history_mix_v1');result=json.loads((experiment/'result.json').read_text())
    assert result['complete'] and result['sample_plans_matched'] and result['source_unchanged'] and result['frozen_backbone_unchanged']
    out=experiment/'final_candidate_evaluation';out.mkdir(exist_ok=True)
    sources={'baseline':Path('/results/rock_supervised_pilot_v1/progress.pt'),'control_final':experiment/'control/progress.pt','mixed_final':experiment/'mixed/progress.pt'}
    config=json.loads((experiment/'experiment.json').read_text());seeds=config['evaluation_seeds']
    original_eval=json.loads((experiment/'evaluation/manifest.json').read_text())
    model=d.g.nt.load_model();adapters=d.g.l.Adapters(model);before=adapters.base_hash()
    planner=d.g.l.Planner().to(model.device);conditions=d.g.l.Conditions().to(model.device);actors={r:d.g.oe.Musician().eval() for r in d.g.rock.ROLES}
    artifacts={};models={};goal=None
    for name,path in sources.items():
        saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['base_sha256']==before and saved['roles']==d.g.rock.ROLES
        expected=128 if name=='baseline' else 64;assert saved['state']['step']==expected and saved['state']['complete']
        if name!='baseline':goal=saved['config']['goal']
        models[name]=dict(source_sha256=d.g.sha(path),step=saved['state']['step'],init_step=saved['state'].get('init_step'),selected_step=saved['state'].get('selected_step'),
            final_selection_passed=saved['state']['validations'][-1]['passed'] if name!='baseline' else None)
        del saved
    for name,path in sources.items():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        adapters.load(saved['adapters']);planner.load_state_dict(saved['planner']);conditions.load_state_dict(saved['conditions']);planner.eval();conditions.eval()
        for seed in seeds:
            key=f'{name}_{seed}';episode_path=out/(key+'.json')
            if episode_path.exists():episode=json.loads(episode_path.read_text())
            elif name=='baseline':
                original=experiment/'evaluation'/f'baseline_{seed}.json'
                assert original_eval['goal']==goal and d.g.sha(original)==original_eval['artifacts'][original.name]['sha256']
                assert models['baseline']['source_sha256']==config['source_sha256']
                episode=json.loads(original.read_text());d.g.write_json(episode_path,episode);volume.commit()
            else:
                print(name,seed,flush=True);p=dict(prefix=[],goal=goal,bpm=120,seed=seed)
                episode=d.compose(model,adapters,planner,conditions,actors,p,'unprompted');d.g.write_json(episode_path,episode);volume.commit()
            assert episode['seed']==seed and not episode['reference_prefix']
            artifacts[key]=dict(sha256=d.g.sha(episode_path),notes=len(episode['notes']),model=name,seed=seed)
        del saved
    assert adapters.base_hash()==before
    manifest=dict(schema='rock-history-final-candidate-listening-v1',seeds=seeds,bars=6,bpm=120,goal=goal,models=models,artifacts=artifacts,
        frozen_backbone_unchanged=True,no_new_training=True,all_samples_retained=True,
        note='Equal-update final candidates are shown for perceptual diagnosis despite selection failures. Original checkpoint remains the selected model.')
    d.g.write_json(out/'manifest.json',manifest);volume.commit();return manifest

@app.local_entrypoint()
def main():print(generate.remote())
