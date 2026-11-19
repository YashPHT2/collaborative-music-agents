"""Launch the isolated, two-hour rock supervised pilot on Modal A100."""
import ast,json
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parent
app=modal.App('music-rock-adapter-pilot-v1')
volume=modal.Volume.from_name('music-rock-training-v1',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.11')
    .apt_install('libsndfile1')
    .pip_install('torch==2.6.0',index_url='https://download.pytorch.org/whl/cu124')
    .pip_install('transformers==4.57.1','peft==0.17.1','huggingface-hub==0.36.0','safetensors==0.6.2','mido==1.3.3','numpy==1.26.4','scipy==1.14.1','soundfile==0.13.1','requests==2.32.3','tqdm==4.67.1','Pillow==11.1.0')
    .env({'USE_TF':'0','USE_FLAX':'0','TOKENIZERS_PARALLELISM':'false','MPLBACKEND':'Agg','PYTHONPATH':'/project:/project/vendor/skytnt-midi-model'})
    .workdir('/project'))

# Upload only the local import dependency closure and required data/weights.
def dependencies(name,seen=None):
    seen=set() if seen is None else seen
    if name in seen:return seen
    path=ROOT/(name+'.py')
    if not path.exists():return seen
    seen.add(name)
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
        modules=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
        for module in modules:
            if (ROOT/(module+'.py')).exists():dependencies(module,seen)
    return seen
for name in sorted(dependencies('train_rock_pilot')):
    image=image.add_local_file(ROOT/(name+'.py'),'/project/'+name+'.py')
for name in ['rock_examples.json.gz','rock_manifest.json']:
    image=image.add_local_file(ROOT/'outputs/rock_training_v1'/name,'/project/outputs/rock_training_v1/'+name)
image=image.add_local_dir(ROOT/'vendor/skytnt-midi-model','/project/vendor/skytnt-midi-model',ignore=['__pycache__','.git','*.ipynb'])
for name in ['config.json','model.safetensors']:
    image=image.add_local_file(ROOT/'models/skytnt-tv2o-medium'/name,'/project/models/skytnt-tv2o-medium/'+name)
image=image.add_local_file(ROOT/'outputs/preferred_model_v1/step224.pt','/project/outputs/preferred_model_v1/step224.pt')

@app.function(image=image,gpu='A100-40GB',cpu=8,memory=32768,timeout=8100,volumes={'/results':volume},max_containers=1)
def train():
    import sys,traceback
    sys.path.insert(0,'/project')
    from train_rock_pilot import run,write
    output=Path('/results/rock_supervised_pilot_v1')
    try:
        result=run(output,steps=128,hours=2,on_save=volume.commit)
        volume.commit();return result
    except Exception:
        output.mkdir(parents=True,exist_ok=True)
        write(output/'failure.json',dict(error=traceback.format_exc(),training_success=False))
        volume.commit();raise

@app.local_entrypoint()
def main():
    print(json.dumps(train.remote()),flush=True)
