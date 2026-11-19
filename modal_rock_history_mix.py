"""A100 paired history experiment, shared two-hour cap, persistent results."""
import modal
from modal_rock_pilot import image,ROOT,volume
app=modal.App('music-rock-history-mix-v1')
training_image=image
for name in ['generate_rock_comparison.py','rock_prompt_diagnostic.py','rock_history_mix.py','train_rock_history_mix.py','modal_rock_pilot.py']:
    training_image=training_image.add_local_file(ROOT/name,'/project/'+name)

@app.function(image=training_image,gpu='A100-40GB',cpu=8,memory=32768,timeout=8100,volumes={'/results':volume},max_containers=1)
def train():
    from pathlib import Path
    import traceback
    from train_rock_history_mix import run,write
    output=Path('/results/rock_history_mix_v1')
    try:return run(output,Path('/results/rock_supervised_pilot_v1/progress.pt'),on_save=volume.commit)
    except Exception:
        output.mkdir(exist_ok=True);write(output/'failure.json',dict(error=traceback.format_exc(),training_success=False));volume.commit();raise

@app.local_entrypoint()
def main():print(train.remote())
