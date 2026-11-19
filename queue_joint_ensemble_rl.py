"""Wait without touching the active GPU run, then run joint RL once."""
import json
import subprocess
import sys
import time
from pathlib import Path
from train_ensemble_repair import OUT as SOURCE,write_json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'joint_ensemble_rl_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Atomic exclusive lock prevents two queue processes starting two GPU jobs.
    lock=OUT/'queue.lock'
    try:handle=lock.open('x')
    except FileExistsError:raise RuntimeError('Joint RL queue already exists; inspect its process before removing the lock')
    import os
    with handle:handle.write(str(os.getpid()))
    try:
        write_json(OUT/'queue_status.json',dict(status='waiting_for_supervised_training_and_evaluation',
                   source=str(SOURCE/'latest.pt'),rl_hours=1,total_project_ceiling_hours=8))
        while True:
            try:
                log=json.loads((SOURCE/'training_log.json').read_text())
                status=json.loads((SOURCE/'campaign_status.json').read_text())
                ready=log.get('complete') and status.get('stage')=='awaiting_listener_ratings' and (SOURCE/'latest.pt').exists()
            except (OSError,json.JSONDecodeError):ready=False
            if ready:break
            time.sleep(10)
        # The repair campaign is capped at 3h; RL adds at most 1h. Earlier
        # completed campaign took <1h, keeping the combined run below 8h.
        write_json(OUT/'queue_status.json',dict(status='running',source=str(SOURCE/'latest.pt'),rl_hours=1))
        with (ROOT/'outputs'/'joint_ensemble_rl_training.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(ROOT/'joint_ensemble_rl.py'),'--smoke','--updates','1','--hours','0.1',
                            '--output',str(ROOT/'outputs'/'joint_ensemble_rl_smoke')],stdout=log,stderr=subprocess.STDOUT,check=True)
            smoke=json.loads((ROOT/'outputs'/'joint_ensemble_rl_smoke'/'training_log.json').read_text())
            if not smoke.get('complete'):raise RuntimeError('Real-model RL smoke test did not complete within its budget')
            subprocess.run([sys.executable,str(ROOT/'joint_ensemble_rl.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
        trained=json.loads((OUT/'training_log.json').read_text())
        write_json(OUT/'queue_status.json',dict(status='complete' if trained.get('complete') else 'budget_stopped',
                                               selected_step=trained.get('selected_step',0),quality='pending listener evaluation'))
        if (OUT/'listen.html').exists():
            with (ROOT/'outputs'/'joint_ensemble_rl_server.log').open('w',encoding='utf-8') as server_log:
                subprocess.Popen([sys.executable,str(ROOT/'serve_evaluation.py'),'--report','joint-rl','--port','8780'],
                                 stdout=server_log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    except BaseException as error:
        write_json(OUT/'queue_status.json',dict(status='failed',error=str(error)));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
