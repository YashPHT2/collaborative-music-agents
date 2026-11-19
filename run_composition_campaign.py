"""Run data, training, then evaluation sequentially without losing long jobs."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-for-data',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    root=ROOT/'outputs'/'offline_ensemble_v1';root.mkdir(parents=True,exist_ok=True)
    status=root/'campaign_status.json'
    def stage(name,**extra):
        status.write_text(json.dumps(dict(stage=name,**extra),indent=2));print(name,flush=True)
    try:
        if a.wait_for_data:
            stage('waiting_for_initial_data_download');started=time.monotonic()
            while not (ROOT/'assets/composition_v1/manifest.json').exists():
                if time.monotonic()-started>1800:raise TimeoutError('Initial data preparation did not finish within 30 minutes')
                time.sleep(5)
        stage('preparing_audited_data')
        subprocess.run([sys.executable,'composition_data.py'],cwd=ROOT,check=True)
        stage('supervised_training')
        subprocess.run([sys.executable,'train_offline_ensemble.py']+(['--resume'] if a.resume else []),cwd=ROOT,check=True)
        if not (root/'latest.pt').exists():
            stage('stopped_at_data_gate_or_compute_budget',report='REPORT.md');return
        stage('generating_listening_panel_and_partner_probes')
        subprocess.run([sys.executable,'evaluate_offline_ensemble.py'],cwd=ROOT,check=True)
        stage('awaiting_listener_ratings',listen='listen.html',quality_improvement_established=False)
        subprocess.run([sys.executable,'serve_evaluation.py','--report','offline','--port','8777'],cwd=ROOT,check=True)
    except BaseException as e:
        stage('failed',error=str(e));raise

if __name__=='__main__':main()
