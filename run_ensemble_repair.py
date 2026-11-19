"""Run/resume the bounded repair and its evaluation, preserving all earlier runs."""
import argparse
import subprocess
import sys
from pathlib import Path
from train_ensemble_repair import OUT,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        write_json(OUT/'campaign_status.json',dict(stage='training'))
        command=[sys.executable,str(Path(__file__).with_name('train_ensemble_repair.py'))]
        if args.resume:command.append('--resume')
        subprocess.run(command,check=True)
        write_json(OUT/'campaign_status.json',dict(stage='evaluating'))
        subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_ensemble_repair.py'))],check=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name('joint_ensemble_rl.py'))],check=True)
    except BaseException as error:
        write_json(OUT/'campaign_status.json',dict(stage='interrupted',error=str(error),resume='python run_ensemble_repair.py --resume'))
        raise


if __name__=='__main__':main()
