"""Add isolated accompaniment listening clips to a completed pilot run."""
import csv
import json
import statistics

from accompany import ROOT,assemble,write_result
from evaluate_baseline import validate_midi,validate_audio


def prepare(root):
    from train_ppo_pilot import listening_page
    checks=json.loads((root/'verification.json').read_text())
    count=0
    for directory in sorted(root.glob('fresh_*')):
        for method in ['matched_initial','ppo_adapter']:
            print(f'Rendering isolated accompaniment: {directory.name}/{method}',flush=True)
            notes=json.loads((directory/f'{method}_notes.json').read_text())
            path=write_result(directory,method+'_solo',assemble([],notes,100),True)
            checks[f'{directory.name}/{method}_solo']=dict(**validate_midi(path,[],notes),audio=validate_audio(path.with_suffix('.wav')))
            count+=1
    (root/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    with (root/'metrics.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    means={method:statistics.mean(float(row['proxy']) for row in rows if row['method']==method)
           for method in sorted({row['method'] for row in rows})}
    wins={}
    for reference in ['original_sampler','matched_initial','rerank_3']:
        wins[reference]=sum(float(next(row['proxy'] for row in rows if row['case']==case and row['method']=='ppo_adapter'))>
                            float(next(row['proxy'] for row in rows if row['case']==case and row['method']==reference))
                            for case in {row['case'] for row in rows})
    decision=dict(default_model='original_pretrained_sampler',promote_adapter=False,
                  reason='Pilot proxy results and four synthetic cases do not establish listener preference. Keep the pretrained default.',
                  mean_exported_proxy=means,ppo_proxy_wins_out_of_four=wins,
                  human_quality_improvement_established=False)
    (root/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    listening_page(root,rows)
    manifest=json.loads((root/'manifest.json').read_text())
    manifest['accompaniment_only_clips']=count
    manifest['online_vs_exported_reward_note']='Online rewards use committed raw history; final proxy uses overlap-cleaned exported notes. Scores may differ because later duplicate-pitch attacks shorten earlier notes.'
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report_path=root/'REPORT.md'
    report=report_path.read_text(encoding='utf-8')
    marker='## Listening modes and reward timing'
    if marker not in report:
        report+='\n'+marker+'\n\nUse the listening-mode selector to switch between full duets and isolated accompaniment. '
        report+='The solo clips preserve exactly the accompaniment note events.\n\n'
        report+=manifest['online_vs_exported_reward_note']+'\n'
        report_path.write_text(report,encoding='utf-8')
    decision_marker='## Promotion decision'
    if decision_marker not in report:
        report+='\n'+decision_marker+'\n\nKeep the original pretrained sampler as the default. '
        report+='PPO proxy wins out of four: '+json.dumps(wins,sort_keys=True)+'. '
        report+='Mean exported proxy: '+json.dumps(means,sort_keys=True)+'. '
        report+='This pilot does not establish improved musical quality. No default generation or baseline files were replaced.\n'
        report_path.write_text(report,encoding='utf-8')


if __name__=='__main__':
    prepare(ROOT/'outputs'/'ppo_pilot_v1')
