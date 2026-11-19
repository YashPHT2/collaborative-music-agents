"""Versioned release after generated-domain reward audit and PPO revision."""
from pathlib import Path
import json
import build_h100_package as base
import build_a100_upgrade_package as previous

ROOT=Path(__file__).resolve().parent

def notebook():
    nb=previous.notebook()
    for cell in nb.cells:
        cell.source=cell.source.replace('a100_music_upgrade_v2.zip','a100_music_upgrade_v3.zip').replace('a100_partner_280_v2','a100_partner_280_v3').replace('a100_joint_rl_v1','a100_joint_rl_v3')
        cell.source=cell.source.replace('"--episodes", "2"','"--episodes", "8"')
        cell.source=cell.source.replace('RL_RUN / "progress.pt",\n                                 "--output"','RL_RUN / "selected.pt",\n                                 "--output"')
        cell.source=cell.source.replace('if isinstance(v,dict)','if isinstance(v,dict) and "loss" in v')
        cell.source=cell.source.replace('"test_orchestra_joint_rl", "-q"','"test_orchestra_joint_rl", "test_orchestra_rl_selection", "-q"')
    nb.cells[0].source += '''\n\n### v3 audit outcome and actual limits
The frozen audio reward was rechecked on six original 19-role generated clips.
It passed 12/12 loop/shuffle contrasts and gain invariance, but ranked silence
above 5/6 originals and contradicted the one explicit listener preference.
It is rejected as a taste reward. See `reward_audit/acceptance.json` and rate
the original clips at `reward_audit/listen.html`. These six ratings can help
collect preferences; they cannot alone establish generalization.

PPO now performs policy/critic updates on CUDA when available, records entropy,
reference KL, clipping and critic explained variance, uses eight independent
performances per update, and selects checkpoints on twelve fixed validation
seeds with per-role participation guards and paired uncertainty. Training reward
does not select checkpoints. The selected candidate still needs listening.

Remaining work is explicit: no accepted reward artifact is shipped, Transformer
rollouts are still sequential, and the reward/advantages are still shared team
signals. Counterfactual credit, batched Transformer rollouts and planner PPO
adaptation are not claimed as implemented. Full A100 execution is unverified.
'''
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'notebook','exec')
    return nb

def main():
    base.DIST=ROOT/'outputs/a100_upgrade_package_v3';base.PAYLOAD=base.DIST/'a100_music_package'
    base.ARCHIVE_NAME='a100_music_upgrade_v3.zip';base.NOTEBOOK_NAME='A100_Orchestra_Training_v3.ipynb'
    base.notebook=notebook
    base.README=previous.README.replace('v2','v3').replace('actors and critic train on CPU','actors and critic train on CUDA when available')+'''
## v3 changes and unresolved requirements
Generated-domain reward audit: intact beats loop/shuffle in 12/12 tests, but silence
beats 5/6 intact clips. Known A100-over-laptop preference is contradicted. No RL is
authorized by these results. Rate reward_audit/listen.html; a larger independent
preference dataset and fitted/calibrated reward are still needed.
PPO policy/critic optimization supports CUDA, uses eight episodes/update, reports
entropy/reference KL/clip fraction/explained variance and selects on twelve fixed
validation seeds with paired bootstrap and per-role participation guards. Those
guards prevent gross disappearance/flooding, not all forms of musical regression.
Sampling remains sequential. Role-specific causal credit and planner adaptation
remain research work. No improved music or validated reward is claimed.
'''
    extras=[('outputs/a100_phrase_snapshot_v1/checkpoints/phrases.pt','checkpoints/listener_favorite_128.pt'),
      ('outputs/reward_validation_v2/acceptance.json','provenance/reward_validation/acceptance.json'),
      ('A100_RL_REWARD_CONTRACT.md','A100_RL_REWARD_CONTRACT.md')]
    for p in (ROOT/'outputs/orchestra_reward_audit_v3').glob('*'):
        if p.is_file():extras.append((str(p.relative_to(ROOT)),'reward_audit/'+p.name))
    base.main(extra_seeds=['orchestra_joint_rl','orchestra_ppo_update','orchestra_rl_selection','test_orchestra_joint_rl','test_orchestra_rl_selection','compare_orchestra_rl'],
      extra_files=extras,initial_checkpoint='outputs/ensemble_upgrade_v1/imported/progress_step280.pt',initial_destination='checkpoints/a100_step280.pt',package_version=3)

if __name__=='__main__':main()
