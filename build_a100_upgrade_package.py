"""Release a fresh A100 notebook from step 280; never overwrite the v1 package."""
import json
import textwrap
from pathlib import Path
import build_h100_package as base

ROOT=Path(__file__).resolve().parent
DIST=ROOT/'outputs/a100_upgrade_package_v2'
ORIGINAL_NOTEBOOK=base.notebook


def notebook():
    import nbformat as nbf
    nb=ORIGINAL_NOTEBOOK()
    for cell in nb.cells:
        cell.source=cell.source.replace('h100_music_training_v1.zip','a100_music_upgrade_v2.zip').replace('h100_music_package','a100_music_package')
        cell.source=cell.source.replace('h100_orchestra_v1','a100_partner_280_v2').replace('h100_music_results.zip','a100_music_results_v2.zip')
        cell.source=cell.source.replace('h100_venv','a100_music_venv').replace('music_dependencies_v1.ready','music_dependencies_v2.ready')
        if cell.cell_type=='code' and 'PROJECT = EXTRACT_ROOT' in cell.source:
            cell.source += '\nINIT = PROJECT / "checkpoints/a100_step280.pt"\nFAVORITE = PROJECT / "checkpoints/listener_favorite_128.pt"'
        if cell.cell_type=='code' and 'SMOKE = ' in cell.source:
            cell.source=cell.source.replace('"test_full_orchestra", "-q"','"test_full_orchestra", "test_orchestra_joint_rl", "-q"')
            cell.source=cell.source.replace('"--steps", "2"','"--init", INIT, "--steps", "2"')
        if cell.cell_type=='code' and 'STEPS = 512' in cell.source:
            cell.source=textwrap.dedent('''
                STEPS = 512  # partner updates; starts from the saved A100 phrase step 280
                PHRASES_PER_ROLE = 8  # unchanged effective sampling: 8 phrases x 19 roles
                HOURS = 6  # cumulative; includes audits, calibration and draft generation
                LEARNING_RATE = 3e-5  # new adaptation stage, intentionally new optimizer
                MICROBatch = 0  # calibrate on the actual A100, without changing effective batch
                TOKEN_BUDGET = 131072
                BASE_ARGS = ["h100_train.py", "--init", INIT, "--output", RUN,
                    "--start-stage", "partner", "--partner-warmup", "16", "--preserve-phrase",
                    "--steps", str(STEPS), "--phrases-per-role", str(PHRASES_PER_ROLE),
                    "--hours", str(HOURS), "--learning-rate", str(LEARNING_RATE),
                    "--microbatch", str(MICROBatch), "--token-budget", str(TOKEN_BUDGET),
                    "--validation-interval", "32", "--save-interval", "8", "--draft-refresh", "64",
                    "--seed", "9932001", *ACCELERATOR_ARGS]
                if not (RUN / "verification.json").exists():
                    verify_args = BASE_ARGS + ["--mode", "verify"]
                    if (RUN / "progress.pt").exists(): verify_args += ["--resume"]
                    run_command(verify_args)
                if not (RUN / "batch_benchmark.json").exists():
                    run_command(BASE_ARGS + ["--mode", "benchmark", "--resume"])
                if not (RUN / "initial_audit.json").exists():
                    current = json.loads((RUN / "training_log.json").read_text())
                    if current.get("updates"):
                        raise RuntimeError("Missing initial audit on a previously updated run; retain this run and use a fresh folder.")
                    run_command(BASE_ARGS + ["--mode", "audit", "--audit-reference", FAVORITE, "--resume"])
                display(json.loads((RUN / "batch_benchmark.json").read_text()))
                print("Step 280 is the starting checkpoint. Step 256 weights were not present in your download.")
                print("The original optimizer/RNG are archived; this is a new partner-adaptation stage.")
            ''').strip()
        if cell.cell_type=='code' and 'print("Selected checkpoints:"' in cell.source:
            cell.source=cell.source.replace('print("Selected checkpoints:", state.get("selections", {}))',
                'print("Selected validation update:", state.get("selected_step", 0), "| final stages:", state.get("selections", {}))')
        if cell.cell_type=='code' and 'command = ["evaluate_trained_orchestra.py"' in cell.source:
            cell.source=cell.source.replace('"--output", destination, "--seed", str(seed)]','"--output", destination, "--seed", str(seed), "--initial-checkpoint", INIT]')
    nb.cells[0].source=textwrap.dedent('''
        # A100: improve your saved orchestra, then evaluate joint RL

        Upload **a100_music_upgrade_v2.zip** beside this notebook. Select an
        **A100 40 GB or 80 GB** CUDA runtime, Python 3.10–3.12, and persistent
        results storage. No H100-name workaround cell is needed. The same GPU
        BF16 check also accepts H100/H200; this release is prepared for your A100.

        Starts from your **actual phrase step 280** (`progress (1).pt`). Your
        listener-favorite step-128 checkpoint and original download remain archived.
        The 19 roles share seven family LoRA adapters over the frozen 233M Transformer.

        **Stage 1:** bounded supervised partner-response adaptation, 512 updates,
        eight phrases per role/update, LR 3e-5. First 16 updates train only the
        partner-condition network. Later updates train adapters, planner and conditions.
        Validation every 32 updates rejects individual-role regressions and checks
        preservation of phrase prediction. No checkpoint is declared better by loss alone.

        **Stage 2:** 19-role joint PPO code is included, with a causal sequence
        critic and checkpoints. **It is blocked in this release:** the earlier reward
        experiment failed and fresh-composition human preferences are missing. The
        RL cells report the reason and do not train until an independently validated,
        hash-bound reward is supplied. See `A100_RL_REWARD_CONTRACT.md`.

        Keep the pretrained backbone frozen. This is new partner adaptation from
        step 280, not an exact continuation of your original phrase optimizer.
        Audited data still uses four-bar targets with two preceding bars; more GPU
        memory does not create new scores or trained long-form structure.
    ''').strip()
    for cell in nb.cells:
        if cell.cell_type=='markdown' and '## Choose a bounded run' in cell.source:
            cell.source='''## Choose a bounded partner run
512 partner updates, eight phrases per role, a six-hour cumulative budget. The
first 16 updates warm up conditioning while adapters and planner stay frozen.
Then all three supervised components adapt together. A100 calibration changes
physical chunks only; it keeps every selected training example and the effective
phrase count. Every eight updates save weights, optimizer, RNG and recipe hashes.

The audit compares exact step-280 weights against the listener-favorite checkpoint
before any optimizer updates. Partner selection must improve balanced event loss
and keep each role's event and attribute losses within 2% of its starting value;
the same per-role preservation check runs without announced drafts. Failed
adaptation falls back to the earlier checkpoint. Start a new folder to change
learning rate or sampling; resume requires the original recipe.'''
    def md(s):return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())
    def py(s):return nbf.v4.new_code_cell(textwrap.dedent(s).strip())
    extra=[md('''
        ## Joint multi-agent RL: readiness check
        The legacy three-instrument PPO is not used here. `orchestra_joint_rl.py`
        supports all 19 role IDs, independent residual note actors and a shared
        sequence critic. It samples private frozen drafts and commits final notes
        synchronously. All note attributes and rests remain policy decisions.

        The shipped reward acceptance record **failed**. This check runs before
        allocating a model or changing policy weights. Do not change the booleans
        to force a pass. `reward.ts` and independent validation evidence are still
        needed; the contract documents inputs and exact generator/code hashes.
    '''),py('''
        RL_RUN = RESULTS_ROOT / "a100_joint_rl_v1"
        RL_SOURCE = RUN / "selected.pt"
        REWARD_DIR = PROJECT / "provenance/reward_validation"
        RL_ARGS = ["orchestra_joint_rl.py", "--source", RL_SOURCE, "--reward", REWARD_DIR,
                   "--output", RL_RUN, "--updates", "24", "--episodes", "2", "--bars", "8", "--hours", "2"]
        if RL_SOURCE.exists():
            result = subprocess.run([str(PY), *map(str, RL_ARGS), "--check"], cwd=PROJECT,
                                    env=ENV, capture_output=True, text=True, check=True)
            RL_READINESS = json.loads(result.stdout.strip().splitlines()[-1])
            print(json.dumps(RL_READINESS, indent=2))
        else:
            RL_READINESS = {"rl_allowed": False, "reason": "Finish partner training and checkpoint selection first."}
            print(RL_READINESS)
        print("Contract:", PROJECT / "A100_RL_REWARD_CONTRACT.md")
    '''),md('''
        ### Optional real-GPU rollout verification
        Disabled during Run All. It performs a two-bar joint rollout and probability
        replay with zero optimizer updates. The zero score is an inference test,
        never a musical-quality reward or an accepted way to start PPO.
    '''),py('''
        RUN_RL_ROLLOUT_SMOKE = False
        if RUN_RL_ROLLOUT_SMOKE:
            assert RL_SOURCE.exists(), "Finish partner selection first."
            run_command(["orchestra_joint_rl.py", "--source", RL_SOURCE,
                         "--output", RESULTS_ROOT / "rl_rollout_smoke", "--rollout-smoke"])
            display(json.loads((RESULTS_ROOT / "rl_rollout_smoke/rollout_smoke.json").read_text()))
    '''),md('''
        ### Start / resume PPO only after validated reward integration
        Default is off. Enabling this does not override reward validation. The
        trainer checks acceptance and source/reward/code hashes again before any
        GPU load or update. No automatic promotion by training reward: listen to
        all fixed comparisons and test actual partner use before replacing a model.
    '''),py('''
        START_RL = False
        if START_RL:
            if not json.loads((RUN / "training_log.json").read_text()).get("complete"):
                raise RuntimeError("Wait for the partner trainer to exit before starting PPO.")
            # Readiness is checked inside the child even if artifacts changed since the check cell.
            if not RL_READINESS.get("rl_allowed"):
                raise RuntimeError(RL_READINESS["reason"])
            RL_RUN.mkdir(parents=True, exist_ok=True)
            RL_PID = RL_RUN / "trainer.pid"
            if RL_PID.exists():
                try: os.kill(int(RL_PID.read_text()), 0)
                except (ProcessLookupError, OSError): RL_PID.unlink()
                else: raise RuntimeError("An RL trainer is already alive; use its monitor.")
            if (RL_RUN / "training_log.json").exists() and json.loads((RL_RUN / "training_log.json").read_text()).get("complete"):
                print("PPO run is already complete; use the comparison cell.")
            else:
                rl_log = (RL_RUN / "console.log").open("a", buffering=1)
                resume_args = ["--resume"] if (RL_RUN / "progress.pt").exists() else []
                rl_trainer = subprocess.Popen([str(PY), "-u", *map(str, RL_ARGS), *resume_args],
                    cwd=PROJECT, env=ENV, stdout=rl_log, stderr=subprocess.STDOUT)
                RL_PID.write_text(str(rl_trainer.pid))
                print("PPO PID:", rl_trainer.pid, "|", RL_RUN / "console.log")
        else:
            print("RL not started. Partner training and listening do not require the failed reward.")
    '''),py('''
        # Live PPO monitor; interrupting this cell does not stop training.
        if "rl_trainer" in globals():
            while True:
                clear_output(wait=True)
                path = RL_RUN / "training_log.json"
                if path.exists():
                    try: rl_state = json.loads(path.read_text())
                    except json.JSONDecodeError: time.sleep(2); continue
                    print(rl_state["status"], "| update", rl_state["step"], "| minutes", round(rl_state["elapsed"]/60, 1))
                    if rl_state["updates"]:
                        last = rl_state["updates"][-1]
                        print("Reward:", last["reward"], "Critic loss:", last["losses"]["critic_loss"])
                        print("Every role's policy loss:", {r:v["loss"] for r,v in last["losses"].items() if isinstance(v,dict)})
                    if rl_state.get("complete"): break
                if rl_trainer.poll() is not None:
                    print("RL exited:", rl_trainer.returncode)
                    print((RL_RUN / "console.log").read_text(errors="replace")[-4000:]); break
                time.sleep(15)
    '''),py('''
        STOP_RL = False  # Run All must not stop a trainer
        if STOP_RL and (RL_RUN / "trainer.pid").exists():
            os.kill(int((RL_RUN / "trainer.pid").read_text()), signal.SIGTERM)
            print("Stop requested; last complete update is saved. Allow the current episode to finish.")
    '''),py('''
        if (RL_RUN / "training_log.json").exists() and json.loads((RL_RUN / "training_log.json").read_text()).get("complete"):
            for seed in [9919101, 9919102, 9919103]:
                destination = RESULTS_ROOT / "rl_listening" / str(seed)
                if not (destination / "evaluation.json").exists():
                    run_command(["compare_orchestra_rl.py", "--source", RL_SOURCE, "--rl", RL_RUN / "progress.pt",
                                 "--output", destination, "--seed", str(seed)])
                print("Seed", seed, "supervised foundation"); display(Audio(filename=str(destination / "supervised.mp3")))
                print("Seed", seed, "PPO candidate"); display(Audio(filename=str(destination / "rl_candidate.mp3")))
                display(FileLink(str(destination / "listen.html")))
        else:
            print("No completed PPO run to compare. The accepted supervised checkpoint remains available.")
    ''')]
    # Insert before the existing results-export heading/cell.
    i=next(i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and '## Save and download results' in c.source)
    nb.cells[i:i]=extra
    nbf.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'a100_notebook_cell','exec')
    return nb


README='''# A100 orchestra upgrade v2

Upload `a100_music_upgrade_v2.zip` and open `A100_Orchestra_Training_v2.ipynb`.
Use an A100 40GB or 80GB runtime, Python 3.10–3.12, internet, at least 15GB free
disk and persistent results storage. No compatibility patch cell is required.
The actual GPU is checked for CUDA/BF16 and benchmarked before training.

This starts a NEW partner-response stage from the user's actual A100 phrase step
280. It does not silently resume the old phrase optimizer or invent step-256
weights. Original checkpoint plus optimizer/RNG are included unmodified; the
listener-favorite step-128 weights are also preserved for reference.

Partner recipe: up to 512 updates; eight real phrases per role/update; new LR
3e-5; 16 conditioning-only warmup updates; then joint supervised adapter/planner/
condition training. Twelve composition-separated validation groups per role;
checks every 32 updates; snapshots every eight; six-hour cumulative budget.
Keep all effective examples. Physical GPU chunks are calibrated at 16..512,
with a padded-token cap and OOM recovery. More memory usage alone is not quality.

Selection requires >=0.5% family-macro event improvement and no more than 2%
event/attribute regression in any role. Phrase prediction is separately protected
when partner conditioning is absent. Initial audit compares exact step 280 with
the listener-favorite reference. If adaptation fails, preserve the starting model.
Matched listening clips and held-out partner probes remain separate evidence.

Data: 6944 train windows / 857 composition groups; 2419 validation windows / 238
groups; 1254 untouched test windows / 102 groups. Targets span four contiguous
bars after two context bars. Seven shared family LoRAs and 19 role conditions over
a frozen 233M Transformer; partial real ensembles, missing-role masks, genuine
rests and playability limits. No invented nineteen-instrument ground-truth scores,
emotion labels, copied reference soundtrack or automatic concentration benefit.

`orchestra_joint_rl.py` provides 19 independent small residual note policies,
a centralized causal sequence critic, shared team GAE, clipped PPO and reference
KL, with exact old-log-prob replay and synchronous commitments. Frozen supervised
weights are unchanged by PPO. Atomic full-update checkpoints include optimizer/
RNG/hash state. `compare_orchestra_rl.py` renders fixed-seed supervised/PPO pairs.

RL IS BLOCKED in this release. The existing reward experiment failed and human
preferences on fresh generated music are missing. The notebook reports the reason
and skips PPO; it does not silently optimize a known-failed proxy. A validated,
hash-bound TorchScript reward and evidence are required. See the included
`A100_RL_REWARD_CONTRACT.md`. Do not edit acceptance booleans to bypass the gate.
Optional real-GPU rollout smoke performs inference/probability replay only.

Download `selected.pt`, `progress.pt`, `training_log.json`, validation evidence and
the results ZIP. Exact resume requires this package and the same recipe/source.
RL progress files also require their exact supervised source checkpoint.

Locally verified: existing supervised tests, conditioning-only warmup integrity,
exact step-280 validation against the listener reference, synthetic PPO updates
for all 19 actors, causal critic including role 18, replay/order invariance and
failed reward blocking. A100 execution of this new release is not yet verified;
no promised throughput, musical preference improvement or 'best model' claim.

Base weights download at pinned revisions and hashes. The included SoundFont and
CC0 felt-piano assets match the existing rendering. No commercial reference MP3,
remote credentials, tokens or full raw MIDI archives are included. Original
H100 package v1 remains unchanged. The isolated environment retains the tested
PyTorch2.6/cu124 pins; these are compatibility pins, not newest-version claims.
'''


def main():
    base.DIST=DIST;base.PAYLOAD=DIST/'a100_music_package'
    base.ARCHIVE_NAME='a100_music_upgrade_v2.zip';base.NOTEBOOK_NAME='A100_Orchestra_Training_v2.ipynb'
    base.README=README;base.notebook=notebook
    extras=[('outputs/a100_phrase_snapshot_v1/checkpoints/phrases.pt','checkpoints/listener_favorite_128.pt'),
        ('outputs/ensemble_upgrade_v1/imported/provenance.json','provenance/step280_import.json'),
        ('outputs/ensemble_upgrade_v1/partner_pilot_280/initial_audit.json','local_verification/initial_audit_step280.json'),
        ('outputs/ensemble_upgrade_v1/partner_pilot_280/warmup_integrity.json','local_verification/partner_warmup_integrity.json'),
        ('outputs/reward_validation_v2/acceptance.json','provenance/reward_validation/acceptance.json'),
        ('A100_RL_REWARD_CONTRACT.md','A100_RL_REWARD_CONTRACT.md')]
    base.main(extra_seeds=['orchestra_joint_rl','test_orchestra_joint_rl','compare_orchestra_rl'],extra_files=extras,
        initial_checkpoint='outputs/ensemble_upgrade_v1/imported/progress_step280.pt',
        initial_destination='checkpoints/a100_step280.pt',package_version=2)


if __name__=='__main__':main()
