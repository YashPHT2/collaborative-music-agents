# Reward acceptance before joint PPO

The package includes a working 19-role PPO optimizer and rollout scheduler. It
does not include an accepted taste reward. Its bundled acceptance record FAILED.
The notebook reports this and skips RL. Partner training can run immediately.
Do not edit flags to pass the gate: validation evidence must come from a separate
experiment using fresh generated music, not just damaged reference recordings.

## Fixed interface

Provide `reward.ts`, a frozen CPU TorchScript module with signature
`forward(notes, mask, brief) -> one scalar tensor in [-1, 1]`.

- `notes`: float32 `[1,4096,6]`, sorted by onset, role, pitch. Each real row is
  `[onset / score_ticks, duration / 3840, role_id / 18, pitch / 127,
  velocity / 127, end / score_ticks]`.
- `mask`: bool `[1,4096]`; padding is false. All false represents genuine silence.
- `brief`: float32 `[1,3]` = `[bars / 32, bpm / 180, 1]` (meter is fixed 4/4).
- No note truncation. More than 4096 notes is an error. Empty scores are legal.
- The entire completed phrase receives a terminal score, propagated using team
  GAE. No reference recording enters rollouts. There is no forced participation
  reward, minimum-note floor, or claim of causal per-role credit assignment.

The directory must also contain `protocol.json`, `validation_results.json`, and
`acceptance.json`. Acceptance must bind their SHA256s, the reward artifact, the
exact supervised source checkpoint and this release's `orchestra_joint_rl.py`.
The schema is `orchestra-terminal-notes-v1`, and `roles` is the ordered list from
`orchestra_learning.ROLES`.

Acceptance fields checked: `reward_sha256`, `protocol_sha256`,
`validation_results_sha256`, `generator_sha256`, `runner_sha256`, and all seven
boolean gates: automated, reference, MIDI preference, temporal, human preference,
generated domain, and RL allowed (see `reward_check` for exact keys).

## Evidence still needed

Before acceptance, fix evaluation splits/protocol and test fresh compositions,
unmodified versus broken controls, loudness/timbre confounds, exact loops, silence,
density manipulation, role dropout, shifted harmony, displaced rhythm and unseen
composition groups. Evaluate agreement with independent listener preferences,
uncertainty and every role's contribution. Two commercial reference tracks do
not supply enough independent human-preference labels. These are requirements
for an acceptance experiment, not results already obtained by this package.

## PPO behavior and limits

19 independent small residual note actors alter the frozen supervised token
distribution within existing playable supports. The frozen 233M Transformer,
seven family adapters, planner and partner encoder remain unchanged during PPO.
Revised-mode announced drafts come from the frozen supervised source, not each
actor's future target. Final proposals see the same committed past and commit
together. A causal GRU critic encodes all 19 role IDs, including sustained notes.
Four PPO epochs use complete role/beat action log probabilities, clipping at 0.2,
reference KL 0.05, entropy 0.005, old-policy KL stopping at 0.02, actor LR 3e-5,
critic LR 1e-4 and gradient clipping 0.5. These are initial research settings,
not a claim of optimal hyperparameters. In v3, residual actors/critic update on CUDA when available;
the frozen Transformer rollout uses the GPU. Autoregressive sampling can be the
speed limit even on A100. This implementation does not batch independent rollouts.

Each complete update saves actors, critic, optimizers, RNG, source/reward/code
hashes. A mid-update interruption leaves the prior checkpoint intact. Budget and
stop requests are checked between complete episodes and updates; one episode may
overrun the budget. No candidate is automatically promoted by its training reward.
Version 3 selects on twelve predetermined validation seeds with paired bootstrap
uncertainty and per-role sounding-fraction guards; training reward does not select
weights. A guard on participation alone does not establish musical cooperation.
Use the fixed-seed before/after listening cell and held-out partner probes before
replacing the supervised listener favorite. RL checkpoints require their exact
supervised source; they are not interchangeable with format-4 supervised files.

Test status: synthetic 19-role PPO updates, replay, causal state, order invariance,
reward rejection and note accounting are unit tested locally. A100 execution and
musical preference improvement remain unverified until the notebook is run.
