# Supported ensemble: expanded data and verified supervised run

Started 2026-10-18. Live dashboard: http://127.0.0.1:8789/.

## Data

The pinned local PDMX MIDI archive supplied additional public-domain/CC0
compositions. Both archive and metadata MD5 hashes matched their pinned values.
The existing role-assignment thresholds, duration limits and split rules were
retained. Unsupported long sustains remain excluded; eligible boundary sustains
are retained as prefix context without artificial new attacks.

| Split | Eligible four-bar targets | Independent composition groups |
| --- | ---: | ---: |
| Training | 355 | 83 |
| Validation | 56 | 19 |
| Test | 35 | 7 |

The expansion adds 27 training and 13 validation groups with eligible targets.
All roles have 19 validation groups, exceeding the 10- and 12-group gates.
No new test sources were selected; the original test example IDs are unchanged.
Held-out checkpoint evaluation uses a fixed, provenance-stratified sample of up
to 12 composition groups per role from this validation pool. The test set is not
used for selection.

Source selection, source exclusions, fingerprints, deduplication and final
examples are recorded in `outputs/supported_corpus_expansion_v1/`. The original
corpus and original audit outputs remain unchanged. The final examples SHA256 is
`9e1e07b4de9d79de09b2ab1470a836338c4754bf392a1b2035a72e75a9acb854`.

## Loader verification

`supported_ensemble_loader.py` supplies contiguous four-bar supervised targets
with up to two preceding bars of context to the production teacher sampler.
Prefix beats and absent roles receive no note loss. Incoming sounding notes
remain visible. No minimum note count or forced-note rule is introduced.

The real-model verification checked 128 organ, 128 strings and 84 cello teacher
decisions, legal action support and finite nonzero LoRA gradients for each role.
It performed no optimizer steps and verified frozen-backbone integrity.
There are 25 passing focused loader, supported-window and repair tests.

## Training

The preserved initialization is `outputs/offline_ensemble_v1/revised.pt`, SHA256
`2747258de3ad0cd9750cc6b63d17fda24d106739819b54af0e08bbc14789335c`.
Its file remains unchanged. The new version-3 checkpoints go under
`outputs/supported_ensemble_run_v1/`.

The 233M Transformer backbone and residual actors stay frozen. Three rank-8
instrument LoRA adapters, the causal 48-feature bar planner and partner-draft
conditioning are trained with masked category cross-entropy and planner MSE.
Instrument and composition sampling are balanced; supported categories and
onset/non-onset examples are balanced within each sampled phrase. Exact role
labels carry weight 1; functional transfers carry weight 0.35.

The bounded recipe is 128 optimizer updates for participation/phrases, followed
by up to 128 updates for partner revision. Four phrases per instrument provide
up to 192 beat examples per update. The configured microbatch capacity is 128;
the current per-role pool has at most 64 examples, so this does not imply 128
examples in every forward pass. Actual loss and example counts are logged each
update. The combined run has a one-hour budget, with a training cutoff at 85%
and time checks between complete updates. Draft generation and validation can
make an individual update/check exceed a cutoff.

Revision uses frozen stage-start private proposals generated from the same
composition and real prefix. A rotating subset samples composition groups;
caches are refreshed at the stage boundary. An agent receives announced
partner drafts, never its own future target notes.

Checkpoint selection requires lower balanced event loss for every role and
at most 2% note-attribute regression, with additional exact/transfer checks.
An unsuccessful revision retains the earlier selected checkpoint. The first
eligible participation checkpoint was update 32; at update 64 the event losses
were 16.0%, 10.0% and 1.9% lower for organ, strings and cello respectively.
These results describe held-out prediction, not listener preference or proven
free-running cooperation. The live logs contain subsequent measurements.

## Inspect and resume

```powershell
.\.venv\Scripts\python.exe -u watch_supported_training.py --port 8789
```

The run is already started. Do not launch a second training process. If it is
interrupted, resume with the identical recipe:

```powershell
.\.venv\Scripts\python.exe -u train_supported_ensemble.py --steps 128 --hours 1 --microbatch 128 --phrases-per-role 4 --resume
```

Progress and selected weights, verification reports, source/data hashes, and
per-role validation metrics are saved in the run directory. This experiment
does not train PPO or DPO and does not change the failed audio reward gate.
Listening quality and held-out generation/cooperation probes remain separate
checks after checkpoint selection.

## Completed evaluation

Both 128-update stages finished in 24.2 minutes. Participation selected update 128; partner revision selected update 80. The three matched blind clips and all instrument solos are at http://127.0.0.1:8790/listen.html. Twenty-one held-out partner cases (seven groups per role) are complete. Correct-versus-missing benefits are inconsistent; strings regress on the consonance diagnostic relative to the earlier model. Cooperation and listening preference remain unproven. See outputs/supported_comparison_v1/RESULTS.md and evaluation.json.
