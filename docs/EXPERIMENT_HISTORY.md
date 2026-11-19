# Collaborative music experiments

## Offline ensemble composition campaign (v2)

The new supervised pipeline learns general ensemble syntax, specializes three
rank-8 instrument adapters, then learns bar descriptions and partner-draft
conditioning. A cinematic-focus stage prefers sustained, harmonically varied
windows by explicit structural heuristics; this is a weak style filter, not
verified soundtrack labels or Hans Zimmer training data. Each two-bar draft is private; one synchronous revision reads only
the other instruments' frozen drafts. All final notes remain agent decisions.
The shared backbone stays frozen. This campaign adds no PPO or taste reward.

```powershell
.\.venv\Scripts\python.exe run_composition_campaign.py
# Resume an interrupted campaign with the same saved recipe:
.\.venv\Scripts\python.exe run_composition_campaign.py --resume
# Individual stages:
.\.venv\Scripts\python.exe composition_data.py
.\.venv\Scripts\python.exe train_offline_ensemble.py
.\.venv\Scripts\python.exe train_offline_ensemble.py --resume
.\.venv\Scripts\python.exe evaluate_offline_ensemble.py
.\.venv\Scripts\python.exe serve_evaluation.py --report offline --port 8777
```

Data uses pinned PDMX record 15571083 (public-domain/CC0 entries without recorded
license conflicts, deduplicated arrangements) plus official Slakh splits. Broader
instrument labels remain intact for general pretraining. Specialization currently
uses only target-instrument parts in context; broader non-target partners are
learned in general pretraining, not mislabelled as target instruments. No Zimmer
recordings are included. Exact/transposition-normalized score and aligned passage
deduplication is implemented; approximate or shifted duplicate arrangements can
remain. Missing target instruments receive no specialized loss.

The data gate requires ten validation compositions per target role. `--pilot`
allows engineering trials below that gate and must not establish cooperation.
The default campaign runs 128 accumulated optimizer updates per stage with a
cumulative eight-hour training limit. Resume preserves the original limit,
optimizer and RNG state. `--steps 2 --pilot --output outputs/offline_smoke`
is a short integration trial. Longer compositions remain experimental until the
eight-bar listening and intervention gates pass.

Checkpoints use format 2 and separate baseline (`cinematic.pt`), planned and
revised stages. `latest.pt` is the endpoint, not a proven quality winner. Existing
format-1 checkpoints and commands are unchanged. Generate with:

```powershell
.\.venv\Scripts\python.exe compose_offline_ensemble.py --checkpoint outputs/offline_ensemble_v1/revised.pt --output outputs/offline_new_seed --bars 8
.\.venv\Scripts\python.exe score_ensemble_ratings.py --output outputs/offline_ensemble_v1 vote1.json vote2.json vote3.json
.\.venv\Scripts\python.exe -m unittest discover
.\.venv\Scripts\python.exe verify_offline_integration.py
```

The anonymous listening panel contains four seeds for each of three methods;
ratings are individual 1–5 scores, exported locally as JSON. A minimum of three
complete ballots, +0.5 enjoyment and ensemble-fit gains, and no development
regression are exploratory project thresholds, not statistical proof. Technical
checks, engineering rewards, likelihood and listener preference remain separate.

## Autonomous note-level organ, strings and cello

Continue with validation-based checkpoint selection:
`.\.venv\Scripts\python.exe continue_note_trio.py --updates 32`.
Resume interrupted work with the same command plus `--resume`. This preserves
the first session and writes `outputs/note_trio_v2`. Four fixed validation seeds
select among the starting checkpoint and later milestones; all instruments must
participate without single-pitch collapse. Two unsuccessful validation checks
stop continuation. The reward and note sampler remain unchanged. A fresh final
listening seed compares the selected model with the starting model. Use
`serve_evaluation.py --report notes-more --port 8774` to listen. Selection by
musical-rule score is not proof of improved enjoyment.

`note_trio.py` is the direct note-generating implementation. Three new independent
residual musicians adapt frozen SkyTNT distributions for note/rest, onset, pitch,
velocity and duration. All propose privately from completed ensemble history and
commit together. Every emitted note is saved in `*_decisions.json` and checked
against the exact MIDI; there is no chord progression, motif seed/copy controller,
phrase bank, rhythmic template or programmed build. Only instrument constraints,
quantization, initial key, tempo and maximum length are specified. Joint PPO uses
a shared, bounded engineering reward, not a validated human quality evaluator.

First session: `.\.venv\Scripts\python.exe note_trio.py --updates 8`.
Continue: `.\.venv\Scripts\python.exe note_trio.py --updates 64 --resume --max-minutes 480`.
The same seed and episode count must be used to resume. Checkpoints include all
three actors/optimizers, critic, initial control, RNG state and cumulative runtime;
training stops after eight cumulative hours. Results are in `outputs/note_trio_v1`.
Run `serve_evaluation.py --report notes --port 8773` for the trained 60-second mix,
solo parts, matched untrained control and partner-hidden diagnostic. These are
initial/latest results; numerical reward improvement does not establish music
quality or a learned cooperation advantage. `verify_note_trio_integration.py`
checks real-model generation, order invariance and exact likelihood replay.

## Original cinematic organ, strings and cello trio

`cinematic_trio.py` generates **First Light**, a 60-second original study with
GM church organ, upper strings and cello. A frozen SkyTNT model proposes seed
motif pitches; three new, independent bar-level phrase policies receive a short
joint PPO pilot. Policies choose motif transformations, voicing/register, cello
pulse and dynamics from a bounded arranging grammar. Harmony, rhythmic templates,
theme reprises and the expression/build curve are engineered. This is a phrase
policy experiment, not instrument-specific Transformer fine-tuning or evidence
of emergent orchestration or better music. No commercial melody is supplied.

Run `.\.venv\Scripts\python.exe cinematic_trio.py`, then
`.\.venv\Scripts\python.exe serve_evaluation.py --report cinematic --port 8772`.
Files are in `outputs/cinematic_trio_v1/`: full mix, exact MIDI and all three
stems, checkpoint, trace, motif provenance, controls and verification report.
Each WAV/MP3 is mastered to 60 seconds; performance MIDI has 57.6 seconds of music.
Controls are retained as MIDI, without a claim that PPO improves the result.

## Phrase-conditioned adaptive performance

Run `adaptive_performance.py` to fit a small drum prior on the official Groove
MIDI Dataset train split, train the three residual actors for eight additional
four-bar episodes under sparse role constraints, then generate a 30-second
performance and a held-intent control. Results go to `outputs/adaptive_band_v2`.
The MIDI-only dataset download is 3.11 MB, verified against its official SHA256.
The prior is categorical by drum position and stroke, not a fine-tuned Transformer.
Credit: Gillick, Roberts, Engel, Eck, Bamman, ICML 2019; CC BY 4.0;
https://magenta.tensorflow.org/datasets/groove.

The piano proposes a bar and broadcasts parsed musical intent. The followers
sample their own parts from that announced plan and committed past. This is
offline bar planning, not unrestricted simultaneous live improvisation.
The generated opening motif is explicitly reprised and rhythmically developed
by an arrangement controller, with a scheduled two-semitone change at 14.4s.
Role/register/scale constraints and quieter velocity ranges are also engineered.
The control keeps the exact same piano and seed while holding follower intent
and hiding changed piano observations after the transition. Matching prefixes
and changed followers establish conditioning, not a proven musical advantage.

Run `prepare_adaptive_demo.py`, then
`serve_evaluation.py --report adaptive --port 8771` for the interactive player.
It includes solo stems, the control, and a note timeline. No learned human
preference or ensemble-quality evaluator is claimed from our few ratings.

## Reward v2 and blind ensemble calibration

`band_reward.py` adds a causal, bounded heuristic combining observed tension
resolution, bass/kick compatibility, drum-pattern continuity, top-voice melodic
motifs over several bars, smooth role-specific density targets, participation
over phrases, and limited degeneracy penalties. Useful bass/drum ostinatos and
brief rests are permitted. The original reward remains the default for existing
experiments. Select the new version through
`Brief(..., reward_version='band-heuristic-v2')`.

`reward_experiment.py` continues all three actors for 16 new four-bar episodes
from the prior update-14 checkpoint, refitting a fresh value critic for the new
scale. It saves into `outputs/band_reward_v2`; previous runs are preserved.
This reward is hand-designed. No learned musical-quality or human-preference
model is claimed; the upcoming ratings are calibration data.

`prepare_band_ballot.py` generates five fresh eight-bar seeds for the previous
and new reward policies, splits each into two four-bar passages, and packages
30 blind comparisons. Ten compare training methods; twenty compare unaltered
parts with timing or bar-order edits. Edits are diagnostic hypotheses, not
ground-truth bad music. Ten passages from five seeds are **not 30 independent
songs**. Rendering and loudness treatment match across versions. The participant
ZIP contains embedded audio and instructions; method mappings, source files,
and reward scores remain outside it. Vote labels are specific to this ballot.

Run `decode_band_votes.py vote1.json vote2.json` to decode returned JSON exports.
Keep the organizer mapping until all votes are collected. Use enjoyment,
coherence and ensemble-fit judgments separately, preserve ties and neither-good
answers, and evaluate generalization on new songs/listeners before learning a
preference reward. Reward-v1 and reward-v2 numerical values have different
scales and should not be compared directly as proof of better music.

## Three independently learning musicians

Continue the first checkpoint with periodic validation and a longer horizon:

```powershell
.\.venv\Scripts\python.exe continue_band.py --updates 14
# Resume an interrupted continuation:
.\.venv\Scripts\python.exe continue_band.py --updates 14 --resume
```

This writes `outputs/band_joint_v2` and preserves the first trial. It continues
the actors, critic, and optimizers, with four episodes per update. Training
length grows from four to eight bars after update ten. Four fixed fresh
validation seeds select `best.pt` every four updates, with the initial untrained
policy also eligible. Selection requires all three instruments in every clip.
`latest.pt` is resumable training state; `best.pt` is for generation. The final
selected and initial comparisons use a separate new 16-bar listening seed.
Neither selection nor a higher heuristic score establishes human preference.

`band.py` implements piano, bass, and drums as three separate trainable residual
policies over one frozen SkyTNT backbone. All propose one beat from committed
history; their proposals are committed together. Each learns rest, onset, pitch,
velocity, and duration. A shared training-only critic evaluates state, and
MAPPO-style PPO updates all three from shared ensemble rewards.

First local joint training trial (8 four-bar episodes, then 16-bar exports):

```powershell
.\band.ps1
# Resume to a larger total number of updates:
.\band.ps1 -Updates 8 -Resume
# Full length curriculum, 4 then 8 then 16 bars; resumable two-hour sessions:
.\.venv\Scripts\python.exe train_band.py train --updates 64 --episodes 8 --output outputs/band_curriculum_v1
# Preview, with working audio seeking:
.\.venv\Scripts\python.exe serve_evaluation.py --report band --port 8768
```

Keep `--episodes` fixed when resuming: it determines the training seed sequence.
The full curriculum command above must use a **new output directory** when
changing episodes from the initial four-episode trial, rather than resuming it.
For example use `--output outputs/band_curriculum_v1` without `--resume` on its
first run, then repeat with `--resume` for later sessions.

Outputs in `outputs/band_joint_v1/` contain initial/latest/per-update checkpoints,
training logs, two fixed validation seeds, and a fresh demonstration seed.
`listen.html` compares initial, trained, and trained-without-partners conditions;
each includes the full mix and exact piano/bass/drum stems. This page is labeled
for inspection, rather than a blind preference study. No musical-quality or
originality benefit is claimed from the first trial.

```powershell
.\.venv\Scripts\python.exe train_band.py generate --checkpoint outputs/band_joint_v1/latest.pt --output outputs/band_new_seed --seed 2028 --bars 16
.\.venv\Scripts\python.exe -m unittest test_band.py
```

The reward is versioned `band-heuristic-v1`: equal-weight harmony, bass/kick
alignment, pitch-interval motif recurrence, and arrangement energy, with
degeneracy penalties. Bar rewards are sparse and causal. It is an engineering
proxy, not a validated music critic. Training and evaluation use the same
sampling mixture, masks, active-note ledger, and rendering. Pitch/duration
constraints prevent repeated-pitch overlaps before commitment, preserving the
same notes in observations, rewards, and export. Tempo is fixed within a clip;
all notes are generated, with no supplied melody or chord progression.

## Pretrained piano generation

This first milestone runs SkyTNT's pretrained MIDI-event Transformer locally.
There is no RL, fine-tuning, or supplied melody in this experiment. Musical
notes, polyphony, rhythm, duration and velocity are sampled from the checkpoint.

## Generate and listen

From PowerShell in this folder:

```powershell
.\generate.ps1 -Seed 42 -Bars 8 -Bpm 100
```

Or without PowerShell script execution:

```powershell
.\.venv\Scripts\python.exe generate_piano.py --seed 42 --bars 8 --bpm 100
```

Results appear in `outputs/`: MIDI, WAV, MP3 (when ffmpeg is installed), and
JSON recording the checkpoint revision, generation settings and measured stats.
Change the seed to generate another sample. Reusing a seed aims for reproducibility
in this environment; other hardware/library versions can produce different samples.
Use `--device cpu` if CUDA is unavailable, or `--no-audio` for MIDI only.

The prompt specifies acoustic grand piano, C major, 4/4 and 100 BPM by default.
Key is a conditioning signal, not a forced pitch filter. The sampler fixes piano
channel/track and disallows later instrument/tempo/controller changes. It samples
note events or EOS, up to the requested bar boundary or an event cap. It does not
guarantee a cadence or exactly eight complete bars. Notes crossing the requested
end are trimmed; FluidSynth adds a release tail to the rendered audio.

## Melody-conditioned accompaniment baseline

```powershell
.\accompany.ps1 -Seed 42 -Bpm 100
# Or:
.\.venv\Scripts\python.exe accompany.py --seed 42
```

The experiment supplies a scripted eight-bar melody, one note per beat, and
samples a separate piano accompaniment with frozen pretrained weights. At bar 5
(9.6 seconds at 100 BPM), the changed variant transposes the melody from C major
to D major. It does not give the policy a new key-signature event. Both runs use
the same seed and have identical input before the intervention.

Listen in `outputs/accompaniment/seed42/`:

- `steady_duet.mp3`: melody in C major throughout, with generated accompaniment.
- `changed_duet.mp3`: melody changes at bar 5, with newly generated responses.
- `changed_frozen_accompaniment.mp3`: the changed melody over the accompaniment
  from the steady run, without recomputing its response.
- `melody_only.mp3` and `changed_melody_only.mp3`: isolated reference melodies.
- Matching `.mid` files preserve melody and accompaniment on separate channels.
- `comparison.json`: generation settings, timings and descriptive statistics.

This is a causal conditioning baseline, not a trained two-agent system or RL.
At each input onset, the current melody note is injected into the chronological
model context; its duration is known, but future melody pitches are hidden.
The scheduler knows the next input time. It discards candidate accompaniment
events at or beyond that boundary; EOS also means waiting for the next input.
Committed accompaniment notes may sustain across a later melody onset.

To keep roles distinct, accompaniment is constrained to MIDI pitches 36..66,
velocities 32..95, positive durations of at most four beats, and channel 1.
A cap of 16 generated notes per input protects against excessive same-time
events. These constraints and the scheduler are part of the baseline; any later
RL comparison must use the same settings. Repeated same-pitch overlaps are
trimmed during output cleanup. The fixed melody stays on channel 0 unmodified.

The reported second-half D-major pitch-class fraction is only descriptive:
it is not a quality measure, a chord-fit score, or proof of successful adaptation.
Matching prefixes and different second halves establish that the generation
changes after the intervention, not that the new music is better. Multiple
seeds and listening comparisons are needed to assess improvement.

Scheduler checks:

```powershell
.\.venv\Scripts\python.exe -m unittest test_accompaniment.py
```

## Evaluation suite

```powershell
.\evaluate.ps1
# Score and save MIDI without rendering listening clips:
.\evaluate.ps1 -NoAudio
```

`evaluation_cases.py` defines two scripted melody families, five conditions
(steady, key shift, changed melody, changed rhythm, combined key/rhythm), and
three seeds per family. The second family has separate seeds and is designated
reserved evaluation for future reward tuning. This is a small diagnostic suite,
not a representative or verified pretraining-independent test set.

Thirty model generations produce 54 MIDI comparisons: responding accompaniment
and, for intervention cases, the steady run's accompaniment kept fixed. Eighteen
clips are rendered from the first seed of each case for listening. Each file is
checked for exact melody preservation, balanced note events, and valid audio.

Open `outputs/evaluation_v1/listen.html` in a browser for eight A/B pairs with
hidden method labels, a jump to the change at 9.6 seconds, and local rating export.
The page makes no external network requests; audio files are relative to the report.
Ratings are saved in browser storage where available and must be exported to
share them. No preference ratings are invented or inferred from metrics.

For a local browser preview, serve only the report directory:

```powershell
.\.venv\Scripts\python.exe serve_evaluation.py
```

Then open http://127.0.0.1:8765/listen.html. Stop the server with Ctrl+C.
Use this server rather than Python's basic `http.server`: audio seeking in the
in-app browser requires byte-range responses. Test it with
`.\.venv\Scripts\python.exe -m unittest test_audio_server.py`.

`REPORT.md` explains the paired results and limitations. `metrics.csv` contains
before/after measurements for scale/chord fit, rough intervals, note density,
coverage, repetition and onset alignment. `paired_summary.json` records paired
deltas over three seeds, including min/max. `sanity.json` scores silence, a
repeated note, a scale and chromatic flooding to expose reward exploits. No
single musical-quality score or training reward is claimed.

Run the checks:

```powershell
.\.venv\Scripts\python.exe -m unittest test_accompaniment.py test_evaluation.py
```

## Team voting handoff

Run `.\.venv\Scripts\python.exe prepare_team_vote.py` to build
`outputs/team_handoff/Piano_Team_Vote_v1.zip`. Send that ZIP as a WhatsApp
document with `WHATSAPP_MESSAGE.txt` from the same folder. Teammates extract it
on a laptop and open `OPEN_ME.html`; all 14 MP3 clips are embedded, so no server
or Python is required. Seven distinct pairs remain after excluding the verified
identical pair. A/B order is balanced and specific to this ballot, so do not mix
letter-only responses from the earlier listening page with team-ballot votes.

The page exports named JSON votes or copyable WhatsApp text. Nothing is submitted
automatically. `organizer_manifest.json` records the A/B mapping and audio hashes
outside the participant ZIP. These are informal team preferences; the reference
method can be revealed in the page, so blinding is voluntary rather than secure.

## First PPO pilot

The submitted team-v1 ballot is recorded in `outputs/feedback/user_ballot_v1.json`.
It decodes to two responding preferences, two frozen preferences, two ties and
one neither-good response. This is one listener, not seven independent listeners,
and does not justify training a preference reward model. The earlier reserved
theme B is now reviewed; do not present it as unseen for future decisions.

```powershell
.\train.ps1 -Updates 8 -EpisodesPerUpdate 4
# Preview after training completes:
.\.venv\Scripts\python.exe serve_evaluation.py --report ppo --port 8767
```

Open http://127.0.0.1:8767/listen.html. The adapter experiment uses new A/B labels,
so old letter-only votes do not apply to it.

This is an actual small PPO **output-adapter pilot**, not fine-tuning of the
233M-parameter foundation model and not multi-agent training. A zero-initialized
residual actor and critic use frozen token probabilities plus current/past
musical features. They learn note-vs-wait and pitch decisions; event timing,
duration and velocity remain frozen-model choices. No annotated key, chord or
future melody is supplied to the adapter. The base model is frozen and all base
tensors are checked for unchanged sums/squared sums after the experiment.

On-policy rollouts retain exact sampled log probabilities and fixed reference
support for PPO ratio computation. The event distribution includes a fixed
2.5% WAIT exploration floor. A matched untrained policy uses that same change;
the original sampler is also evaluated so sampler changes are not credited to
RL. Pitch support is filtered using the frozen distribution, then kept fixed
when training residual logits. Complete episodes use GAE and clipped PPO with
a critic, reference KL, entropy regularization, gradient clipping and a KL stop.

The hand-designed local proxy combines sounding coverage, compatibility with
the current melody, excess density and pitch dominance. It is explicitly not
an inferred human preference, nor proof of music quality. Fixture tests check
silence, repeated notes and flooding. The proxy may still discourage useful
rests and tension or be exploited in other ways.
Training/validation rewards are computed online on committed notes before final
overlap cleanup. Final-test proxy values are recomputed on exported notes;
duplicate-pitch cleanup can shorten earlier notes, so the two need not match.

By default 32 four-bar synthetic episodes train one policy seed. Four separate
validation melodies select a checkpoint (including the untrained initial state
if none improves the validation proxy). Four fresh eight-bar final tests are
then evaluated once. Train/validation/test seed namespaces are separate; all
share a synthetic melody generator, so this is only an engineering pilot.
No final-test score is used to choose the checkpoint. Rerank-3 selects the best
of three untrained samples under the same proxy at three times sampling cost.

Artifacts in `outputs/ppo_pilot_v1/`: selected and final adapter checkpoints,
training log, corpus definitions, experiment manifest, per-method MIDI/audio,
metrics, verification, diagnostic report, and a fresh blind listening page.
Higher heuristic scores alone do not establish better-sounding music.

```powershell
.\.venv\Scripts\python.exe -m unittest test_ppo_pilot.py test_accompaniment.py test_evaluation.py test_audio_server.py
```

## Shared-context two-agent duet

```powershell
.\duet.ps1 -Seed 2027 -Bars 8 -Bpm 100
# Or:
.\.venv\Scripts\python.exe collaborative_duet.py --seed 2027 --bars 8
```

Both the melody and accompaniment are now AI-generated. There is no scripted
melody input. Two inference agents share the frozen SkyTNT checkpoint but have
separate random streams, register masks, channels/tracks and per-bar contexts.
The melody role is monophonic, MIDI pitches 67..84. Accompaniment uses pitches
36..66 and can play chords. Both use acoustic piano. Header conditioning is
C major, 4/4 and the requested tempo; pitches are not restricted to a scale.

For each bar, melody composes from both prior exported parts. Accompaniment
then hears melody notes causally at their onsets, filling intervals between
them. The completed parts enter the shared ledger before the next bar. This
means accompaniment can affect the following melody phrase. The scheduler knows
the next melody onset time but does not expose its pitch early. This is scheduled
bar-based composition, not a claim of unconstrained simultaneous improvisation.

Files in `outputs/collaborative_duet/seed2027/`:

- `shared_duet.mid/.mp3/.wav`: two AI parts with shared context.
- `independent_duet.*`: partner history/current inputs removed, same role masks
  and seed derivation. This control removes both one-way and feedback conditioning.
- `melody_only.*` and `accompaniment_only.*`: exact isolated generated tracks.
- `context_trace.json`: history hashes, partner-event counts, role seeds and caps.
- `sensitivity_probe.json`: replay one melody bar with the same seed, then repeat
  after transposing the preceding accompaniment bar in its observation.
- `shared_events.json`, `independent_events.json`, `manifest.json`, `verification.json`,
  `REPORT.md`: generated events, settings, audio/MIDI checks and interpretation.

The counterfactual probe checks sensitivity to partner input, not musical quality.
If its output is unchanged, that result is retained rather than searching until
a positive example appears. Both roles remain untrained: separate role-conditioned
inference agents are not evidence of learned multi-agent cooperation. The earlier
PPO pilot is not used. Joint RL, comparisons against one-way conditioning and a
centralized generator, and originality checks remain research work.

```powershell
.\.venv\Scripts\python.exe -m unittest test_collaborative_duet.py test_accompaniment.py
```

## Supervised organ / strings / cello ensemble pilot

`ensemble_data.py` downloads a checksum-pinned Slakh MIDI archive, uses the
official redux exclusions and composition splits, audits actual instrument
programs, and excludes canonical exact duplicates across splits. Organ programs
16–20, upper strings 40/41/48/49, and cello 42 are used; bass and piano are not
substituted. Pieces containing at least two real roles are eligible. Dataset
coverage, source revisions and attribution are in `assets/trio_ensemble/audit.json`.

`supervised_ensemble.py` trains three separate rank-8 LoRA adapters on aligned
four-bar ensemble windows. Teacher forcing uses the exact generation masks,
temperature and exploration mixture. Each instrument receives only committed
past partner notes plus its own earlier notes in the current beat. The corpus
has no reliable key labels, so key-signature prompting is omitted consistently
in this experiment. Missing source instruments have no supervised loss; they
are never labelled as rests. Examples are balanced across available roles.
Teacher-data preprocessing prunes source voices to the agents' event caps and
polyphony, retaining upper string chord voices and a lowest monophonic cello
voice with overlapping notes shortened. It never invents pitches or onsets;
source/retained note counts are recorded. Remaining invalid windows are rejected.
Validation selects epochs independently for each instrument;
an unadapted checkpoint remains eligible if validation does not improve.

The subsequent joint PPO pilot trains separate residual note policies using
`sequence_critic.py`: a GRU over up to 256 ordered notes from the preceding eight
bars, including note timing, pitch, duration, velocity and instrument, plus the
brief. The critic estimates the existing engineering reward; it is not a human
musical-taste model. The pretrained backbone remains frozen throughout; LoRA
adapters remain frozen during PPO. Every exported note is still a policy action.

```powershell
.\.venv\Scripts\python.exe ensemble_data.py
.\.venv\Scripts\python.exe supervised_ensemble.py --examples 256 --epochs 2 --ppo-updates 2
.\.venv\Scripts\python.exe serve_evaluation.py --report supervised --port 8775
.\.venv\Scripts\python.exe -m unittest test_supervised_ensemble.py
```

Use `--resume` after interruption, or `--output PATH` for a separate experiment.
Supervised progress is saved at epoch boundaries, PPO progress after each joint
update. The default directory is `outputs/supervised_trio_v1`: `supervised.pt`
contains the selected instrument adapters, `latest.pt` adds the residual actors,
sequence critic and optimizer states, and `REPORT.md` / `verification.json`
describe actual training counts, validation losses and frozen-weight checks.
The player offers the supervised-plus-PPO mix and solos, and the supervised-only
mix with the same seed. Lower teacher-forcing NLL does not establish better music;
listener preference, longer-form composition and generalization remain open.

Generate another performance from the saved weights without training again:

```powershell
.\.venv\Scripts\python.exe compose_ensemble.py --seed 9601002 --bars 24 --output outputs/ensemble_seed9601002
```

`verify_supervised_integration.py` checks isolated real-backbone adapter updates,
frozen backbone weights, frozen adapters during PPO, actor/critic learning,
execution-order independence and exact saved-action likelihood replay.

## Partner-response curriculum, then joint training

`train_partner_response.py` continues `outputs/supervised_trio_v1/latest.pt`:

1. Six warm-up epochs teach each residual policy to predict actual source notes
   from correct causal partner history. A small ranking loss compares aligned
   and different-song partner history. Positive validation likelihood is gated;
   worsening shuffled-context predictions alone is not a quality result.
2. Four PPO updates per role, four episodes per update. The learner generates
   its own three-bar continuation after a one-bar dataset prefix. The available
   one or two partner parts stay fixed. Only the learner policy and its separate
   response critic update. Missing source instruments are never fabricated.
3. Eight joint PPO updates, four episodes per update, progressing from four to
   eight bars. All three agents generate from an empty ledger, propose privately,
   and commit together. A small supervised training-data anchor is retained
   during both PPO stages to reduce forgetting.

The source backbone and the three supervised LoRA adapters stay frozen. The
three residual note policies learn, and joint training uses the sequence-aware
GRU critic. The bounded engineering reward measures cross-role consonance,
sounding coverage, density and voice leading for available roles, with a
repetition penalty. It is not a validated measure of taste or a counterfactual
credit-assignment method.

```powershell
.\.venv\Scripts\python.exe train_partner_response.py
.\.venv\Scripts\python.exe train_partner_response.py --resume
.\.venv\Scripts\python.exe serve_evaluation.py --report partners --port 8776
.\.venv\Scripts\python.exe compose_ensemble.py --checkpoint outputs/partner_response_v1/latest.pt --seed 9618002 --output outputs/partner_composition_seed9618002
```

Default outputs are `outputs/partner_response_v1`. The source experiment is
preserved. `prepare_partner_cache.py` also runs automatically when needed and
records actual window and beat counts; broader source coverage does not add new
compositions. Exact projected training/validation passages and shared song IDs
are excluded. `progress.pt` resumes stages, optimizers and RNG states;
`before.pt`, `warmup.pt`, `response.pt`, and `latest.pt` are inference bundles.

`partner_evaluation.json` measures the **full residual policy** on identical
teacher targets with correct, absent and different-song past partner notes.
`closed_loop_evaluation.json` additionally generates responses with the actual
dataset partners held fixed while varying only the learner's observations and
keeping its seed fixed. Validation and engineering rewards are descriptive;
listener preference and consistent learned cooperation are not assumed.
The final 60-second listening performances contain only agent-generated notes,
with no dataset prefix or fixed parts. The player defaults to the final mix and
offers solos and the two earlier training stages without a voting questionnaire.

```powershell
.\.venv\Scripts\python.exe -m unittest test_partner_response.py
.\.venv\Scripts\python.exe verify_partner_integration.py
```

The integration check verifies full-policy teacher support and probability
replay within BF16 numerical tolerance, isolated learner updates, frozen source
weights, order-independent joint generation, and direct note-action provenance.

## Trio participation repair (version 3)

The repair starts from the previous `planned.pt` adapters, adds a 16-position
bar rhythm histogram, and trains on contiguous two-bar phrases. Instrument and
composition sampling are balanced; onset and non-onset decisions receive equal
weight within each loss category. Sustained notes, true silence and missing
instrument labels are audited separately, including labeled parts excluded by
the existing eight-note window eligibility threshold. Training uses stable masked
cross-entropy; validation reports the actual sampler's 2% exploration mixture.
The note decoder remains teacher
forced; generated history is introduced to the planner up to 50% of examples.

```powershell
.\.venv\Scripts\python.exe run_ensemble_repair.py
# Resume the same recipe after an interruption:
.\.venv\Scripts\python.exe run_ensemble_repair.py --resume
.\.venv\Scripts\python.exe serve_evaluation.py --report repair --port 8778
.\.venv\Scripts\python.exe score_repair_ratings.py path\to\trio-repair-votes.json
```

Two stages are capped at 512 updates each and share a three-hour training/evaluation
budget, reserving 45 minutes for evaluation. The selection gate requires lower
balanced event loss for every role and at most 2% note-attribute loss regression.
A failed revision stage retains the earlier checkpoint, including its generation
mode. All old version-2 checkpoints and listening pages remain usable.

`outputs/ensemble_repair_v1` contains the data audit, selected checkpoints,
12 fixed seeds per condition, held-out fixed-partner probes and a three-clip player
with solos. The clips use a predetermined seed, not a best-sounding sample.
`compose_offline_ensemble.py --checkpoint outputs/ensemble_repair_v1/latest.pt
--output outputs/my_repair_piece --profile warm_build` generates a new piece.
The profile interpolates density/register/velocity targets between training-set
35th and 65th percentiles. It is an arrangement control, not learned emotion;
it supplies no notes, melody or chord progression. Human preference and each
instrument's cooperation are reported separately from imitation loss.

## Joint multi-agent RL after the foundation

`queue_joint_ensemble_rl.py` waits for the active supervised repair and its
evaluation to finish, runs a real-model one-update smoke test, then starts
`joint_ensemble_rl.py`. The RL stage has a separate one-hour limit (the smoke
test is capped at six minutes), remaining within the project's eight-hour ceiling.
The queue uses an exclusive lock and never runs alongside the foundation GPU job.

All three separate residual note policies receive PPO updates from jointly
generated performances, with a centralized sequence critic, shared GAE returns,
clipped policy ratios, entropy regularization and a KL anchor to the frozen
supervised distribution. The LoRA adapters, backbone, planner and conditioning
network stay fixed. This is policy learning over the foundation, not additional
dataset fine-tuning. Actors see committed partner history, propose privately,
and commit together. The online one-pass RL baseline is evaluated under the
same schedule; it is distinct from the previous offline two-pass revision.

Rewards use the existing harmony/coverage/density/voice-leading/degeneracy
diagnostics with an extra exact-unison penalty. These are transparent engineering
proxies, not a trained human preference reward. No claim of musical improvement
or cooperative credit assignment follows from increasing reward alone.
Validation requires higher shared reward and no role diagnostic falling more
than 0.02 below its initial value; otherwise the selected policy remains initial.
Held-out seeds compare missing/mismatched partner observations and identical
rendering for the initial and selected policies, with instrument solos.

```powershell
.\.venv\Scripts\python.exe queue_joint_ensemble_rl.py
.\.venv\Scripts\python.exe joint_ensemble_rl.py --resume
.\.venv\Scripts\python.exe serve_evaluation.py --report joint-rl --port 8780
```

The live training page on port 8779 shows the RL queue and, once running, each
actor's PPO loss and the rollout reward. Results are saved separately in
`outputs/joint_ensemble_rl_v1`. They never overwrite the supervised panel.

## Four-bar melodic phrase experiment

`train_ensemble_repair.py --phrase-focus` extends supervision from two to four
bars and supplements masked note cross-entropy with a differentiable pitch
interval contour loss (weight 0.15). Real target notes remain unchanged.
The curriculum draws 75% of composition selections from groups with preferred
material, retaining general examples and rests. Melodic passages require at
least 12 onsets, five pitches, stepwise movement and a recurring non-constant
interval trigram. Cello examples favour varied pitches with sustained notes.
These are structural heuristics, not emotion or soundtrack genre annotations.

The current `outputs/phrase_ensemble_v1` experiment starts from the older v2
checkpoint with a weight-preserving upgrade, uses 128 updates per stage, and
then runs partner-draft supervision. A fresh six-bar demo compares the old
initialization and trained checkpoint with seed 9860407 and identical rendering.
Neither user-provided commercial recording is training data. Motif statistics,
per-role participation, selection outcomes and small held-out partner probes
are reported in `demo_verification.json`; listener preference is still required.

```powershell
.\.venv\Scripts\python.exe train_ensemble_repair.py --source outputs/offline_ensemble_v1/revised.pt --output outputs/phrase_ensemble_v1 --phrase-focus --steps 128 --microbatch 32 --phrases-per-role 2 --hours 0.75
.\.venv\Scripts\python.exe evaluate_phrase_demo.py
.\.venv\Scripts\python.exe serve_evaluation.py --directory outputs/phrase_ensemble_v1 --port 8783
```

## Reference reward validation before RL

`validate_reference_reward.py` builds a fixed feature-based reward from disjoint
calibration excerpts in the two supplied reference recordings, then evaluates
nine held-out sections (36 original/disruption comparisons plus controls).
The structured candidate combines rhythm, tonal motion, development and a small
timbre component. A static-feature baseline is evaluated under the same protocol.
No policy training or human-preference reward fitting occurs.

```powershell
.\.venv\Scripts\python.exe -m unittest test_reference_reward_validation -v
.\.venv\Scripts\python.exe validate_reference_reward.py
.\.venv\Scripts\python.exe serve_evaluation.py --directory outputs/reward_validation_v1 --port 8785
```

The first run is preserved in `outputs/reward_validation_v1`. Rerunning into the
same version is rejected. The candidate failed: 55.6% of assumed disruptions
ranked below the original, and two noise samples outranked originals. The static
baseline reached 75%, also below the predeclared acceptance gate. Volume and
mild-room controls were comparatively stable. `acceptance.json` keeps RL use
disabled; synthetic labels require listener review and the reward has not been
validated on generated MIDI renderings. These are within-recording tests on two
pieces, not evidence of general human-taste recognition. Any revised reward
needs a new experiment version and untouched evaluation material.

## Reward validation v2: preference is separate from resemblance

`outputs/reward_validation_v2/RESEARCH.md` records the primary research and the
deployment choice: frozen TuneJury **A1** human-preference scoring, separate
reference-style diagnostics, and a small temporal MIDI contrast ranker with
individual organ/strings/cello outputs. This does not train a composition policy.

```powershell
.\.venv\Scripts\python.exe prepare_preference_reward.py
.\.venv\Scripts\python.exe verify_preference_reward.py
.\.venv\Scripts\python.exe validate_music_reward_v2.py
.\.venv\Scripts\python.exe -m unittest test_preference_reward test_reference_reward_validation test_joint_ensemble_rl -v
```

Audio analysis uses librosa 0.11.0, soundfile 0.13.1, numba 0.61.2 and
llvmlite 0.44.0. Optional audio reward dependencies installed in `.venv` are
laion-clap 1.1.7, torchlibrosa 0.1.0, braceexpand 0.1.7, webdataset 1.0.2,
wget 3.2, progressbar 2.5 and ftfy (existing). CLAP was installed with
`--no-deps` to preserve the working CUDA/Transformers environment; its optional
text/training pipeline is not used. The tested NumPy is 2.0.2; upstream CLAP's
package metadata requests NumPy <2, so this is a tested audio-only adapter,
not a claim of compatibility with all CLAP utilities.

Pinned checkpoints and hashes live in `models/tunejury_a1/provenance.json`.
The 2.35 GB upstream encoder bundle is converted once to a smaller audio-only
safetensors cache, using restricted tensor/NumPy numeric deserialization.
The audio-only adapter must reproduce upstream's embedding/score exactly.

The previous altered MP3 clips were diagnostic edits, **not generated music**.
Fresh original agent candidates and independent ratings are prepared with
`prepare_reward_candidates.py`, under `outputs/reward_original_candidates_v1`.
Every note is checked against agent actions. No training, seed selection by
reward/listening, reference audio sampling or supplied melody is involved.

The normal `joint_ensemble_rl.py` entry point now refuses policy training while
reward validation is incomplete/failed. `--legacy-proxy-experiment` explicitly
reproduces the historical engineering-proxy experiment only; it does not enable
or imply the new preference reward. A synthetic test pass alone cannot enable RL.
No scalar blend is silently chosen from failed or uncalibrated reward heads.

## Functional role coverage audit (no training)

`audit_functional_roles.py` recovers original source tracks from the cached MIDI
archives, verifies original note content, applies fixed behavioral transfer gates
and performs exclusive role assignment on four-bar windows. Exact labels and
weak role-transfer labels remain separate. Existing data and models are unchanged.

```powershell
.\.venv\Scripts\python.exe -m unittest test_functional_role_audit -v
.\.venv\Scripts\python.exe audit_functional_roles.py
```

See `FUNCTIONAL_ROLE_AUDIT.md` for exact thresholds, provenance, limitations and
corrections to the proposed preference-training pipeline. Results are stored in
`outputs/functional_role_audit_v1`. Coverage counts are pre-projection and do not
establish musical quality or authorize training. The RL validation gate remains.

## Playable-projection survival and inspection (no training)

`audit_projected_roles.py` audits every qualified full trio from the role audit,
reproduces the current canonicalization and production pruning, measures note/
coverage/polyphony retention, and rechecks the frozen role gates. It compares
full-trio eligibility with and without the legacy eight-note rule, then performs
role-aware exact/global-transposition passage deduplication without moving splits.

```powershell
.\.venv\Scripts\python.exe audit_projected_roles.py --render-inspection
.\.venv\Scripts\python.exe -m unittest test_projected_role_audit test_functional_role_audit -v
.\.venv\Scripts\python.exe serve_evaluation.py --directory outputs/projected_role_audit_v1 --port 8788
```

See `PROJECTED_ROLE_AUDIT.md`. Numeric results and the predetermined training-only
raw/projected listening panel are in `outputs/projected_role_audit_v1`. These are
dataset excerpts on the target patches, not generated songs or proof of learned
cooperation. Original corpus/models and reward gates are untouched.

## Supported boundary-sustain validation preflight

`audit_supported_phrase_gate.py` checks a conservative prefix/target schema:
retain legal boundary sustains in a two-bar context with no prefix loss, exclude
source notes longer than the eight-beat action support, and count validation
groups only after projection, gate rechecks and deduplication. It does not add
ties, fake attacks, new compositions, training runs or replace existing loaders.

```powershell
.\.venv\Scripts\python.exe audit_supported_phrase_gate.py
.\.venv\Scripts\python.exe -m unittest test_supported_phrase_gate test_projected_role_audit -v
```

See `SUPPORTED_PHRASE_GATE.md` and `outputs/supported_phrase_gate_v1/REPORT.md`.
The current offline trainer requires ten validation compositions per role; the
repair trainer requires twelve. The audit reports both and keeps training
disabled until coverage and a verified context/masking loader are ready. The
independent PPO reward gate is unchanged.

## Local environment

Tested on Python 3.12.7, PyTorch 2.6.0+cu126, Transformers 4.57.1,
PEFT 0.17.1, and an RTX 4050 Laptop GPU with 6 GB VRAM.
The project `.venv` reuses existing system packages (especially CUDA PyTorch);
additional dependencies were installed into `.venv`. Global packages were not changed.

To recreate with a suitable CUDA PyTorch already installed:

```powershell
git clone https://github.com/SkyTNT/midi-model.git vendor/skytnt-midi-model
git -C vendor/skytnt-midi-model checkout f504d5cb58f769ab0f2909c679238f6621034573
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_assets.py
```

The asset preparation script downloads only config and safetensors weights
(about 468 MB), a SoundFont, and a Windows x64 FluidSynth renderer. Large
downloads are reused on subsequent runs. Asset revisions are recorded in
`assets/sources.json`. MIDI generation is custom code adapted from the upstream
sampling logic; it preserves the upstream source checkout without modifications.

## Sources and experiment limits

- Model: https://huggingface.co/skytnt/midi-model-tv2o-medium
- Code (Apache-2.0): https://github.com/SkyTNT/midi-model
- SoundFont source: https://huggingface.co/skytnt/midi-model
- Renderer: https://github.com/FluidSynth/fluidsynth

This confirms local pretrained inference and playback. It does not establish
musical quality, originality, learned collaboration, or any benefit from RL.
Listen to multiple seeds before choosing this checkpoint as the agent foundation.
