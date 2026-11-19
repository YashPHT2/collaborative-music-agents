# Projection-survival experiment

`audit_projected_roles.py` implements the second data audit without changing any
existing training cache, source labels, sampler limits, model weights or reward
acceptance file. Its input is all 443 full-trio windows from the first audit:
353 transferred and 90 exact. Four bars at 100 BPM plus release produce 12-second
inspection files. These excerpts are source training data, not newly composed
music and not evidence that our agents improved.

## Reproducing

```powershell
.\.venv\Scripts\python.exe -m unittest test_projected_role_audit test_functional_role_audit -v
.\.venv\Scripts\python.exe audit_projected_roles.py
.\.venv\Scripts\python.exe audit_projected_roles.py --render-inspection
.\.venv\Scripts\python.exe serve_evaluation.py --directory outputs/projected_role_audit_v1 --port 8788
```

The protocol and numeric output are under `outputs/projected_role_audit_v1`.
`windows.json` contains every source assignment, raw/projected statistics,
retention ratios, source-note IDs, projected notes and gate outcomes.

## What is reproduced

Reconstruct the selected source track/channel/program for every role, preserving
raw incoming sustains for before-projection statistics. Canonicalize source notes
using the existing range, velocity, duplicate and same-pitch-overlap rules.
Onsets round to 30 ticks; durations use 60-tick steps up to **3840 ticks/eight
beats**. Clip to the four-bar interval and apply the existing onset-only window
rule. A note already sounding before the interval is therefore omitted from the
projected target; its contribution remains present in raw coverage statistics.

Instrumented pruning must exactly reproduce
`supervised_ensemble.playable_projection` inside `offline_ensemble.limits`, and
the result must pass every beat's `note_trio.commit` without changing notes.
This includes strings cap 4/polyphony 4, organ cap 4/polyphony 4 and cello cap
2/polyphony 1. Cello keeps the lower simultaneous voice, and later notes can
shorten earlier cello durations. No alternative projection is silently substituted.

Incoming sustains omitted by cropping, source range exclusions, duration/onset/
velocity changes, duplicate cleanup, event-cap losses and voice pruning are logged
separately. Source events that never formed a complete note are recounted from
the original MIDI and reported separately. **They were already omitted from the
old raw cache; they did not artificially inflate the previous coverage audit.**
No missing note-off is guessed and no unfinished note is completed into a target.

Mean durations and sounding coverage are clipped to the inspected interval.
Onset counts exclude incoming source sustains. Ratios may exceed one slightly
because quantization extends coverage. A null polyphonic-time ratio means no
polyphonic source time existed; it is not a failed-retention value.

## Eligibility and deduplication

Recheck exactly the prior role-transfer gates with the original source GM patch.
Exact source parts retain their less restrictive exact eligibility. Recheck the
joint register condition after projection. Compare the resulting complete trios
with and without at least eight projected note events in every role. That rule
is evaluated, not adopted or imposed on generation.

Deduplicate both candidate pools separately. Exact fingerprints retain role,
aligned onset, duration, pitch and velocity. Structural fingerprints preserve
role, timing and relative pitches while ignoring velocity and normalizing one
global transposition. This conservatively recognizes the same passage with
different dynamics. It never normalizes each instrument independently or removes
leading silence. Test takes priority over validation, then training. Any detected
cross-split overlap excludes the entire lower-priority composition group from
that candidate pool. No group is moved between partitions.

The ten-validation-compositions-per-role gate is reported for this **trio-only
pool**. It must not be confused with coverage of the unchanged broader dataset,
which includes instrument pairs. Adding or removing windows within the same
seven works cannot turn them into ten independent validation compositions.

## Listening inspection

One deterministic training-only example is selected from each fixed bucket:
all exact, high-retention transfer, and transfer with at least one role losing
15–35% of source notes. Other roles can lose more and all retention values are
visible. Empty buckets remain unavailable rather than being replaced by a
good-sounding choice. Examples are not selected by reward or listening.

Render raw recovered and projected notes with the same target organ/strings/
cello programs, role levels, SoundFont, tempo and release treatment. Raw incoming
sustains are explicitly retriggered at the crop boundary for audible inspection;
this is not a newly invented training target or an agent action. Raw source
events retain integer-tick precision, while projected notes use the actual
training grid. Repeated-pitch overlaps use separate render channels with the
same instrument patch. A MIDI round-trip maps these channels back to logical
roles and checks exact note identity.

Mixes and solos are matched per raw/projected pair to a common integrated LUFS
target with **constant gain**, limited by true-peak headroom. No dynamic loudness
normalization is applied. Silent stems are labelled and excluded from gain
matching. Final audio must be finite, unclipped and 12 seconds. The page pauses
other players on play, offers a mix/solo selector and downloads two simple
inspection judgments per passage. Labels remain inspection feedback, not DPO
preference pairs or automatically accepted positive training examples.

## Before training

Inspect whether an unsuitable raw arrangement slipped through the transfer rules
or the existing projection damaged a plausible source arrangement. Any proposed
repair must be developed on training data under a new version, with held-out
validation preserved. Revisit incoming-sustain context and the eight-note data
filter without adding notes or forcing constant participation. Expand independent
validation coverage before a non-pilot trio-only campaign. PPO and DPO remain off.
