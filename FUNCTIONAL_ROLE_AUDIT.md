# Functional role audit v1

Run `audit_functional_roles.py` before changing training data. This is an audit of
the existing 1,806-record corpus, not a new training run. It neither changes
`composition_data.py`/`ensemble_data.py` labels nor replaces their caches.

## Source identity comes first

The old raw-note cache contains program, pitch, timing and velocity, but no source
track/channel identity. Recover these fields from the two original local MIDI
archives. Match every recovered record against the original raw note content;
exclude and report mismatches. Preserve original MIDI SHA256, track name/index,
channel and zero-based GM program. All official/composition-group splits remain.

Use aligned, nonoverlapping four-bar windows at 480 ticks/beat. Include notes
sustaining across the left window boundary. Clip their contribution to window
statistics, without adding a new onset. Do not transpose or fabricate notes.

## Fixed first-pass transfer thresholds

These are conservative weak-label hypotheses, not learned equivalence or rules
for generation. No thresholds are selected to maximize held-out coverage.

All transfers must keep at least 80% of the source window's note events inside
the existing target register. Behavior is measured on the complete source part
before range filtering, to prevent a mostly high melody becoming a fake bass.

| Role | Behavior required for a transfer |
|---|---|
| Organ | At least 50% sounding coverage; mean clipped note duration at least 1 beat; median pitch 48–76; at most 6 onset groups/bar; either at least 15% of sounding time is polyphonic or a regular pulse qualifies. |
| Strings | At least 25% sounding coverage; mean clipped duration at least 0.5 beats; median pitch at least 58; at least 60% of top-onset contour intervals are at most 4 semitones, with at least two measured transitions. An alternative sustained texture qualifies with mean duration at least 2 beats and at least 50% coverage. |
| Cello | Median pitch 36–62; mean clipped duration at least 0.75 beats; at most 10% of sounding time has overlapping notes. No lowest-voice pruning in this audit. |

A regular organ pulse requires at least four onset groups, median inter-onset
interval 0.5–4 beats and inter-onset coefficient of variation at most 0.25.
An onset group counts simultaneous chord notes as one attack time; it is not a
recovered per-voice onset measure. Top-onset contour likewise approximates voice
leading rather than recovering voices.

Exact GM parts need at least one sounding note in the target range; they do not
inherit the transfer behavior restrictions. An exact sustained organ pedal stays
eligible without a minimum note count. Missing parts stay missing.

Candidate programs are frozen in `outputs/functional_role_audit_v1/protocol.json`.
GM programs are zero-based; do not call every program in 21–23 a harmonium or
assume timbre labels establish function. Keyboard transfers receive the same
strict sustain/harmony gates as other candidate programs.

## Exclusive assignment

Choose the maximum-weight combination of organ/strings/cello candidates with
dummy missing roles allowed. Each physical source track may fill at most one
role. Exact edges weigh 2, transfers 1, with a 0.01*coverage tie preference.
These are assignment priorities, not training-loss weights or quality scores.

When either strings or cello is transferred, the strings median must be strictly
above the cello median. This is a data-selection heuristic. Existing exact parts
may cross registers; generated parts are not forced into nonoverlapping bands.

Enumerating strings/cello choices and picking the best remaining organ computes
the optimum including that pair-dependent register constraint. An ordinary
independent edge-matching call alone cannot enforce the coupled constraint.

Physical track exclusivity conservatively prevents a format-0 multitimbral file
from supplying multiple roles. This limitation is reported rather than silently
claiming recovered instrument identities that the file does not provide.

## Outputs and interpretation

- `summary.json` and `REPORT.md`: original exact record counts, accepted exact
  and transferred window counts, and distinct composition-group counts by split.
- `windows.jsonl.gz`: source IDs, statistics, assignments and role provenance
  (`exact`, `transfer`, `missing`) for every audited window.
- `rejections.jsonl.gz`: every eligible-program candidate failing a gate and its
  exact rejection reasons/statistics. Program-ineligible role pairs are counted.
- `source_tracks/`: recovered original source notes, including track identities.
- `protocol.json`: frozen numeric thresholds and candidate program sets.

Group categories may overlap if different windows of a work use different
instrument combinations. Many windows do not become many independent works.
Reported full-trio coverage is pre-projection; playable projection, retained-note
audit, deduplication of transferred passages, and listening inspection remain
necessary before using this as a trainable dataset.

Suggested future training weights are exact=1, transfer=0.35, missing=0, but no
loss weighting or model training is performed by the audit. These weights would
need training-only validation; they are not established optimal values.

## Remaining corrections to the proposed preference pipeline

Conditioned revision DPO requires winner and loser to share the same fixed
external history/plan/partner-draft context. Logging separate contexts for two
independent clips does not make them a conditional preference pair. For complete
multi-segment pieces, later histories depend on earlier generated choices, so a
fixed-context two-bar pilot is simpler to verify than pretending those later
contexts are shared. Full-trajectory preference optimization is a separate route.

The existing `planned` mode still proposes private two-bar segments; it does not
automatically switch to beat-by-beat shared-history generation. That requires an
explicit schedule and matching likelihood evaluator. Decisions files also omit
some stop/rest/token choices, so they alone do not establish exact probability.

Length normalization changes the DPO objective. Do not describe standard IPO as
simply length-normalized DPO. Benchmark any chosen variant against standard DPO,
record the precise formula, and retain density/length diagnostics.

Distance-to-target feature maps assume one ideal target and do not solve general
context-dependent musical preference. Fit centers/scales on training-only data;
validation/test passages must not fit feature transforms. Symbolic features also
cannot be calculated reliably from a mixed MP3 without suitable transcription.

A Davidson tie model can retain ties, but 'neither good' need not imply equal
quality. Nor is a winner among two poor clips automatically a positive anchor.
Any cross-context absolute-quality labels require explicit judgments. Pair splits
must keep all connected compositions/prefixes in one partition, not just match
numeric random seeds. No DPO or PPO training is enabled by this audit.
