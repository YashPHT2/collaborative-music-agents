# Supported boundary-sustain preflight

`audit_supported_phrase_gate.py` audits the existing 443 full-trio candidate
windows under a versioned, conservative support contract. It does not add works,
change splits, retrain a model or replace a trainer's input cache.

```powershell
.\.venv\Scripts\python.exe audit_supported_phrase_gate.py
.\.venv\Scripts\python.exe -m unittest test_supported_phrase_gate test_projected_role_audit -v
```

Artifacts are in `outputs/supported_phrase_gate_v1`.

## Contract

- Existing role ranges, event caps, polyphony and timing/velocity grids remain.
- One source note must be representable by one existing duration decision.
  An in-range note longer than 3840 ticks/eight beats that overlaps the selected
  context or target makes that trio window ineligible. Its original duration is
  logged. No capped pseudo-target, tied token or fake repeated attack is created.
- Canonicalize and apply the current production playable projection to each full
  source part before cropping. This prevents a crop from forgetting the earlier
  notes that affected active-note and polyphony decisions.
- Preserve a two-bar/eight-beat context prefix before each four-bar target. For
  supported notes, every note still sounding at target start must have started
  within that prefix. Keep the true quantized onset there, with positive container
  time. Never shift it to the target boundary.
- Prefix loss is zero. Supervise only the sixteen target beats, respecting the
  per-role presence mask. A sparse target is not removed for having fewer than
  eight notes. A sustaining-only role has no new attack target at the boundary.
- Distinguish `missing`, `present_resting`, `present_sustaining`, `present_sparse`
  and `present_active`. A genuine rest needs nearby source-role evidence and no
  raw target activity that disappeared through projection. Projection damage is
  not relabelled as an intentional rest. The current full-trio candidate pool does
  not establish coverage of all missing/resting cases; those need broader loader
  examples before training integration.
- Recheck unchanged transfer behavior gates using sounding notes, including
  incoming supported notes. Check the full context/target ledger with production
  simultaneous commits. Note durations remain within eight beats.
- Deduplicate role-aware exact/global-transposition fingerprints on targets with
  incoming notes at their true relative onsets. Exclude whole lower-priority
  composition groups on detected cross-split overlap. No groups move between
  training, validation and test.

The audit retains intermediate capped source projections for diagnostics only;
windows with unsupported original durations have `notes=null` and are never
included in the eligible example pool. Checking original durations before
acceptance prevents a silent duration cap from hiding a support mismatch.

## Gate

Count distinct validation composition groups **after** support exclusions,
projection, behavioral qualification and deduplication. Do not count windows,
repeated motifs or different random seeds as independent compositions.

The actual current entry points differ:

| Trainer | Required validation compositions per role |
|---|---:|
| `train_offline_ensemble.py` | 10 |
| `train_ensemble_repair.py` | 12 |

`require_supervised_gate` fails when either the requested count requirement is
not met or the new prefix/masking schema lacks a verified trainer integration.
It is a reusable preflight for that future integration; it is not claimed to have
patched the legacy trainers. A count-only pass must not silently train with the
old onset-only window loader.

PPO's separate reward-validation gate remains untouched. Passing a data count
would not validate a taste reward or establish musical quality.

## Current result and next work

Read `REPORT.md` and `summary.json` for exact counts. Track01757 remains excluded
because its 22-beat source notes are outside the current representation. Restored
windows from supported boundary handling do not necessarily add any new distinct
composition group. The stricter long-note support check can exclude additional
groups that previously appeared eligible after silent truncation.

If the gate fails, add independently eligible works from the original pinned
sources under a separately versioned corpus expansion. Preserve licence/source
provenance, group assignments and held-out splits, then deduplicate the new pool
against existing works. Do not lower behavioral thresholds, count extra windows
as new works, move test works into validation, or turn unrepresentable notes into
repeated attacks to reach a quota.

After adequate coverage exists, integrate and test causal teacher examples,
balanced category losses, source/transfer weights and draft conditioning against
this support contract. Only then start the bounded supervised experiment.
