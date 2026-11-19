# Reproducibility

This source-only repository excludes weights, data archives, render assets,
clips, ballots and local run manifests. Exact reproduction needs the original
manifests, composition splits, checkpoint hashes and environment. Different
hardware or regenerated data may change results.

Initialize the pinned dependency with `git submodule update --init --recursive`.
Install requirements and run `prepare_assets.py` as needed. Check platform-specific
rendering binaries before generating audio.

## Rock data

```sh
python prepare_rock_candidates.py --download
python audit_rock_training_data.py
python prepare_rock_pilot.py
```

Metadata discovery does not verify mood or training eligibility. Inspect the
verified role/window and composition-separated data gates. Composition coverage
and deduplication remain limitations. The rock pipeline expects an external
orchestral initialization at `outputs/preferred_model_v1/step224.pt`; inspect
script arguments and do not substitute unrelated checkpoints or invent manifests.

## Remote training

Install `requirements-modal.txt`, authenticate to your own account, and inspect
the resource caps, A100 configuration, volume paths and checkpoint inputs before
running these paid-compute entry points:

```sh
modal token new
modal run --detach modal_rock_pilot.py
modal run --detach modal_rock_history_mix.py
```

Preserve existing runs and checkpoints; do not assume automatic resume support.
The notebook is an earlier orchestral package and expects its external package
and assets; it is not the latest rock workflow.

## Evaluation and listening

Historical listening scripts require saved protocols, episodes and run outputs.
Some one-off manifests are external. Audio-analysis helpers have machine-local
reference paths: provide your own recordings and paths. Recordings are excluded.

Preserve composition-separated validation, checkpoint hashes and feedback. Use
matched seeds and identical rendering for comparisons. Report event loss,
note-attribute loss, partner probes and listener preference separately. Failed
validation/reward gates do not establish musical improvement. A local HTTP server
can serve existing listening directories; source alone does not recreate clips.
