# Collaborative Music Agents

Experimental multi-agent MIDI composition using a frozen SkyTNT Transformer,
instrument-specific LoRA adapters, supervised phrase training, partner-response
training, and gated reinforcement-learning experiments.

## Status

Research code, not a finished music-generation product. Orchestral experiments
use 19 roles across seven families; rock experiments use five roles across four
families. A GRU planner supplies shared context and instrument policies generate
notes for MIDI rendering. The preferred step-224 orchestral checkpoint and all
other weights are external. Rock autonomous generation remains unsatisfactory
in listening tests; happy mood labels are unverified. Lower prediction loss does
not establish musical enjoyment or cooperation. Reward validation must precede
further RL training.

## Setup

```sh
git clone --recurse-submodules https://github.com/YashPHT2/collaborative-music-agents.git
cd collaborative-music-agents
python -m venv .venv
```

Activate the environment and install `requirements.txt`. Use Python 3.11 or 3.12
and a CUDA-compatible PyTorch build for GPU training. Optional requirements are
in `requirements-modal.txt` and `requirements-audio-analysis.txt`.

Run `prepare_assets.py` for pinned model and rendering downloads. Rendering needs
FFmpeg, FluidSynth and a SoundFont; Linux users must supply native binaries.
See [reproducibility notes](docs/REPRODUCIBILITY.md) for external inputs.

## Entry points

| Purpose | Source |
| --- | --- |
| Assets | `prepare_assets.py` |
| Rock metadata | `prepare_rock_candidates.py` |
| Rock audit and preparation | `audit_rock_training_data.py`, `prepare_rock_pilot.py` |
| Rock training | `train_rock_pilot.py`, `modal_rock_pilot.py` |
| History mixing | `modal_rock_history_mix.py` |
| Ensemble training | `train_offline_ensemble.py`, `train_ensemble_repair.py` |
| Earlier A100 orchestra notebook | `notebooks/A100_Orchestra_Training_v3.ipynb` |

The notebook has saved outputs removed. Historical scripts expect local artifacts;
[experiment history](docs/EXPERIMENT_HISTORY.md) documents successive experiments,
not a single turnkey pipeline.

## Checks

```sh
python -m unittest test_rock_pilot test_rock_prompt_diagnostic test_rock_history_mix test_rock_foundation_audit -v
```

Other integration checks may require downloaded assets and local run outputs.

## Contents and licensing

Source, documentation and a sanitized notebook are included. Datasets, weights,
recordings, generated audio, credentials and private ballots are excluded.

SkyTNT is an Apache-2.0 submodule pinned to
`f504d5cb58f769ab0f2909c679238f6621034573`; its license is in the submodule.
No project-wide license has been selected for first-party source.
