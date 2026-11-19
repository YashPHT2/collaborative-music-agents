"""Frozen human-preference evaluator. Style similarity is a separate diagnostic.

TuneJury A1 uses the exact HTSAT-base audio branch and projection from CLAP,
without constructing the unused text encoder/tokenizers. No remote code runs.
The preference head is upstream's inspected four-layer MLP. Weights-only loads.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import librosa

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models' / 'tunejury_a1'
SR = 48000
SAMPLES = 10 * SR


def window_starts(length, window=SAMPLES):
    """Deterministic full coverage, overlapping only the last window if needed."""
    if length <= 0:
        raise ValueError('Empty audio')
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, window))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def canonical_waveform(audio, sr):
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)  # soundfile convention: samples, channels
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('Expected finite, nonempty mono or samples/channels audio')
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    silent = rms < 1e-7
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    # Keep relative dynamics; prevent mastering gain from becoming taste.
    if not silent:
        y = y * min(.1 / rms, .98 / max(float(np.max(np.abs(y))), 1e-8))
    return y, rms, silent


class PreferenceReward:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(4)
        # Import only the installed package's audio architecture, avoiding hook.py
        # which downloads three unused tokenizers and a text model at import.
        package = Path(importlib.util.find_spec('laion_clap').origin).parent
        if str(package) not in sys.path:
            sys.path.insert(0, str(package))
        from clap_module.htsat import create_htsat_model
        cfg = json.loads((package / 'clap_module/model_configs/HTSAT-base.json').read_text())
        self.audio_branch = create_htsat_model(SimpleNamespace(**cfg['audio_cfg']), False)
        self.projection = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 512))
        provenance = json.loads((MODELS / 'provenance.json').read_text())
        from safetensors.torch import load_file, save_file
        audio_only = MODELS / 'audio_only.safetensors'
        if audio_only.exists():
            state = load_file(str(audio_only))
        else:
            # The legacy upstream bundle also contains optimizer statistics with
            # NumPy scalar metadata. Allow only these specific numeric types;
            # never switch to unrestricted pickle loading.
            numeric = [(np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
                       np.dtype, type(np.dtype('float64')), type(np.dtype('float32'))]
            with torch.serialization.safe_globals(numeric):
                state = torch.load(MODELS / provenance['models']['encoder']['file'], map_location='cpu', weights_only=True)
            state = state.get('state_dict', state)
            state = {k.removeprefix('module.'): v for k, v in state.items()}
            state = {k: v.contiguous() for k, v in state.items() if k.startswith(('audio_branch.', 'audio_projection.'))}
            save_file(state, str(audio_only), metadata={'source_sha256': provenance['models']['encoder']['sha256']})
        for prefix, model in [('audio_branch.', self.audio_branch), ('audio_projection.', self.projection)]:
            model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        del state
        spec = importlib.util.spec_from_file_location('tunejury_a1_head', MODELS / 'head_model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.head = module.TuneJury(input_dim=512)
        state = torch.load(MODELS / provenance['models']['head']['file'], map_location='cpu', weights_only=True)
        self.head.load_state_dict(state.get('state_dict', state), strict=True)
        for model in (self.audio_branch, self.projection, self.head):
            model.eval().requires_grad_(False).to(self.device)
        self.provenance = provenance

    @torch.inference_mode()
    def evaluate(self, audio, sr, style_profiles=None):
        y, rms, silent = canonical_waveform(audio, sr)
        scores, embeddings = [], []
        for start in window_starts(len(y)):
            clip = y[start:start + SAMPLES]
            # Exactly upstream repeatpad filling for clips shorter than 10 s.
            if len(clip) < SAMPLES:
                clip = np.tile(clip, SAMPLES // len(clip))
                clip = np.pad(clip, (0, SAMPLES - len(clip)))
            clip = ((np.clip(clip, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32767)
            wave = torch.from_numpy(clip)[None].to(self.device)
            features = self.audio_branch({'waveform': wave}, device=self.device)['embedding']
            embedding = F.normalize(self.projection(features), dim=-1)
            scores.append(float(self.head(embedding).item()))
            embeddings.append(embedding[0].cpu().numpy())
        mean = np.mean(embeddings, axis=0)
        mean /= max(float(np.linalg.norm(mean)), 1e-9)
        style = {name: float(np.dot(mean, np.asarray(profile))) for name, profile in (style_profiles or {}).items()}
        return dict(preference=float(np.mean(scores)), window_preference=scores,
                    window_starts_seconds=[s / SR for s in window_starts(len(y))],
                    worst_window_preference=float(min(scores)), style_similarity=style,
                    embedding=mean.tolist(), source_rms=rms, silence=silent,
                    valid_for_optimization=not silent,
                    interpretation='Uncalibrated human-preference logit; style cosine is not a quality reward.')


def require_validated_reward(directory):
    """Fail closed. A stress-test pass alone cannot authorize taste optimization."""
    path = Path(directory) / 'acceptance.json'
    if not path.exists():
        raise RuntimeError('Reward validation has not completed; no acceptance record exists.')
    status = json.loads(path.read_text())
    required = ('automated_gate_passed', 'human_preference_gate_passed', 'generated_domain_gate_passed', 'rl_training_allowed')
    if not all(status.get(key) is True for key in required):
        raise RuntimeError('Reward is not validated for RL: structural, generated-domain and human-preference gates must all pass.')
    return status
