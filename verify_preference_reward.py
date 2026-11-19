"""Integration checks of the audio-only adapter against upstream CLAP's method."""
import ast
import importlib.util
import json
from pathlib import Path
import types
import numpy as np
import torch
import torch.nn.functional as F
from preference_reward import PreferenceReward, MODELS, SR, canonical_waveform


def main():
    model = PreferenceReward()
    package = Path(importlib.util.find_spec('laion_clap').origin).parent
    tree = ast.parse((package / 'clap_module/model.py').read_text())
    # Reuse the exact inspected upstream method without constructing/downloading
    # an irrelevant text encoder. No change to the audio network or weights.
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CLAP')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_audio_embedding')
    module = ast.Module(body=[method], type_ignores=[])
    scope = dict(torch=torch, F=F)
    exec(compile(ast.fix_missing_locations(module), str(package / 'clap_module/model.py'), 'exec'), scope)
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.branch = model.audio_branch
            self.audio_projection = model.projection
        def encode_audio(self, x, device):
            return self.branch(x, device=device)
    adapter = Adapter()
    t = np.arange(SR * 10) / SR
    y = (.05 * np.sin(2*np.pi*220*t) * (.6 + .4*np.sin(2*np.pi*.5*t))).astype(np.float32)
    wave, _, _ = canonical_waveform(y, SR)
    wave = ((np.clip(wave, -1, 1)*32767).astype(np.int16).astype(np.float32)/32767)
    with torch.inference_mode():
        expected = scope['get_audio_embedding'](adapter, [{'waveform': torch.from_numpy(wave)}])
        actual = model.evaluate(y, SR)
        second = model.evaluate(y, SR, {'synthetic_profile': actual['embedding']})
        expected_score = float(model.head(expected).item())
    error = abs(expected_score - actual['preference'])
    assert error < 1e-6, error
    assert abs(actual['preference'] - second['preference']) < 1e-6
    assert second['style_similarity']['synthetic_profile'] > .99999
    assert all(not p.requires_grad for m in (model.audio_branch, model.projection, model.head) for p in m.parameters())
    result = dict(upstream_audio_embedding_score_absolute_error=error,
                  deterministic_score=True, style_does_not_change_preference_score=True,
                  all_parameters_frozen=True, device=model.device,
                  gpu_peak_memory_gb=torch.cuda.max_memory_allocated()/1e9 if model.device.startswith('cuda') else None)
    (MODELS / 'integration_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
