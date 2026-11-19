"""Fetch pinned, inspected TuneJury A1 resources; does not start music training."""
import hashlib
import json
import os
from pathlib import Path

os.environ['HF_HUB_DISABLE_XET'] = '1'
import requests
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent / 'models' / 'tunejury_a1'
CODE_REVISION = '9dd8ea3e4a72f358ce311a6a856194ca6932c89e'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = dict(code_revision=CODE_REVISION, models={}, license='A1: Apache-2.0; see upstream checkpoints/README.md')
    for source, destination in [('tunejury/model.py', 'head_model.py'),
                                ('checkpoints/input_ablation/A1_clap_audio_only.json', 'head_config.json'),
                                ('LICENSE', 'LICENSE'), ('checkpoints/README.md', 'UPSTREAM_CHECKPOINTS.md')]:
        r = requests.get(f'https://raw.githubusercontent.com/yonghyunk1m/TuneJury/{CODE_REVISION}/{source}', timeout=60)
        r.raise_for_status()
        (ROOT / destination).write_bytes(r.content)
    for repo, filename, name in [('TuneJury/tunejury', 'A1_clap_audio_only.pt', 'head'),
                                  ('lukewys/laion_clap', 'music_audioset_epoch_15_esc_90.14.pt', 'encoder')]:
        revision = HfApi().model_info(repo).sha
        print(f'Fetching {name}: {repo}@{revision}', flush=True)
        path = ROOT / filename
        if not path.exists():
            # requests avoids this machine's malformed global HF cache path.
            response = requests.get(f'https://huggingface.co/{repo}/resolve/{revision}/{filename}', stream=True, timeout=90)
            response.raise_for_status()
            temporary = path.with_suffix('.download')
            written = 0
            with temporary.open('wb') as f:
                for block in response.iter_content(4 * 1024 * 1024):
                    f.write(block)
                    written += len(block)
                    if written % (128 * 1024 * 1024) == 0:
                        print(f'{name}: {written / 1e6:.0f} MB', flush=True)
            expected = int(response.headers.get('content-length', written))
            if written != expected:
                raise ValueError('Incomplete checkpoint download')
            temporary.replace(path)
        h = hashlib.sha256()
        with path.open('rb') as f:
            while block := f.read(8 * 1024 * 1024):
                h.update(block)
        manifest['models'][name] = dict(repo=repo, revision=revision, file=filename, sha256=h.hexdigest())
        (ROOT / 'provenance.json').write_text(json.dumps(manifest, indent=2))
        print(f'Ready {name}: {path.stat().st_size / 1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
