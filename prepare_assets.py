"""Download the pinned pretrained checkpoint and a local MIDI renderer."""
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import requests
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
MODEL_ID = "skytnt/midi-model-tv2o-medium"
MODEL_REVISION = "0f8f265d4330f4e46527ac2313200254c5757f5f"
SOUNDFONT_ID = "skytnt/midi-model"


def main():
    model_dir = ROOT / "models" / "skytnt-tv2o-medium"
    for filename in ("config.json", "model.safetensors"):
        print(f"Downloading {MODEL_ID}/{filename}", flush=True)
        hf_hub_download(MODEL_ID, filename, revision=MODEL_REVISION, local_dir=model_dir)
    # Pin the SoundFont independently: it lives in the older model repository.
    from huggingface_hub import HfApi
    manifest_path = ROOT / "assets" / "sources.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    sf_revision = manifest.get("soundfont_revision") or HfApi().model_info(SOUNDFONT_ID).sha
    print("Downloading the piano SoundFont", flush=True)
    hf_hub_download(SOUNDFONT_ID, "soundfont.sf2", revision=sf_revision, local_dir=ROOT / "assets")
    renderer = ROOT / "tools" / "fluidsynth"
    if not list(renderer.rglob("fluidsynth.exe")):
        url = "https://github.com/FluidSynth/fluidsynth/releases/download/v2.6.1/fluidsynth-v2.6.1-win10-x64-cpp11.zip"
        print("Downloading FluidSynth v2.6.1 (Windows x64)", flush=True)
        archive = ROOT / "tools" / "fluidsynth.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with archive.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        renderer.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                if not (renderer / member).resolve().is_relative_to(renderer.resolve()):
                    raise ValueError("Unsafe archive path")
            bundle.extractall(renderer)
    manifest.update(model_id=MODEL_ID, model_revision=MODEL_REVISION,
                    soundfont_id=SOUNDFONT_ID, soundfont_revision=sf_revision,
                    fluidsynth_version="2.6.1",
                    source_revision="f504d5cb58f769ab0f2909c679238f6621034573")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Assets ready.", flush=True)


if __name__ == "__main__":
    main()
