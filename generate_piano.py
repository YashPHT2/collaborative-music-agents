"""Generate original piano MIDI from SkyTNT's pretrained weights; no training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "skytnt-midi-model"))

import MIDI
import mido
import numpy as np
import torch
from transformers import DynamicCache
from midi_model import MIDIModel


@torch.inference_mode()
def compose(model, seed, bars, bpm, temperature, max_events):
    """Upstream event sampling, restricted to acoustic piano channel 0.

    Only note events and EOS are sampled after the conditioning header.
    Pitches, polyphony, durations, velocities and note times remain model choices.
    """
    tokenizer = model.tokenizer
    prompt = [
        [tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1),
        tokenizer.event2tokens(["time_signature", 0, 0, 0, 3, 1]),
        tokenizer.event2tokens(["key_signature", 0, 0, 0, 7, 0]),
        tokenizer.event2tokens(["set_tempo", 0, 0, 0, bpm]),
        tokenizer.event2tokens(["patch_change", 0, 0, 1, 0, 0]),
    ]
    sequence = torch.tensor([prompt], dtype=torch.long, device=model.device)
    generator = torch.Generator(device=model.device).manual_seed(seed)
    event_cache = DynamicCache()
    past_length = 0
    absolute_beats = 0
    notes = 0
    stop_reason = "event_limit"
    for step in range(max_events):
        hidden = model(sequence[:, past_length:], cache=event_cache)[:, -1]
        token_cache = DynamicCache()
        row = None
        event_name = None
        for index in range(tokenizer.max_token_seq):
            if index == 0:
                allowed = [tokenizer.event_ids["note"], tokenizer.eos_id]
                token_input = None
            else:
                parameter = tokenizer.events[event_name][index - 1]
                allowed = tokenizer.parameter_ids[parameter]
                if parameter == "channel":
                    allowed = [allowed[0]]
                elif parameter == "track":
                    allowed = [allowed[1]]
                elif parameter in ("duration", "velocity"):
                    allowed = allowed[1:]
                token_input = row[:, -1:]
            logits = model.forward_token(hidden if index == 0 else None,
                                         token_input, cache=token_cache)[:, -1]
            mask = torch.full_like(logits, float("-inf"))
            mask[:, allowed] = 0
            probabilities = torch.softmax((logits + mask) / temperature, dim=-1)
            sampled = model.sample_top_p_k(probabilities, 0.98, 20, generator=generator).reshape(1, 1)
            row = sampled if row is None else torch.cat([row, sampled], dim=1)
            if index == 0:
                if sampled.item() == tokenizer.eos_id:
                    stop_reason = "eos"
                    break
                event_name = tokenizer.id_events[sampled.item()]
            elif index == len(tokenizer.events[event_name]):
                break
        if stop_reason == "eos":
            break
        row = torch.nn.functional.pad(row, (0, tokenizer.max_token_seq - row.shape[1]),
                                      value=tokenizer.pad_id)
        event = tokenizer.tokens2event(row[0].tolist())
        absolute_beats += event[1]
        if absolute_beats + event[2] / 16 >= bars * 4:
            stop_reason = "bar_limit"
            break
        past_length = sequence.shape[1]
        sequence = torch.cat([sequence, row.unsqueeze(1)], dim=1)
        notes += 1
        if notes % 32 == 0:
            print(f"Generated {notes} note events; beat {absolute_beats + event[2] / 16:.2f}", flush=True)
    score = tokenizer.detokenize(sequence[0].cpu().numpy())
    end_tick = bars * 4 * score[0]
    for track in score[1:]:
        for event in track:
            if event[0] == "note":
                event[2] = min(event[2], end_tick - event[1])
    return score, stop_reason


def render(midi_path, wav_path):
    executables = list((ROOT / "tools" / "fluidsynth").rglob("fluidsynth.exe"))
    executable = os.environ.get('FLUIDSYNTH_BIN') or (str(executables[0]) if os.name=='nt' and executables else shutil.which('fluidsynth'))
    if not executable:
        raise FileNotFoundError("Install fluidsynth or set FLUIDSYNTH_BIN.")
    subprocess.run([executable, "-ni", "-F", str(wav_path), "-r", "44100",
                    "-g", "0.6", str(ROOT / "assets" / "soundfont.sf2"), str(midi_path)],
                   check=True, capture_output=True, text=True)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
                        "-codec:a", "libmp3lame", "-q:a", "2", str(wav_path.with_suffix(".mp3"))],
                       check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--bpm", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-events", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.bars <= 32 or not 30 <= args.bpm <= 240 or args.temperature <= 0 or args.max_events < 1:
        parser.error("Use bars 1..32, BPM 30..240, and positive temperature/event limit.")
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    print(f"Loading pretrained SkyTNT on {device} ({dtype})", flush=True)
    model = MIDIModel.from_pretrained(ROOT / "models" / "skytnt-tv2o-medium",
                                      dtype=dtype, local_files_only=True).to(device).eval()
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    started = time.perf_counter()
    score, stop_reason = compose(model, args.seed, args.bars, args.bpm, args.temperature, args.max_events)
    elapsed = time.perf_counter() - started
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    stem = f"piano_seed{args.seed}"
    midi_path = output_dir / f"{stem}.mid"
    midi_path.write_bytes(MIDI.score2midi(score))
    notes = [event for track in score[1:] for event in track if event[0] == "note"]
    if not notes:
        raise RuntimeError("Checkpoint produced no notes; try another seed.")
    midi = mido.MidiFile(midi_path)
    metadata = dict(model="skytnt/midi-model-tv2o-medium",
                    revision="0f8f265d4330f4e46527ac2313200254c5757f5f",
                    seed=args.seed, requested_bars=args.bars, bpm=args.bpm,
                    key_condition="C major", temperature=args.temperature, device=device,
                    dtype=str(dtype), note_count=len(notes), duration_seconds=round(midi.length, 3),
                    pitch_range=[int(min(n[4] for n in notes)), int(max(n[4] for n in notes))],
                    generation_seconds=round(elapsed, 3), stop_reason=stop_reason,
                    peak_vram_mb=round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" else None,
                    trained=False, supplied_melody=False)
    (output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not args.no_audio:
        print("Rendering piano audio...", flush=True)
        render(midi_path, output_dir / f"{stem}.wav")
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Saved {midi_path}", flush=True)


if __name__ == "__main__":
    main()
