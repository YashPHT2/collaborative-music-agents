"""Causal piano accompaniment baseline using frozen SkyTNT weights."""
import argparse
import copy
import json
from pathlib import Path
import time

from generate_piano import ROOT, MIDI, MIDIModel, DynamicCache, np, torch, render

TPB = 480
BAR = 4 * TPB
CHANGE_TICK = 4 * BAR
END_TICK = 8 * BAR


def melody(changed=False):
    """Scripted test material, not an AI-composed or supplied commercial song."""
    phrase = [72, 76, 79, 76, 69, 72, 77, 76,
              71, 74, 79, 77, 76, 74, 72, 72]
    return [["note", i * TPB, 420, 0,
             pitch + (2 if changed and i >= 16 else 0), 90]
            for i, pitch in enumerate(phrase * 2)]


def note_signature(notes):
    return sorted(tuple(int(x) for x in n[1:]) for n in notes)


def assemble(lead, accompaniment, bpm):
    return [TPB,
            [["track_name", 0, b"Conductor"], ["set_tempo", 0, int(60_000_000 / bpm)],
             ["time_signature", 0, 4, 2, 24, 8], ["key_signature", 0, 0, 0]],
            [["track_name", 0, b"Fixed melody"], ["patch_change", 0, 0, 0], *copy.deepcopy(lead)],
            [["track_name", 0, b"Generated accompaniment"], ["patch_change", 0, 1, 0],
             *copy.deepcopy(accompaniment)]]


class EventSampler:
    """Only committed past/current events enter the model's context.

    The scheduler knows when the next input arrives, but never gives its pitch
    to the policy early. A sampled event at/beyond that boundary is discarded.
    EOS is interpreted as waiting for the next melody input for this test.
    """
    def __init__(self, model, seed, bpm, temperature):
        self.model = model
        self.tok = model.tokenizer
        self.generator = torch.Generator(device=model.device).manual_seed(seed)
        self.temperature = temperature
        self.rows = [[self.tok.bos_id] + [self.tok.pad_id] * (self.tok.max_token_seq - 1)]
        for event in (["time_signature", 0, 0, 0, 3, 1],
                      ["key_signature", 0, 0, 0, 7, 0],
                      ["set_tempo", 0, 0, 0, bpm],
                      ["patch_change", 0, 0, 1, 0, 0],
                      ["patch_change", 0, 0, 2, 1, 0]):
            self.rows.append(self.tok.event2tokens(event))
        self.cache = DynamicCache()
        self.cached_length = 0
        self.last_tick = 0

    def append_note(self, note, track):
        _, tick, duration, channel, pitch, velocity = note
        if tick < self.last_tick or tick % 30 or duration % 30:
            raise ValueError("Input notes must be chronological and on the 1/16-beat grid.")
        row = self.tok.event2tokens(["note", tick // TPB - self.last_tick // TPB,
                                    (tick % TPB) // 30, track, channel, pitch,
                                    velocity, duration // 30])
        if not row:
            raise ValueError("Note is not representable by the checkpoint tokenizer.")
        self.rows.append(row)
        self.last_tick = tick

    @torch.inference_mode()
    def sample(self):
        x = torch.tensor([self.rows[self.cached_length:]], dtype=torch.long, device=self.model.device)
        hidden = self.model(x, cache=self.cache)[:, -1]
        self.cached_length = len(self.rows)
        row = None
        token_cache = DynamicCache()
        parameters = self.tok.events["note"]
        for index in range(len(parameters) + 1):
            if index == 0:
                allowed = [self.tok.event_ids["note"], self.tok.eos_id]
            else:
                parameter = parameters[index - 1]
                allowed = self.tok.parameter_ids[parameter]
                if parameter == "track":
                    allowed = [allowed[2]]
                elif parameter == "channel":
                    allowed = [allowed[1]]
                elif parameter == "pitch":
                    allowed = allowed[36:67]  # Accompaniment register C2..F#4.
                elif parameter == "velocity":
                    allowed = allowed[32:96]
                elif parameter == "duration":
                    allowed = allowed[1:65]  # At most one bar; note crossing input boundaries is allowed.
                elif parameter == "time2":
                    time1_index = 1 + parameters.index("time1")
                    delta = row[0, time1_index].item() - self.tok.parameter_ids["time1"][0]
                    if delta == 0:
                        allowed = allowed[(self.last_tick % TPB) // 30:]
            logits = self.model.forward_token(hidden if index == 0 else None,
                                               None if row is None else row[:, -1:],
                                               cache=token_cache)[:, -1]
            mask = torch.full_like(logits, float("-inf"))
            mask[:, allowed] = 0
            probs = torch.softmax((logits + mask) / self.temperature, dim=-1)
            token = self.choose_token(index, parameters, logits, probs, allowed, row)
            row = token if row is None else torch.cat([row, token], dim=1)
            if index == 0 and token.item() == self.tok.eos_id:
                return None
        event = self.tok.tokens2event(row[0].tolist())
        _, delta, substep, _, channel, pitch, velocity, duration = event
        tick = (self.last_tick // TPB + delta) * TPB + substep * 30
        return ["note", tick, duration * 30, channel, pitch, velocity]

    def choose_token(self, index, parameters, logits, probs, allowed, row):
        """Override for experimental policies; default preserves the frozen baseline."""
        return self.model.sample_top_p_k(probs, 0.98, 20, generator=self.generator).reshape(1, 1)


def accompany(sampler, lead, end_tick=END_TICK, per_input_limit=16, verbose=True):
    accompaniment = []
    stats = dict(boundary_waits=0, eos_waits=0, density_cap_waits=0)
    for index, note in enumerate(lead):
        sampler.append_note(note, track=1)
        boundary = lead[index + 1][1] if index + 1 < len(lead) else end_tick
        for _ in range(per_input_limit):
            candidate = sampler.sample()
            if candidate is None:
                stats["eos_waits"] += 1
                break
            if candidate[1] >= boundary:
                stats["boundary_waits"] += 1
                break
            candidate[2] = min(candidate[2], end_tick - candidate[1])
            sampler.append_note(candidate, track=2)
            accompaniment.append(candidate)
        else:
            stats["density_cap_waits"] += 1
        if verbose and (index + 1) % 4 == 0:
            print(f"Bar {(index + 1) // 4}/8: {len(accompaniment)} accompaniment notes", flush=True)
    return accompaniment, stats


def resolve_overlaps(notes):
    """End repeated same-pitch notes before the next onset on that channel."""
    result = copy.deepcopy(notes)
    next_onset = {}
    for note in reversed(sorted(result, key=lambda n: n[1])):
        key = (note[3], note[4])
        if key in next_onset:
            note[2] = min(note[2], max(0, next_onset[key] - note[1]))
        next_onset[key] = note[1]
    return [note for note in result if note[2] > 0]


def summarize(notes):
    tail = [n for n in notes if n[1] >= CHANGE_TICK]
    d_major = {1, 2, 4, 6, 7, 9, 11}
    return dict(note_count=len(notes), second_half_notes=len(tail),
                second_half_d_major_fraction=(round(sum(n[4] % 12 in d_major for n in tail) / len(tail), 3)
                                               if tail else None))


def write_result(directory, name, score, audio):
    midi_path = directory / f"{name}.mid"
    midi_path.write_bytes(MIDI.score2midi(score))
    if audio:
        render(midi_path, directory / f"{name}.wav")
    return midi_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bpm", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if not 30 <= args.bpm <= 240 or args.temperature <= 0:
        parser.error("Use BPM 30..240 and a positive temperature.")
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    print(f"Loading frozen SkyTNT on {device}", flush=True)
    model = MIDIModel.from_pretrained(ROOT / "models" / "skytnt-tv2o-medium",
                                      dtype=dtype, local_files_only=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    directory = ROOT / "outputs" / "accompaniment" / f"seed{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    parts = {}
    summary = dict(seed=args.seed, bpm=args.bpm, temperature=args.temperature,
                   model="skytnt/midi-model-tv2o-medium",
                   revision="0f8f265d4330f4e46527ac2313200254c5757f5f",
                   device=device, dtype=str(dtype),
                   melody_source="scripted original test phrase",
                   change="transpose melody +2 semitones at bar 5 (C major to D major)",
                   context="past/current events only; future melody pitches not exposed",
                   scheduler="one melody note per beat; EOS or crossing next input boundary means wait",
                   constraints="accompaniment channel 1, track 2, MIDI pitches 36..66, velocity 32..95, duration <=4 beats; cap 16 notes per melody input",
                   initial_key_condition="C major; no new key signature given to the model at bar 5",
                   trained=False, rl=False)
    for changed, name in [(False, "steady_duet"), (True, "changed_duet")]:
        print(f"Generating {name}", flush=True)
        lead = melody(changed)
        started = time.perf_counter()
        accompaniment, stats = accompany(EventSampler(model, args.seed, args.bpm, args.temperature), lead)
        accompaniment = resolve_overlaps(accompaniment)
        if not accompaniment:
            raise RuntimeError(f"No accompaniment generated for {name}")
        parts[name] = accompaniment
        summary[name] = dict(**summarize(accompaniment), scheduler=stats,
                             generation_seconds=round(time.perf_counter() - started, 3))
        write_result(directory, name, assemble(lead, accompaniment, args.bpm), not args.no_audio)
    before = lambda notes: [n for n in notes if n[1] < CHANGE_TICK]
    # Onsets/sampling histories must match until the intervention; durations may
    # be truncated differently by overlap repair after the intervention.
    prefix = lambda notes: [(n[1], *n[3:]) for n in before(notes)]
    assert prefix(parts["steady_duet"]) == prefix(parts["changed_duet"]), "Pre-change histories differ"
    frozen = parts["steady_duet"]
    write_result(directory, "changed_frozen_accompaniment",
                 assemble(melody(True), frozen, args.bpm), not args.no_audio)
    write_result(directory, "melody_only", assemble(melody(), [], args.bpm), not args.no_audio)
    write_result(directory, "changed_melody_only", assemble(melody(True), [], args.bpm), not args.no_audio)
    summary["changed_frozen_accompaniment"] = summarize(frozen)
    summary["prechange_accompaniment_onsets_identical"] = True
    summary["second_half_accompaniment_differs"] = (note_signature([n for n in parts["steady_duet"] if n[1] >= CHANGE_TICK]) !=
                                                    note_signature([n for n in parts["changed_duet"] if n[1] >= CHANGE_TICK]))
    summary["metric_caveat"] = "D-major fraction is a descriptive pitch-class statistic, not a musical-quality or causal adaptation score. One seed is not an evaluation."
    (directory / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved comparison in {directory}", flush=True)


if __name__ == "__main__":
    main()
