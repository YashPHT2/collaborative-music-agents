"""Generate and score a reproducible suite of musical intervention probes."""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import random
import statistics
import time
import wave

import mido

from accompany import (ROOT, MIDI, MIDIModel, torch, EventSampler, accompany,
                       resolve_overlaps, assemble, note_signature, write_result)
from evaluation_cases import build_cases, SUITE_VERSION, CHANGE, END
from music_metrics import diagnostics, adversarial_examples


def validate_midi(path, lead, accompaniment):
    midi = mido.MidiFile(path)
    active, decoded, tick = {}, [], 0
    for msg in mido.merge_tracks(midi.tracks):
        tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            key = (msg.channel,msg.note)
            if key in active:
                raise AssertionError(f"Same-pitch overlap in {path}: {key}")
            active[key] = (tick,msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            start,velocity = active.pop((msg.channel,msg.note))
            if tick <= start:
                raise AssertionError("Non-positive note duration")
            decoded.append(["note",start,tick-start,msg.channel,msg.note,velocity])
    assert not active, "Hanging notes"
    assert note_signature([n for n in decoded if n[3] == 0]) == note_signature(lead)
    assert note_signature([n for n in decoded if n[3] == 1]) == note_signature(accompaniment)
    assert all(n[1]+n[2]<=END for n in decoded)
    return dict(melody_preserved=True, accompaniment_preserved=True, notes_balanced=True,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def validate_audio(path):
    import numpy as np
    with wave.open(str(path),"rb") as audio:
        assert audio.getsampwidth() == 2
        duration = audio.getnframes()/audio.getframerate()
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32)/32768
    peak = float(np.max(np.abs(pcm)))
    assert 0.001 < peak < 0.999, f"Silent/clipped audio: {path}"
    return dict(seconds=round(duration,3),peak=round(peak,4))


def paired_summary(rows):
    groups = {}
    for row in rows:
        if row["window"] == "after" and row["condition"] != "steady":
            groups.setdefault((row["case_id"],row["seed"]),{})[row["variant"]] = row
    summary = []
    metrics = ["scale_fit_duration","intended_chord_fit_duration",
               "rough_melody_interval_fraction","melody_onset_alignment",
               "notes_per_beat","dominant_pitch_share","sounding_coverage"]
    for case_id in sorted({key[0] for key in groups}):
        pairs = [value for (name,_),value in groups.items() if name == case_id]
        item = dict(case_id=case_id, split=pairs[0]["responding"]["split"], seeds=len(pairs))
        for metric in metrics:
            differences = [pair["responding"][metric]-pair["frozen"][metric] for pair in pairs
                           if pair["responding"][metric] is not None and pair["frozen"][metric] is not None]
            item[metric] = dict(mean_delta=round(statistics.mean(differences),6),
                                min_delta=round(min(differences),6), max_delta=round(max(differences),6),
                                valid_pairs=len(differences)) if differences else None
        summary.append(item)
    return summary


def write_listening_page(root, examples):
    groups = []
    for example in examples:
        if example["condition"] == "steady":
            continue
        # Blinding is for casual listening, not protection against source inspection.
        candidates = [("responding",example["responding"]),("frozen",example["frozen"])]
        random.Random(example["case_id"]+str(example["seed"])).shuffle(candidates)
        groups.append(dict(id=example["case_id"], seed=example["seed"], description=example["description"],
                           split=example["split"], a=candidates[0][1], b=candidates[1][1],
                           methods=dict(A=candidates[0][0], B=candidates[1][0])))
    data = json.dumps(groups).replace("</","<\\/")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>Piano accompaniment listening test</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;background:#111827;color:#e5e7eb;max-width:960px;margin:36px auto;padding:0 20px}
h1{line-height:1.2}section{border:1px solid #374151;border-radius:12px;padding:20px;margin:18px 0;background:#182233}
audio{width:100%;margin:8px 0}button,select{background:#293b57;color:white;border:1px solid #64748b;padding:9px 14px;border-radius:6px;margin:4px;cursor:pointer}
button[aria-pressed=true]{background:#047857}small,.muted{color:#b3bed0}.players{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:650px){.players{grid-template-columns:1fr}}.mapping{color:#fbbf24}a{color:#93c5fd}
</style><h1>Piano accompaniment: what sounds better?</h1>
<p>Each pair has the same melody. The musical change starts at <b>9.6 seconds</b> (bar 5).
Listen for a suitable response, musical continuity, and pleasant accompaniment. A and B use the same piano sound and gain.</p>
<p class="muted">One candidate responds to the changed melody; the other keeps its previous accompaniment.
The order is hidden until you reveal it. These are scripted probes, not proof of general musical ability.</p>
<button id="export">Export my ratings</button><button id="reveal">Reveal A/B methods</button>
<p id="status" role="status"></p><div id="pairs"></div>
<p><a href="REPORT.md">Read the diagnostic report</a> · <a href="metrics.csv">Download all metrics</a> · <a href="suite.json">Inspect test cases</a></p>
<script>
const pairs=__DATA__;
const storageKey='rl-piano-evaluation-v1';let ratings={};
try{ratings=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch{}
const container=document.getElementById('pairs');
for(const pair of pairs){
 const card=document.createElement('section');const title=document.createElement('h2');
 const titles={key_shift:'Key change',melody_change:'Melody change',rhythm_change:'Rhythm change',combined_change:'Key and rhythm change'};
 const condition=pair.id.replace(/^theme_[ab]_/, '');title.textContent=(pair.id.startsWith('theme_a')?'Theme A: ':'Theme B: ')+titles[condition];card.append(title);
 const description=document.createElement('p');description.textContent=pair.description+' Seed '+pair.seed+' · '+(pair.split==='development'?'Development set':'Reserved evaluation set');card.append(description);
 const players=document.createElement('div');players.className='players';
 for(const label of ['A','B']){const box=document.createElement('div');const name=document.createElement('b');name.textContent=label;box.append(name);
  const player=document.createElement('audio');player.controls=true;player.preload='metadata';player.src=pair[label.toLowerCase()]+'?playback=2';player.setAttribute('aria-label',pair.id+' candidate '+label);
  player.addEventListener('play',()=>document.querySelectorAll('audio').forEach(other=>{if(other!==player)other.pause()}));box.append(player);
  const skip=document.createElement('button');skip.textContent='Start at change';skip.onclick=async()=>{try{
   if(player.readyState<1){await new Promise((resolve,reject)=>{player.addEventListener('loadedmetadata',resolve,{once:true});player.addEventListener('error',reject,{once:true});player.load()})}
   player.currentTime=9.6;await player.play()
  }catch{document.getElementById('status').textContent='Audio could not start. Use the player controls.'}};box.append(skip);players.append(box)}
 card.append(players);const buttons=[];for(const vote of ['A','B','Tie','Neither good']){const button=document.createElement('button');button.textContent=vote;button.setAttribute('aria-pressed',ratings[pair.id]===vote);
  button.onclick=()=>{ratings[pair.id]=vote;try{localStorage.setItem(storageKey,JSON.stringify(ratings))}catch{}
   buttons.forEach(b=>b.setAttribute('aria-pressed',b===button));document.getElementById('status').textContent=Object.keys(ratings).length+' pairs rated.'};buttons.push(button);card.append(button)}
 const mapping=document.createElement('p');mapping.className='mapping';mapping.hidden=true;mapping.textContent='A: '+pair.methods.A+' · B: '+pair.methods.B;card.append(mapping);container.append(card)
}
document.getElementById('reveal').onclick=()=>document.querySelectorAll('.mapping').forEach(el=>el.hidden=false);
document.getElementById('export').onclick=()=>{const payload={suite_version:'1.0',created_at:new Date().toISOString(),ratings:pairs.map(p=>({case_id:p.id,seed:p.seed,vote:ratings[p.id]||null,methods:p.methods}))};
 const a=document.createElement('a');const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.href=url;a.download='listening-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></html>""".replace("__DATA__",data)
    (root/"listen.html").write_text(page,encoding="utf-8")


def write_report(root, summary, sanity, manifest):
    lines = ["# Frozen SkyTNT accompaniment baseline", "",
             f"Suite {SUITE_VERSION}: {manifest['generations']} model generations, {manifest['midi_files']} MIDI files, "
             f"{manifest['audio_files']} listening clips. No weights were trained.", "",
             "Two scripted melody families × five conditions × three seeds. The second family and its seeds are "
             "designated reserved evaluation for future tuning. This does not mean the source music is verified absent "
             "from pretraining, and this tiny suite is not a representative benchmark.", "",
             "## Listening", "", "Open `listen.html` and rate the eight A/B pairs before revealing the methods. "
             "Only the first seed of each family/condition is rendered; all three seeds are scored and saved as MIDI. "
             "The change is at 9.6 seconds. Ratings stay in this browser until exported; no human ratings have been collected.", "",
             "## Paired diagnostics after the change", "",
             "Deltas are responding minus frozen accompaniment, averaged across three seeds. They are descriptive, "
             "not listener preferences or statistically established improvements. Chord/scale fit higher and rough "
             "interval fraction lower match these narrow reference rules; none establishes musical quality.", "",
             "| Case | Scale-fit Δ | Intended-chord-fit Δ | Rough-interval Δ | Notes/beat Δ |",
             "|---|---:|---:|---:|---:|"]
    for item in summary:
        def value(metric):
            return f"{item[metric]['mean_delta']:+.3f}" if item[metric] else "unavailable"
        lines.append(f"| {item['case_id']} | {value('scale_fit_duration')} | {value('intended_chord_fit_duration')} | "
                     f"{value('rough_melody_interval_fraction')} | {value('notes_per_beat')} |")
    lines += ["", "## What the metrics mean", "",
              "- Scale fit: note-duration-weighted membership of the scripted local major scale.",
              "- Intended chord fit: duration-weighted membership of a supplied bar-level chord plan used only by the evaluator. "
              "Alternative valid harmonizations can score lower; passing notes are allowed musically.",
              "- Rough intervals: overlap-weighted accompaniment–melody pairs with pitch-class interval 1, 6 or 11 semitones. "
              "Tension and its resolution are not modeled; this is not an invalid-note detector.",
              "- Coverage, density, polyphony, dominant-pitch share, pitch entropy and repeated-pitch runs expose silence, flooding and repetition.",
              "- Onset alignment: accompaniment attacks within 1/16 beat of a melody attack. Syncopation can be intentional; "
              "the input scheduler strongly affects this statistic.",
              "- Empty harmony scores are null, never a perfect score. Notes sustained into the window are scored for fit, "
              "while attack counts include only onsets inside the window.", "",
              "## Reward sanity probes", "",
              "| Fixture | Scale fit | Coverage | Notes/beat | Dominant pitch | Max polyphony |",
              "|---|---:|---:|---:|---:|---:|"]
    for name,metrics in sanity.items():
        lines.append(f"| {name} | {metrics['scale_fit_duration']} | {metrics['sounding_coverage']} | "
                     f"{metrics['notes_per_beat']} | {metrics['dominant_pitch_share']} | {metrics['max_polyphony']} |")
    lines += ["", "A repeated C and a continuous scale can have perfect scale fit. Silence has no fit score; "
              "dense chromatic notes show high polyphony/density. These fixtures demonstrate why a single tonal reward "
              "or an unvalidated weighted sum would be unsafe to optimize.", "",
              "## Controls and limits", "",
              "The unchanged and intervention runs use identical seeds and verified identical raw accompaniment "
              "before bar 5. The frozen comparison reuses the steady run's complete accompaniment. Both methods "
              "receive the same changed melody during output mixing; only responding regenerates its accompaniment.", "",
              "Only past/current melody pitches enter the policy. The scheduler does know the next input time and "
              "discards candidates crossing that boundary; EOS means wait. This gives rhythm-related assistance "
              "and is not fully unconstrained live improvisation. Fixed-note durations are known at onset. "
              "Future input times, register masks, per-input density cap and output overlap cleanup are held constant "
              "as procedures, but different rhythm inputs create different decision windows.", "",
              "Accompaniment: channel 1, MIDI pitches 36..66, velocities 32..95, durations 1/16 beat..4 beats, "
              "at most 16 notes per input. Initial key prompt stays C major even after modulation. "
              "Tempo 100 BPM, 4/4, eight bars, temperature 0.9, top-p 0.98, top-k 20. "
              "Mix and soundfont/gain are identical. No gain normalization per candidate.", "",
              "## Next experiment", "",
              "Collect blind preferences on the listening page and inspect weak cases. Choose an objective only if "
              "the diagnostic agrees with listening judgments on development cases. Keep the reserved family out of "
              "reward/weight selection. Compare later RL against this frozen checkpoint with the same sampler and "
              "against candidate reranking before attributing benefits to RL. Do not claim a musical-quality gain "
              "from the automatic numbers alone.", "",
              "Raw results: `metrics.csv`, `paired_summary.json`, `suite.json`, `run.json`, and `sanity.json`."]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-audio",action="store_true")
    parser.add_argument("--device",choices=["auto","cuda","cpu"],default="auto")
    args = parser.parse_args()
    root = ROOT/"outputs"/"evaluation_v1"
    root.mkdir(parents=True,exist_ok=True)
    cases = build_cases()
    (root/"suite.json").write_text(json.dumps(dict(version=SUITE_VERSION,cases=[asdict(c) for c in cases]),indent=2)+"\n")
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    torch.set_num_threads(4)
    torch.manual_seed(0)
    print(f"Loading frozen SkyTNT on {device}",flush=True)
    model = MIDIModel.from_pretrained(ROOT/"models"/"skytnt-tv2o-medium",
                                      dtype=dtype,local_files_only=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rows, examples, steady, checks, details = [], [], {}, {}, []
    generations, midi_files, audio_files = 0,0,0
    started = time.perf_counter()
    for case in cases:
        for seed in case.seeds:
            print(f"[{generations+1}/30] {case.id} seed={seed}",flush=True)
            sampler = EventSampler(model,seed,100,0.9)
            before = time.perf_counter()
            raw, scheduler = accompany(sampler,case.melody,verbose=False)
            elapsed = time.perf_counter()-before
            generated = resolve_overlaps(raw)
            generations += 1
            key = (case.family,seed)
            if case.condition == "steady":
                steady[key] = (raw,generated)
            else:
                previous_raw,previous_clean = steady[key]
                assert note_signature([n for n in raw if n[1]<CHANGE]) == note_signature([n for n in previous_raw if n[1]<CHANGE]), "Input histories differ before intervention"
            variants = {"responding":generated}
            if case.condition != "steady":
                variants["frozen"] = steady[key][1]
            example = dict(case_id=case.id,condition=case.condition,description=case.description,
                           split=case.split,seed=seed)
            for variant,notes in variants.items():
                directory = root/case.id/f"seed{seed}"
                directory.mkdir(parents=True,exist_ok=True)
                (directory/f"{variant}_notes.json").write_text(json.dumps(notes)+"\n")
                audio = not args.no_audio and seed==case.seeds[0]
                midi_path = write_result(directory,variant,assemble(case.melody,notes,100),audio)
                midi_files += 1
                relative = midi_path.relative_to(root).as_posix()
                checks[relative] = validate_midi(midi_path,case.melody,notes)
                if audio:
                    checks[relative]["audio"] = validate_audio(midi_path.with_suffix(".wav"))
                    audio_files += 1
                    example[variant] = midi_path.with_suffix(".mp3").relative_to(root).as_posix()
                for window,left,right in [("before",0,CHANGE),("after",CHANGE,END)]:
                    rows.append(dict(case_id=case.id,split=case.split,condition=case.condition,seed=seed,
                                     variant=variant,window=window,
                                     **diagnostics(case.melody,notes,case.harmony,left,right)))
            if not args.no_audio and seed == case.seeds[0]:
                examples.append(example)
            details.append(dict(case_id=case.id,seed=seed,generation_seconds=round(elapsed,3),
                                raw_notes=len(raw),clean_notes=len(generated),scheduler=scheduler))
    sanity_case = cases[0]
    sanity = {name:diagnostics(sanity_case.melody,notes,sanity_case.harmony,0,END)
              for name,notes in adversarial_examples().items()}
    with (root/"metrics.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = paired_summary(rows)
    manifest = dict(suite_version=SUITE_VERSION,generations=generations,midi_files=midi_files,
                    audio_files=audio_files,device=device,dtype=str(dtype),
                    elapsed_seconds=round(time.perf_counter()-started,3),
                    checkpoint="skytnt/midi-model-tv2o-medium",
                    revision="0f8f265d4330f4e46527ac2313200254c5757f5f",
                    source_revision="f504d5cb58f769ab0f2909c679238f6621034573",
                    bpm=100,temperature=0.9,top_p=0.98,top_k=20,
                    trained=False,human_preferences_collected=False,
                    prechange_histories_verified=True,results=details)
    for filename,data in [("run.json",manifest),("paired_summary.json",summary),
                          ("sanity.json",sanity),("verification.json",checks)]:
        (root/filename).write_text(json.dumps(data,indent=2)+"\n")
    write_report(root,summary,sanity,manifest)
    if not args.no_audio:
        write_listening_page(root,examples)
    print(f"DONE: {generations} generations; {midi_files} MIDI files; {audio_files} clips",flush=True)
    print(root/"REPORT.md",flush=True)


if __name__ == "__main__":
    main()
