"""Original v2 agent notes with explicit arrangement and production controls."""
import hashlib
import json
import torch
import offline_ensemble as oe
import ensemble_repair as er
import note_trio as nt
from train_ensemble_repair import corpus,write_json

OUT=nt.ROOT/'outputs'/'cinematic_reference_demo_v1'
SOURCE=nt.ROOT/'outputs'/'offline_ensemble_v1'/'revised.pt'


def build_plan(quantiles):
    def transform(plan,bar,bars,ledger):
        phase=bar/max(1,bars-1)
        # A human-chosen energy arc controls summaries, never inserts notes.
        energy=phase if phase<.85 else .85-(phase-.85)*2
        for index in range(12,21):
            low,high=quantiles[str(index)]
            plan[index]=low+(high-low)*energy
        # Return toward the generated opening's tonal colour, without copying
        # its melody or adding a fixed scale/chord progression.
        opening=oe.description(ledger,0,bars)
        if bar>=bars-3 and opening[:12].sum()>0:
            plan[:12]=.75*plan[:12]+.25*opening[:12]
        return plan.clamp(0,1)
    return transform


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    _,_,_,quantiles=corpus()
    model,adapters,agents,planner,conditions=oe.load(SOURCE)
    ep=oe.compose(model,adapters,agents,planner,conditions,12,9619102,'revised',plan_transform=build_plan(quantiles))
    ep['tempo_map']=[(bar*1920,82+32*bar/11) for bar in range(12)]
    ep['profile']='gradual_build_with_opening_colour_return'
    # Generation retains the backbone's 100 BPM context; the tempo map is an
    # explicit performance control applied to the completed score.
    ep['generation_bpm']=100
    write_json(OUT/'episode.json',ep)
    write_json(OUT/'plans.json',ep['plans'])
    dry=nt.export(OUT,'clear',ep,allow_silent_stems=True)
    ep['room_effect']=True
    spacious=nt.export(OUT,'spacious',ep,stems=False,allow_silent_stems=True)
    write_json(OUT/'verification.json',dict(source=str(SOURCE.resolve()),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               seed=ep['seed'],bars=12,music_seconds=nt.tempo_seconds(12,100,ep['tempo_map']),
               note_count=len(ep['notes']),participation=er.participation(ep['notes'],12),
               all_notes_match_agent_actions=sorted(ep['notes'])==sorted(d['note'] for d in ep['decisions']),
               dataset_prefix=False,tempo_map=ep['tempo_map'],generation_bpm=100,
               arrangement_controls='Training-set density/register/velocity quantiles, tempo build and generated opening chroma return',
               production='SoundFont instruments and optional algorithmic room echoes, identical notes in both mixes',
               limitations='No new training. This is a 12-bar extrapolation experiment, not validated long-form learning or a reproduction of either reference.',
               audio=dict(clear=dry,spacious=spacious)))
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Across the Quiet — agent demo</title><style>body{font:19px system-ui;background:#101724;color:#e8edf5;max-width:850px;margin:45px auto;padding:20px}p{line-height:1.65}.card{background:#1c293d;padding:24px;border-radius:14px}audio{width:100%;margin:18px 0}select,button{font:inherit;padding:10px;margin:6px}a{color:#9bd2ff}</style>
<h1>Across the Quiet</h1><p>An original short experiment with organ, strings, and cello. The earlier model chooses all notes. A gradual tempo/energy arc and optional room effect shape the performance.</p>
<div class="card"><label>Listen to <select id="version"><option value="spacious_mix">Spacious mix</option><option value="clear_mix">Clear mix — same notes</option><option value="clear_organ">Organ</option><option value="clear_strings">Strings</option><option value="clear_cello">Cello</option></select></label><audio id="audio" controls src="spacious_mix.mp3"></audio><button id="play">Play from beginning</button><p id="message">Room effect is production processing. Solo levels are adjusted for listening.</p></div>
<p>This demo has no supplied melody, chord progression, or copied reference notes. The notes are agent-generated; the energy and tempo direction are explicit controls. It is an experiment beyond the model's eight-bar training windows, with no new RL training.</p><p><a href="clear_mix.mid">Download MIDI with tempo changes</a> · <a href="verification.json">Generation and verification details</a></p>
<script>const a=document.getElementById('audio'),v=document.getElementById('version'),m=document.getElementById('message');v.onchange=()=>{a.pause();a.src=v.value+'.mp3';a.load()};document.getElementById('play').onclick=()=>{a.currentTime=0;a.play().catch(()=>m.textContent='Use the audio play control to start.')};</script></html>'''
    (OUT/'listen.html').write_text(page,encoding='utf-8')
    print('Demo ready:',OUT/'listen.html',flush=True)


if __name__=='__main__':main()
