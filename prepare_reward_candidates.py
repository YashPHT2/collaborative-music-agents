"""Fresh original agent outputs for taste labels. No policy training or MP3 edits."""
import hashlib
import json
from pathlib import Path
import torch
import offline_ensemble as oe
import note_trio as nt

OUT = nt.ROOT / 'outputs/reward_original_candidates_v1'
SOURCE = nt.ROOT / 'outputs/offline_ensemble_v1/revised.pt'
SEEDS = (9886101, 9886102)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model, adapters, agents, planner, conditions = oe.load(SOURCE)
    manifest = dict(source=str(SOURCE.resolve()), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    purpose='Fresh original compositions for listener labels, not a claim of improvement',
                    reference_audio_samples_used=False, reference_melodies_used=False,
                    policy_training_started=False, seeds=list(SEEDS), candidates=[])
    for i, seed in enumerate(SEEDS):
        name = f'candidate_{i+1}'
        path = OUT / f'{name}_episode.json'
        if path.exists():
            ep = json.loads(path.read_text())
        else:
            print('Generating original candidate', i+1, seed, flush=True)
            ep = oe.compose(model, adapters, agents, planner, conditions, 8, seed, 'revised')
            assert sorted(ep['notes']) == sorted(d['note'] for d in ep['decisions'])
            nt.export(OUT, name, ep, allow_silent_stems=True)
        manifest['candidates'].append(dict(id=name, seed=seed, audio=f'{name}_mix.mp3',
            episode=f'{name}_episode.json', note_count=len(ep['notes']),
            notes_sha256=hashlib.sha256(json.dumps(ep['notes']).encode()).hexdigest(),
            role_counts={r: sum(n[3] == ch for n in ep['notes']) for ch, r in enumerate(oe.ROLES)},
            all_notes_match_actions=sorted(ep['notes']) == sorted(d['note'] for d in ep['decisions'])))
        (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    cards = ''.join(f'''<section><h2>Original candidate {i+1}</h2>
<audio controls src="{r['audio']}"></audio><p>Solo: ''' + ' · '.join(f'<a href="{r["id"]}_{role}.mp3">{role}</a>' for role in oe.ROLES) + '</p>' + ''.join(f'<label>{dimension} <select data-id="{r["id"]}" data-dimension="{dimension}"><option value="">Choose</option>'+''.join(f'<option>{score}</option>' for score in range(1,6))+'</select></label>' for dimension in ('enjoyment','coherence','development','ensemble_fit')) + '</section>' for i,r in enumerate(manifest['candidates']))
    (OUT / 'listen.html').write_text('''<!doctype html><meta charset="utf-8"><title>Original agent candidates</title>
<style>body{font:19px system-ui;max-width:850px;margin:40px auto;padding:20px;background:#101724;color:white}section{background:#1d2b40;border-radius:14px;padding:24px;margin:20px 0}audio{width:100%}label{display:block;margin:12px 0}select,button{font:inherit}a{color:#9cd5ff}</style>
<h1>New notes, new compositions</h1><p>Two fresh seeds from the earlier model. Neither uses audio or melodies cut from your MP3s. These are candidates for taste labels, not a newly improved model.</p><p>Rate each independently: 1 = poor, 5 = excellent. It is fine to dislike both.</p>''' + cards + '''<button id="save">Download ratings</button><script>
document.getElementById('save').onclick=async()=>{const manifest=await(await fetch('manifest.json')).json();const ratings={};document.querySelectorAll('select').forEach(s=>{(ratings[s.dataset.id]??={})[s.dataset.dimension]=s.value?Number(s.value):null});const blob=new Blob([JSON.stringify({schema:'original-agent-ratings-v1',manifest,ratings},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='original-agent-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),5000)};
</script>''', encoding='utf-8')
    print('Original candidates ready:', OUT / 'listen.html', flush=True)


if __name__ == '__main__':
    main()
