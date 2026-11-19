"""Pre-policy reward experiment: pretrained preferences + MIDI sequence ranking.

Frozen protocol, composition-separated local tests, per-role failures, controls,
and explicit fail-closed RL acceptance. Does not train any music policy.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import soundfile as sf
import torch

from ensemble_coherence_reward import ROLES, KINDS, CoherenceRanker, disrupt, sequence, seed_for, pairwise_loss
from preference_reward import PreferenceReward

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/reward_validation_v2'
PROTOCOL = dict(version=2, seed=988510, bars=8, bpm=100,
    pretrained_preference='TuneJury A1 CLAP audio only; frozen, no local audio-head fitting',
    temporal_model='Bidirectional GRU; three role-specific heads; pairwise logistic loss',
    temporal_labels='Synthetic: unchanged real score preferred to one disrupted part. NOT human preferences.',
    splits='Existing corpus composition groups; train/validation/test remain separate',
    caps=dict(train=240, validation=40, test=12), temporal_updates=256,
    temporal_selection='Validation macro role/kind win rate, evaluate every 64 updates; test once after selection',
    musical_disruptions=list(KINDS),
    temporal_augmentation='Same global pitch-class rotation on both members; equal role/group sampling',
    audio_aggregation='Mean score of deterministic covering 10-second windows; no random crops',
    style='CLAP cosine against calibration-only references; reported separately, never treated as quality',
    required_overall_win_rate=.80, required_per_role_and_kind_win_rate=2/3,
    audio_control_p90_pairwise_probability_shift=.10,
    sanity='All originals must beat silence and RMS-matched noise',
    rl_gate='All automatic gates plus verified human preferences on agent-generated music; otherwise blocked',
    limits=['Known v1 audio stress tests are diagnostic reuse, not a fresh blind test.',
            'Only one test composition in the source corpus has all three real labelled roles.',
            'Synthetic disruptions may be musically valid. Listener review is required.',
            'Short-window preference scores cannot prove long-form composition quality.'])


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def prepare_dataset():
    path = OUT / 'dataset.json'
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(ROOT / 'assets/composition_v1/corpus.json.gz', 'rt') as f:
        corpus = json.load(f)
    grouped = {s: {} for s in PROTOCOL['caps']}
    # Fixed ranking prefers the scarce complete trio, then samples other groups
    # by hash. Selection never reads model scores or listens to the results.
    for rec in sorted(corpus, key=lambda r: (-len(r['available_roles']), seed_for(r['group']))):
        split = rec['split']
        if len(rec['available_roles']) < 2 or rec['group'] in grouped[split]:
            continue
        if len(grouped[split]) >= PROTOCOL['caps'][split]:
            continue
        end = max((n[1] + n[2] for n in rec['notes']), default=0)
        starts = list(range(0, end - 15360 + 1, 15360))
        rng = np.random.default_rng(seed_for(rec['group']))
        rng.shuffle(starts)
        for start in starts[:20]:
            notes = []
            for n in rec['notes']:
                if start <= n[1] < start + 15360:
                    note = copy.deepcopy(n)
                    note[1] -= start
                    note[2] = min(note[2], 15360 - note[1]) // 60 * 60
                    if note[2] >= 60:
                        notes.append(note)
            roles = [i for i in range(3) if sum(n[3] == i for n in notes) >= 8]
            if len(roles) < 2:
                continue
            # Exclude unsupported minor fragments, preserving each retained role.
            notes = [n for n in notes if n[3] in roles]
            grouped[split][rec['group']] = dict(group=rec['group'], song=rec['song'],
                source=rec['source'], split=split, start_tick=start, roles=roles, notes=notes,
                notes_sha256=hashlib.sha256(json.dumps(notes).encode()).hexdigest())
            break
    result = {s: list(v.values()) for s, v in grouped.items()}
    assert not (set(grouped['train']) & (set(grouped['validation']) | set(grouped['test'])))
    assert not (set(grouped['validation']) & set(grouped['test']))
    fingerprints = [{r['notes_sha256'] for r in result[s]} for s in result]
    assert not any(a & b for i, a in enumerate(fingerprints) for b in fingerprints[i + 1:])
    write(path, result)
    write(OUT / 'dataset_audit.json', dict(counts={s: len(v) for s, v in result.items()},
        compositions_per_role={s: {role: sum(i in r['roles'] for r in rows) for i, role in enumerate(ROLES)} for s, rows in result.items()},
        full_trios={s: sum(len(r['roles']) == 3 for r in rows) for s, rows in result.items()},
        source_corpus_sha256=sha(ROOT / 'assets/composition_v1/corpus.json.gz'),
        scoring_selection=False, reference_audio_used_for_temporal_training=False))
    return result


def contrasts(rows):
    examples = []
    excluded = []
    for record in rows:
        positive = sequence(record['notes'])
        for role in record['roles']:
            for kind in KINDS:
                negative = disrupt(record['notes'], role, kind, 15360, seed_for(record['group'] + kind + str(role)))
                if negative == sorted(record['notes'], key=lambda n: (n[1], n[3], n[4])):
                    excluded.append(dict(group=record['group'], role=ROLES[role], kind=kind,
                                         reason='No change, e.g. scrambling a constant-pitch part'))
                    continue
                examples.append(dict(group=record['group'], role=role, kind=kind,
                                     positive=positive, negative=sequence(negative)))
    if rows:
        write(OUT / ('excluded_noop_' + rows[0]['split'] + '.json'), excluded)
    return examples


@torch.inference_mode()
def symbolic_results(model, examples, device):
    rows = []
    for start in range(0, len(examples), 64):
        batch = examples[start:start + 64]
        pos = model(torch.tensor(np.stack([r['positive'] for r in batch]), device=device))
        neg = model(torch.tensor(np.stack([r['negative'] for r in batch]), device=device))
        for i, r in enumerate(batch):
            margin = float(pos[i, r['role']] - neg[i, r['role']])
            rows.append(dict(group=r['group'], role=ROLES[r['role']], kind=r['kind'], margin=margin, win=margin > 0))
    return summarize_pairs(rows)


def summarize_pairs(rows):
    by_role = {role: float(np.mean([r['win'] for r in rows if r['role'] == role])) for role in ROLES if any(r['role'] == role for r in rows)}
    by_role_kind = {role: {kind: float(np.mean([r['win'] for r in rows if r['role'] == role and r['kind'] == kind])) for kind in KINDS} for role in by_role}
    rng = np.random.default_rng(PROTOCOL['seed'])
    groups = sorted({r['group'] for r in rows})
    group_rates = [np.mean([r['win'] for r in rows if r['group'] == group]) for group in groups]
    boots = np.mean(rng.choice(group_rates, size=(2000, len(groups)), replace=True), axis=1)
    return dict(count=len(rows), win_rate=float(np.mean([r['win'] for r in rows])), by_role=by_role,
                by_role_kind=by_role_kind, composition_bootstrap_95_interval=np.quantile(boots, [.025, .975]).tolist(), pairs=rows)


def train_temporal(data):
    if (OUT / 'coherence.pt').exists():
        if not (OUT / 'coherence_test.json').exists():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = CoherenceRanker().to(device).eval()
            state = torch.load(OUT / 'coherence.pt', weights_only=True, map_location=device)
            model.load_state_dict(state['state_dict'])
            result = symbolic_results(model, contrasts(data['test']), device)
            result['selected_update'] = state['selected_update']
            result['label_source'] = 'synthetic structural contrasts, not listener taste'
            write(OUT / 'coherence_test.json', result)
            print('Temporal held-out result:', result['win_rate'], result['by_role_kind'], flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(PROTOCOL['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CoherenceRanker().to(device)
    train, validation = contrasts(data['train']), contrasts(data['validation'])
    roles = {role: {} for role in range(3)}
    for row in train:
        roles[row['role']].setdefault(row['group'], []).append(row)
    available = [role for role in roles if roles[role]]
    if len(available) != 3:
        raise ValueError('Cannot train all role heads without training examples for every role')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    rng = np.random.default_rng(PROTOCOL['seed'])
    best, logs = -1, []
    for update in range(1, PROTOCOL['temporal_updates'] + 1):
        model.train()
        batch = []
        for _ in range(48):
            role = int(rng.choice(available))
            group = rng.choice(sorted(roles[role]))
            row = roles[role][group][int(rng.integers(len(roles[role][group])))]
            shift = int(rng.integers(12))
            pair = []
            for key in ('positive', 'negative'):
                x = row[key].reshape(-1, 3, 19).copy()
                x[:, :, :12] = np.roll(x[:, :, :12], shift, axis=2)
                pair.append(x.reshape(-1, 57))
            batch.append((pair, role))
        positive = torch.tensor(np.stack([x[0][0] for x in batch]), device=device)
        negative = torch.tensor(np.stack([x[0][1] for x in batch]), device=device)
        role_tensor = torch.tensor([x[1] for x in batch], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = pairwise_loss(model, positive, negative, role_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if update % 16 == 0:
            print(f'Reward-head update {update}/256 pairwise_loss={float(loss):.4f} (music policies unchanged)', flush=True)
        if update % 64 == 0:
            model.eval()
            result = symbolic_results(model, validation, device)
            macro = np.mean([v for kinds in result['by_role_kind'].values() for v in kinds.values()])
            logs.append(dict(update=update, loss=float(loss), validation_macro=float(macro), by_role_kind=result['by_role_kind']))
            if macro > best:
                best = macro
                torch.save(dict(version=1, state_dict=model.cpu().state_dict(), selected_update=update,
                                protocol_sha256=sha(OUT / 'protocol.json')), OUT / 'coherence.pt')
                model.to(device)
            write(OUT / 'coherence_training.json', logs)
    state = torch.load(OUT / 'coherence.pt', weights_only=True, map_location=device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    result = symbolic_results(model, contrasts(data['test']), device)
    result['selected_update'] = state['selected_update']
    result['label_source'] = 'synthetic structural contrasts, not listener taste'
    write(OUT / 'coherence_test.json', result)
    print('Temporal held-out result:', result['win_rate'], result['by_role_kind'], flush=True)


def render(notes, name):
    from note_trio import midi_file
    midi = OUT / 'audio' / (name + '.mid')
    audio = midi.with_suffix('.flac')
    if not audio.exists():
        midi_file(midi, notes, bpm=100)
        fluid = next((ROOT / 'tools/fluidsynth').rglob('fluidsynth.exe'))
        wave = midi.with_suffix('.wav')
        subprocess.run([str(fluid), '-ni', '-F', str(wave), '-r', '48000', '-g', '.6',
                        str(ROOT / 'assets/soundfont.sf2'), str(midi)], check=True, capture_output=True)
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(wave), '-af',
                        'apad,atrim=duration=19.2,afade=t=out:st=18.9:d=0.3', str(audio)], check=True)
        wave.unlink()
    return audio


def score_audio(data, wait_for_render_batch=False):
    from validate_reference_reward import decode, SR, FILES, SPLITS
    model = PreferenceReward()
    print('Frozen human-preference model loaded', flush=True)
    profiles = {}
    for name, path in FILES.items():
        audio = decode(path)
        embeddings = []
        for start in SPLITS[name]['calibration']:
            embeddings.append(model.evaluate(audio[start * SR:(start + 12) * SR], SR)['embedding'])
        profile = np.mean(embeddings, axis=0)
        profiles[name] = (profile / np.linalg.norm(profile)).tolist()
    write(OUT / 'style_profiles.json', dict(profiles=profiles, usage='Style diagnostic only; excluded from quality score',
                                          reference_sources={k: dict(path=str(p), sha256=sha(p)) for k, p in FILES.items()}))
    cache_path = OUT / 'audio_scores.json'
    rows = json.loads(cache_path.read_text()) if cache_path.exists() else []
    done = {r['id'] for r in rows}

    def score(path, identity, **metadata):
        if identity in done:
            return next(r for r in rows if r['id'] == identity)
        audio, sr = sf.read(path, dtype='float32', always_2d=True)
        result = model.evaluate(audio, sr, profiles)
        embedding = result.pop('embedding')
        features = OUT / 'embeddings'
        features.mkdir(exist_ok=True)
        np.save(features / (identity + '.npy'), np.asarray(embedding, dtype=np.float32))
        row = dict(id=identity, audio=str(path.resolve()), sha256=sha(path), **metadata, **result)
        rows.append(row)
        done.add(identity)
        write(cache_path, rows)
        print(f'Preference scored {len(rows)}: {identity} = {row["preference"]:.3f}', flush=True)
        return row

    # Identical known stress-test files, for a fair baseline comparison. No claim
    # that reusing the previous stress set is a new blind audio benchmark.
    for row in json.loads((ROOT / 'outputs/reward_validation_v1/scores.json').read_text()):
        identity = f'reference_{row["reference"]}_{row["start_seconds"]}_{row["kind"]}'
        score(ROOT / 'outputs/reward_validation_v1' / row['audio'], identity,
              domain='reference_diagnostic', group=f'{row["reference"]}_{row["start_seconds"]}',
              reference=row['reference'], kind=row['kind'])
    if wait_for_render_batch:
        verification = OUT / 'midi_roundtrip_render_verification.json'
        started = time.monotonic()
        while not verification.exists():
            if time.monotonic() - started > 600:
                raise TimeoutError('External MIDI render batch did not finish; scores are retained for resume')
            time.sleep(1)
        if json.loads(verification.read_text())['failures']:
            raise ValueError('MIDI render/round-trip verification failed')
    for i, record in enumerate(data['test']):
        original = render(record['notes'], f'composition_{i:02}_original')
        score(original, f'midi_{i}_original', domain='midi_test', group=record['group'], kind='original')
        for role in record['roles']:
            for kind in KINDS:
                notes = disrupt(record['notes'], role, kind, 15360, seed_for(record['group'] + kind + str(role)))
                if notes == sorted(record['notes'], key=lambda n: (n[1], n[3], n[4])):
                    continue
                path = render(notes, f'composition_{i:02}_{ROLES[role]}_{kind}')
                score(path, f'midi_{i}_{role}_{kind}', domain='midi_test', group=record['group'], kind=kind, role=ROLES[role])
        # Same renderer and fixed notes, only benign audio changes or sanity tests.
        audio, sr = sf.read(original, dtype='float32', always_2d=True)
        rng = np.random.default_rng(PROTOCOL['seed'] + i)
        for kind in ['gain_minus_6db', 'silence', 'noise']:
            altered = audio * .501187 if kind == 'gain_minus_6db' else np.zeros_like(audio)
            if kind == 'noise':
                altered = rng.normal(size=audio.shape).astype(np.float32) * np.sqrt(np.mean(audio ** 2))
            path = OUT / 'audio' / f'composition_{i:02}_{kind}.flac'
            if not path.exists():
                sf.write(path, altered, sr, subtype='PCM_24')
            score(path, f'midi_{i}_{kind}', domain='midi_test', group=record['group'], kind=kind)
    # Actual agent compositions are separate, unlabeled domain-transfer anchors.
    # These are original model outputs, NOT edits to the two MP3 recordings.
    for name in ['earlier', 'phrase']:
        path = ROOT / f'outputs/phrase_ensemble_v1/{name}_mix.mp3'
        if path.exists():
            score(path, f'agent_{name}', domain='agent_anchor', group=f'agent_{name}', kind='unlabelled_agent_composition',
                  listener_context='User disliked both; not used for fitting or an accuracy claim')
    candidates = ROOT / 'outputs/reward_original_candidates_v1'
    if (candidates / 'manifest.json').exists():
        manifest = json.loads((candidates / 'manifest.json').read_text())
        for row in manifest['candidates']:
            score(candidates / row['audio'], 'fresh_' + row['id'], domain='agent_anchor',
                  group='fresh_' + row['id'], kind='unlabelled_agent_composition',
                  seed=row['seed'], notes_sha256=row['notes_sha256'],
                  listener_context='Fresh original agent notes from the earlier checkpoint; no improvement claim')
    return rows


def audio_results(rows, domain, broken):
    selected = [r for r in rows if r['domain'] == domain]
    originals = {r['group']: r for r in selected if r['kind'] == 'original'}
    pairs = []
    for r in selected:
        if r['kind'] in broken:
            margin = originals[r['group']]['preference'] - r['preference']
            pairs.append(dict(group=r['group'], role=r.get('role', r.get('reference', '')), kind=r['kind'], margin=margin, win=margin > 0))
    result = dict(count=len(pairs), win_rate=float(np.mean([r['win'] for r in pairs])), pairs=pairs)
    result['by_kind'] = {kind: float(np.mean([r['win'] for r in pairs if r['kind'] == kind])) for kind in broken}
    result['by_role_or_reference'] = {role: float(np.mean([r['win'] for r in pairs if r['role'] == role])) for role in sorted({r['role'] for r in pairs})}
    result['by_role_kind'] = {role: {kind: float(np.mean([r['win'] for r in pairs if r['role'] == role and r['kind'] == kind])) for kind in broken} for role in result['by_role_or_reference']}
    result['sanity_passed'] = all(originals[r['group']]['preference'] > r['preference'] for r in selected if r['kind'] in ('silence', 'noise'))
    result['sanity_failures'] = [dict(group=r['group'], kind=r['kind'], original=originals[r['group']]['preference'], altered=r['preference']) for r in selected if r['kind'] in ('silence', 'noise') and originals[r['group']]['preference'] <= r['preference']]
    sigmoid = lambda x: float(torch.sigmoid(torch.tensor(x)))
    result['control_p90_probability_shift'] = {kind: float(np.percentile([abs(sigmoid(originals[r['group']]['preference'] - r['preference']) - .5) for r in selected if r['kind'] == kind], 90)) for kind in ['gain_minus_6db', 'mild_room', 'transpose_2_semitones'] if any(r['kind'] == kind for r in selected)}
    result['automated_passed'] = bool(result['win_rate'] >= .8 and result['sanity_passed'] and
        all(v >= 2/3 for kinds in result['by_role_kind'].values() for v in kinds.values()) and
        all(v <= .10 for v in result['control_p90_probability_shift'].values()))
    return result


def report(rows):
    from validate_reference_reward import BROKEN
    temporal = json.loads((OUT / 'coherence_test.json').read_text())
    reference = audio_results(rows, 'reference_diagnostic', BROKEN)
    midi = audio_results(rows, 'midi_test', list(KINDS))
    temporal_pass = temporal['win_rate'] >= .8 and all(v >= 2/3 for kinds in temporal['by_role_kind'].values() for v in kinds.values())
    write(OUT / 'results.json', dict(pretrained_reference_diagnostic=reference, pretrained_midi_test=midi,
                                    temporal_midi_test=temporal, policy_training_started=False))
    write(OUT / 'implementation_manifest.json', dict(files={name: sha(ROOT / name) for name in
          ['preference_reward.py', 'ensemble_coherence_reward.py', 'validate_music_reward_v2.py',
           'prepare_preference_reward.py', 'verify_preference_reward.py', 'joint_ensemble_rl.py']},
          protocol_sha256=sha(OUT / 'protocol.json'),
          note='No test-based reward-weight fitting. Test results remain in this version, including failures.'))
    acceptance = dict(automated_gate_passed=bool(reference['automated_passed'] and midi['automated_passed'] and temporal_pass),
        reference_gate_passed=reference['automated_passed'], midi_preference_gate_passed=midi['automated_passed'],
        temporal_gate_passed=bool(temporal_pass), human_preference_gate_passed=False, generated_domain_gate_passed=False,
        rl_training_allowed=False, policy_training_started=False,
        reason='Human ratings of fresh agent compositions and synthetic-label review are missing; automatic failures also block optimization.',
        protocol_sha256=sha(OUT / 'protocol.json'), coherence_checkpoint_sha256=sha(OUT / 'coherence.pt'),
        audio_model_provenance=json.loads((ROOT / 'models/tunejury_a1/provenance.json').read_text()))
    write(OUT / 'acceptance.json', acceptance)
    audit = json.loads((OUT / 'dataset_audit.json').read_text())
    text = f'''# Reward validation v2

No music-policy training. The reference audio is never used as generated music.

| Evaluator | Test | Unaltered beats disruption |
|---|---|---|
| Previous handcrafted reward | Known reference audio stress set | 55.6% |
| Previous static similarity baseline | Same known reference stress set | 75.0% (also failed its gate) |
| Frozen human-preference head, TuneJury A1 | Same known reference stress set | {reference['win_rate']:.1%} |
| Frozen human-preference head, TuneJury A1 | New composition-separated MIDI renders | {midi['win_rate']:.1%} |
| Trained temporal MIDI head | Composition-separated synthetic contrasts | {temporal['win_rate']:.1%} |

RL allowed: **No**. Automated combined gate passed: **{acceptance['automated_gate_passed']}**.

Pretrained audio sanity failures: reference={len(reference['sanity_failures'])},
MIDI={len(midi['sanity_failures'])}. A model ranking silence/noise above intact music
is unsuitable as a standalone RL reward, regardless of its average win rate.

The MIDI ranker learns to detect these particular synthetic disruptions, not your musical taste.
An altered part can sometimes be musically valid. A pass requires listener confirmation.
The human-preference audio model is pretrained on other listeners and generators; it can
disagree with you or reward SoundFont/production details. No weights or thresholds were
selected on the test set. All versions and failures are retained.

## Instrument results

```
Temporal head: {json.dumps(temporal['by_role_kind'], indent=2)}
Audio preference head: {json.dumps(midi['by_role_kind'], indent=2)}
```

Dataset counts: {audit['counts']}. Complete trios: {audit['full_trios']}.
The small held-out trio coverage prevents a strong three-instrument cooperation claim.
Both reference style similarities are separate diagnostics. No combined style/quality
reward is enabled, and neither high similarity nor increased note counts imply good music.

## Next evidence needed

Fresh original agent compositions rated for coherence, development, ensemble fit and enjoyment.
Fit/calibrate a small preference head on those ratings, retain composition-disjoint tests,
and include reward-hacking probes (constant notes, loops, silence, density, timing and tuning).
Only then enable conservative, KL-anchored joint RL. Render all candidates identically.
'''
    (OUT / 'REPORT.md').write_text(text, encoding='utf-8')
    # Evaluation page clearly separates diagnostics from actual original outputs.
    import html
    agent = [r for r in rows if r['domain'] == 'agent_anchor' and r['id'].startswith('fresh_')]
    cards = []
    for r in agent:
        cards.append(f'<section><h2>Original agent composition: {html.escape(r["id"])}</h2><p>New seed from the earlier checkpoint; no new policy training. This is not the reference MP3.</p><audio controls src="{html.escape(Path(r["audio"]).as_uri())}"></audio></section>')
    # Serve local files through a small copied anchor, keeping the page portable.
    import shutil
    for r in agent:
        destination = OUT / 'audio' / (r['id'] + Path(r['audio']).suffix)
        shutil.copyfile(r['audio'], destination)
    cards = [card.replace(Path(r['audio']).as_uri(), 'audio/' + r['id'] + Path(r['audio']).suffix) for card, r in zip(cards, agent)]
    (OUT / 'listen.html').write_text('<!doctype html><meta charset="utf-8"><title>Reward validation v2</title><style>body{font:18px system-ui;background:#101724;color:#e8edf4;max-width:950px;margin:40px auto;padding:20px}section{background:#1c2b40;padding:20px;margin:20px 0;border-radius:14px}a{color:#88cfff}audio{width:100%}pre{white-space:pre-wrap}</style><h1>Reward validation — no new policy training</h1><p>The cut-up reference clips were tests, not new music. Our goal remains original agent compositions.</p><p>Reference stress-test wins: '+f'{reference["win_rate"]:.1%}. MIDI preference wins: {midi["win_rate"]:.1%}. Temporal detection wins: {temporal["win_rate"]:.1%}.</p><p>RL remains blocked pending validation. These numbers are not music-quality ratings.</p><p><a href="REPORT.md">Full report</a> · <a href="results.json">Every test result</a> · <a href="RESEARCH.md">Research and method</a></p>'+''.join(cards)+'<h2>Structural tests</h2><p>Below are diagnostic disruptions of dataset compositions. They are not newly generated songs or improvements.</p>'+''.join(f'<section><p>Test composition {i+1}, unchanged</p><audio controls src="audio/composition_{i:02}_original.flac"></audio></section>' for i in range(min(3, audit['counts']['test']))), encoding='utf-8')
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbolic-only', action='store_true')
    parser.add_argument('--wait-for-render-batch', action='store_true', help='Wait for an already running independent render batch before scoring MIDI files')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audio').mkdir(exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != PROTOCOL:
        raise ValueError('Existing protocol differs; use a new version, never retune a held-out run')
    if not path.exists():
        write(path, PROTOCOL)
    if not (OUT / 'acceptance.json').exists():
        write(OUT / 'acceptance.json', dict(automated_gate_passed=False,
              human_preference_gate_passed=False, generated_domain_gate_passed=False,
              rl_training_allowed=False, policy_training_started=False, status='validation in progress'))
    data = prepare_dataset()
    print('Dataset:', {s: len(v) for s, v in data.items()}, flush=True)
    train_temporal(data)
    if not args.symbolic_only:
        rows = score_audio(data, args.wait_for_render_batch)
        report(rows)


if __name__ == '__main__':
    main()
