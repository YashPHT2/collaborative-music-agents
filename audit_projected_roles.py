"""Projection survival and listening inspection; no trainable cache replacement.

Primary projection reproduces the existing offline pipeline: 30-tick onsets,
60-tick durations up to 3840 ticks, range/velocity limits, onset-only crops,
same-pitch cleanup, and the existing playable_projection under offline limits.
Raw listening clips preserve incoming sustains via a labelled boundary retrigger.
"""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import mido
import numpy as np
import soundfile as sf

import audit_functional_roles as a

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'outputs/functional_role_audit_v1'
OUT = ROOT / 'outputs/projected_role_audit_v1'
ROLES = a.ROLES
END = 7680
PROTOCOL = dict(version=1, bars=4, bpm=100, primary='faithful_offline_onset_only_projection',
    duration_cap_ticks=3840, onset_grid_ticks=30, duration_grid_ticks=60,
    existing_legacy_eligibility='At least 8 note events per role; diagnostic comparison only',
    alternative_eligibility='All roles pass their original exact/transfer qualification after projection; no minimum-note rule',
    input_windows='All 353 transferred and 90 exact qualified full-trio windows from functional_role_audit_v1',
    incoming_sustains='Existing onset-only training crops omit them; log this loss explicitly. Raw audio retriggers them at the crop boundary only for inspection.',
    source_anomalies='Recount per source track from original cached MIDI; unmatched/unfinished/zero-duration events omitted, never filled in',
    gate_recheck='Same original-source program and behavioral thresholds, but projected durations/pitches/onsets; no gate changes',
    dedup='Role-aware exact and globally transposition-normalized aligned four-bar fingerprints; normalized fingerprint ignores velocity',
    split_policy='Preserve input split; exclude whole lower-priority groups on cross-split overlap, test > validation > train. Then keep one representative per duplicate passage.',
    inspection=dict(split='train only', examples_per_bucket=1,
        buckets=['all_exact', 'high_retention_transfer', 'borderline_transfer'],
        high='All role note and coverage ratios >= .90 and passes without legacy threshold',
        borderline='At least one role note retention .65..85; other retention may be worse and is shown',
        fallback='If a transfer bucket is empty, label unavailable; do not replace it with an attractive clip',
        sampling='First deterministic SHA256 rank, distinct source group where possible; no listening or reward selection',
        audio='Same target GM patches, renderer, tempo, release and constant-gain LUFS matching per raw/projected stem pair'),
    training_started=False, cache_replacement=False,
    limitations=['Raw transfer clips are dataset excerpts reassigned to target timbres, not agent compositions.',
                 'Inspection samples are deliberately selected diagnostics, not preference evaluation.',
                 'Projection survival does not establish musical quality.',
                 'Ten validation compositions per role is checked for this trio-only pool, not the untouched broader corpus.'])


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def selected_windows():
    with gzip.open(INPUT / 'windows.jsonl.gz', 'rt', encoding='utf-8') as f:
        return [r for line in f if (r := json.loads(line))['category'] in ('exact_full_trio', 'transferred_full_trio')]


def source_event_anomalies(blob):
    midi = mido.MidiFile(file=io.BytesIO(blob))
    events, active = [], defaultdict(deque)
    for track, messages in enumerate(midi.tracks):
        tick = 0
        for index, msg in enumerate(messages):
            tick += msg.time
            events.append((tick, track, index, msg))
    issues = []
    for tick, track, _, msg in sorted(events, key=lambda row: row[:3]):
        if getattr(msg, 'channel', None) == 9: continue
        if msg.type == 'note_on' and msg.velocity:
            active[msg.channel, msg.note].append((tick, track))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and not msg.velocity):
            queue = active[msg.channel, msg.note]
            if not queue:
                issues.append(dict(kind='unmatched_note_off', track=track, tick=tick * 480 / midi.ticks_per_beat))
            else:
                start, onset_track = queue.popleft()
                if tick <= start:
                    issues.append(dict(kind='zero_or_negative_duration', track=onset_track, tick=start * 480 / midi.ticks_per_beat))
    for queue in active.values():
        for start, track in queue:
            issues.append(dict(kind='unfinished_note_on', track=track, tick=start * 480 / midi.ticks_per_beat))
    return issues


def anomaly_cache(windows, directory):
    path = directory / 'source_anomalies.json'
    if path.exists(): return read_json(path)
    needed = {r['song'] for r in windows}
    result = {}
    for source, archive in [('pdmx', ROOT / 'assets/composition_v1/mid.tar.gz'),
                            ('slakh', ROOT / 'assets/trio_ensemble/slakh_midi.tar.gz')]:
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith('.mid'): continue
                song = 'pdmx:' + Path(member.name).stem if source == 'pdmx' else 'slakh:' + next((p for p in Path(member.name).parts if p.startswith('Track')), '')
                if song in needed:
                    result[song] = source_event_anomalies(tar.extractfile(member).read())
    if set(result) != needed: raise ValueError('Missing anomaly source')
    a.write(path, result)
    return result


def source_parts(window):
    cache = INPUT / 'source_tracks' / (hashlib.sha256(window['song'].encode()).hexdigest() + '.json.gz')
    with gzip.open(cache, 'rt', encoding='utf-8') as f: parsed = json.load(f)
    if parsed['midi_sha256'] != window['midi_sha256']: raise ValueError('Source MIDI identity changed')
    result = {}
    for role, assignment in window['assignment'].items():
        result[role] = [dict(n, source_note_id=i) for i, n in enumerate(parsed['notes']) if
            n['source_track'] == assignment['source_track'] and n['source_channel'] == assignment['source_channel'] and n['program'] == assignment['source_program']]
    return result


def to_metrics(notes, role, program):
    return [dict(start=n[1], duration=n[2], pitch=n[4], velocity=n[5], program=program) for n in notes if n[3] == ROLES.index(role)]


def metric(notes):
    if not notes:
        return dict(note_events=0, onset_note_events=0, onset_groups=0, onset_groups_per_bar=0.,
                    duty=0., polyphonic_sounding_fraction=0., max_polyphony=0, mean_duration_beats=0.,
                    median_pitch=None, contour_transitions=0, small_interval_fraction=None,
                    ioi_cv=None, median_ioi_beats=None, pitch_diversity=0)
    return a.statistics(notes, 0, END)


def raw_window(parts, start):
    result = {}
    for role, notes in parts.items():
        # Keep original relative onset for metrics; interval_coverage clips it.
        result[role] = [dict(n, start=n['start'] - start,
                            duration=min(n['start'] + n['duration'], start + END) - n['start'])
                        for n in notes if n['start'] < start + END and n['start'] + n['duration'] > start]
    return result


def canonical_window(parts, start):
    projected, provenance, changes = [], {}, {r: Counter() for r in ROLES}
    for role, source in parts.items():
        channel = ROLES.index(role)
        lo, hi = a.RANGES[role]
        unique = {}
        for n in source:
            if not lo <= n['pitch'] <= hi:
                if n['start'] < start + END and n['start'] + n['duration'] > start:
                    changes[role]['range_discarded'] += 1
                continue
            tick = max(0, round(n['start'] / 30) * 30)
            duration = max(60, min(3840, round(n['duration'] / 60) * 60))
            velocity = max(32, min(96, round(n['velocity'] / 4) * 4))
            p = ['note', tick, duration, channel, n['pitch'], velocity]
            key = tick, n['pitch']
            if key not in unique or duration > unique[key][0][2]:
                unique[key] = p, n
        seq = sorted(unique.values(), key=lambda row: (row[0][1], row[0][4]))
        pitches = defaultdict(list)
        for p, n in seq: pitches[p[4]].append((p, n))
        for voice in pitches.values():
            for (p, n), (next_note, _) in zip(voice, voice[1:]):
                old = p[2]
                p[2] = min(p[2], (next_note[1] - p[1]) // 60 * 60)
                if p[2] < old and start <= p[1] < start + END:
                    changes[role]['same_pitch_overlap_shortened'] += 1
        # Count duplicate discarded sources attributable to this crop.
        represented = {n['source_note_id'] for p, n in seq}
        for n in source:
            if start <= n['start'] < start + END and lo <= n['pitch'] <= hi and n['source_note_id'] not in represented:
                changes[role]['duplicate_onset_pitch_discarded'] += 1
        for p, n in seq:
            if n['start'] < start < n['start'] + n['duration']:
                if p[1] < start < p[1] + p[2]:
                    changes[role]['incoming_sustain_omitted_by_onset_crop'] += 1
                elif p[1] == start:
                    changes[role]['incoming_sustain_quantized_to_boundary'] += 1
                else:
                    changes[role]['incoming_sustain_lost_before_crop_by_duration_cap_or_cleanup'] += 1
            if not start <= p[1] < start + END: continue
            p = copy.deepcopy(p)
            p[1] -= start
            p[2] = min(p[2], END - p[1]) // 60 * 60
            if p[2] < 60:
                changes[role]['duration_below_60_discarded'] += 1
                continue
            if p[1] + start != n['start']: changes[role]['onset_quantized'] += 1
            if p[2] != n['duration']: changes[role]['duration_quantized_capped_or_boundary_trimmed'] += 1
            if n['duration'] > 3840: changes[role]['source_duration_above_8_beat_cap'] += 1
            if p[5] != n['velocity']: changes[role]['velocity_quantized'] += 1
            projected.append(p)
            provenance[channel, p[1], p[4]] = n['source_note_id']
    return sorted(projected, key=lambda n: (n[1], n[3], n[4])), provenance, changes


def traced_projection(notes):
    import note_trio as nt
    import offline_ensemble as oe
    import supervised_ensemble as se
    changes = {r: Counter() for r in ROLES}
    with oe.limits():
        expected = se.playable_projection(notes)
        actual = []
        for role, settings in nt.ROLES.items():
            own, counts = [], Counter()
            for n in sorted([copy.deepcopy(n) for n in notes if n[3] == settings['channel']], key=lambda n: (n[1], -n[4] if role == 'strings' else n[4])):
                if counts[n[1] // 480] >= settings['cap']:
                    changes[role]['event_cap_discarded'] += 1
                    continue
                active = [p for p in own if p[1] <= n[1] < p[1] + p[2]]
                if role == 'cello' and active:
                    previous = active[-1]
                    duration = (n[1] - previous[1]) // 60 * 60
                    if duration < 60:
                        changes[role]['simultaneous_cello_voice_discarded'] += 1
                        continue
                    old = previous[2]
                    previous[2] = min(previous[2], duration)
                    if previous[2] < old: changes[role]['cello_overlap_shortened'] += 1
                    active = []
                if len(active) >= settings['polyphony']:
                    changes[role]['polyphony_discarded'] += 1
                    continue
                if n[4] in {p[4] for p in active}:
                    changes[role]['repeated_pitch_overlap_discarded'] += 1
                    continue
                own.append(n); counts[n[1] // 480] += 1
            actual.extend(own)
        actual.sort(key=lambda n: (n[1], n[3], n[4]))
        if actual != expected: raise ValueError('Instrumented projection differs from production implementation')
        ledger = []
        for beat in range(16):
            actions = {r: [n for n in actual if n[3] == i and n[1] // 480 == beat] for i, r in enumerate(ROLES)}
            ledger = nt.commit(ledger, actions, beat, nt.Brief(4))
        if ledger != actual: raise ValueError('Commit changed projected notes')
    return actual, changes


def ratio(after, before):
    return float(after / before) if before > 0 else None


def fingerprint(notes, transpose=False):
    ordered = sorted(notes, key=lambda n: (n[1], n[3], n[4]))
    anchor = ordered[0][4] if ordered and transpose else 0
    rows = [(n[3], n[1], n[2], n[4] - anchor, *([] if transpose else [n[5]])) for n in ordered]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def inspect_window(window, anomalies):
    parts = source_parts(window)
    raw = raw_window(parts, window['start_tick'])
    canonical, source_ids, changes = canonical_window(parts, window['start_tick'])
    projected, pruning = traced_projection(canonical)
    role_results = {}
    for role, assignment in window['assignment'].items():
        own = to_metrics(projected, role, assignment['source_program'])
        before, after = metric(raw[role]), metric(own)
        provenance, reasons, _ = a.qualify(role, own, 0, END) if own else (None, ['no_projected_notes'], None)
        if provenance != assignment['provenance'] and not reasons:
            raise ValueError('Projection unexpectedly changed provenance class')
        local_issues = Counter(issue['kind'] for issue in anomalies if issue['track'] == assignment['source_track'] and window['start_tick'] <= issue['tick'] < window['end_tick'])
        all_track_issues = Counter(issue['kind'] for issue in anomalies if issue['track'] == assignment['source_track'])
        changes[role].update(pruning[role])
        before_poly = before['duty'] * before['polyphonic_sounding_fraction']
        after_poly = after['duty'] * after['polyphonic_sounding_fraction']
        role_results[role] = dict(provenance=assignment['provenance'], source_program=assignment['source_program'],
            source_track=assignment['source_track'], source_channel=assignment['source_channel'],
            before=before, after=after,
            retention=dict(notes=ratio(after['note_events'], before['note_events']), sounding_coverage=ratio(after['duty'], before['duty']),
                           polyphonic_time=ratio(after_poly, before_poly)),
            gate_passed=not reasons, gate_rejection_reasons=reasons,
            legacy_eight_note_passed=after['note_events'] >= 8,
            changes=changes[role], anomaly_events_in_window=local_issues,
            anomaly_events_entire_source_track=all_track_issues)
    strings, cello = role_results['strings'], role_results['cello']
    register_ok = not ('transfer' in (strings['provenance'], cello['provenance']) and
        (strings['after']['median_pitch'] is None or cello['after']['median_pitch'] is None or strings['after']['median_pitch'] <= cello['after']['median_pitch']))
    source_key = window['song'] + ':' + str(window['start_tick'])
    return dict(id=hashlib.sha256(source_key.encode()).hexdigest()[:16], song=window['song'], group=window['group'],
        split=window['split'], category=window['category'], start_tick=window['start_tick'], end_tick=window['end_tick'],
        midi_sha256=window['midi_sha256'], assignment=window['assignment'], roles=role_results,
        passes_without_legacy_threshold=bool(register_ok and all(r['gate_passed'] for r in role_results.values())),
        passes_with_legacy_threshold=bool(register_ok and all(r['gate_passed'] and r['legacy_eight_note_passed'] for r in role_results.values())),
        projected_register_constraint_passed=register_ok,
        exact_fingerprint=fingerprint(projected), transposition_fingerprint=fingerprint(projected, True),
        projected_notes=projected,
        source_note_ids=[dict(role=ROLES[n[3]], projected_note=n, source_note_id=source_ids[n[3], n[1], n[4]]) for n in projected])


def deduplicate(rows, eligibility):
    eligible = [r for r in rows if r[eligibility]]
    priority = {'test': 0, 'validation': 1, 'train': 2}
    index = defaultdict(list)
    for row in eligible:
        for key in ('exact_fingerprint', 'transposition_fingerprint'):
            index[key, row[key]].append(row)
    excluded = {}
    for group in index.values():
        highest = min(priority[r['split']] for r in group)
        for row in group:
            if priority[row['split']] > highest:
                excluded[row['split'], row['group']] = 'cross_split_projected_passage_overlap'
    kept, seen, decisions = [], set(), []
    for row in sorted(eligible, key=lambda r: (priority[r['split']], r['id'])):
        reason = excluded.get((row['split'], row['group']))
        keys = [(key, row[key]) for key in ('exact_fingerprint', 'transposition_fingerprint')]
        if not reason and any(key in seen for key in keys): reason = 'duplicate_projected_passage'
        if reason:
            decisions.append(dict(id=row['id'], split=row['split'], group=row['group'], reason=reason))
        else:
            kept.append(row)
            seen.update(keys)
    return kept, dict(excluded_composition_groups=[dict(split=s, group=g, reason=reason) for (s, g), reason in sorted(excluded.items())],
                      excluded_windows=decisions)


def summarize(rows, directory):
    summary = dict(input_windows=len(rows), transferred_input_windows=sum(r['category'] == 'transferred_full_trio' for r in rows),
                   exact_input_windows=sum(r['category'] == 'exact_full_trio' for r in rows), splits={}, training_started=False)
    for name, eligibility in [('without_legacy', 'passes_without_legacy_threshold'), ('with_legacy', 'passes_with_legacy_threshold')]:
        kept, log = deduplicate(rows, eligibility)
        a.write(directory / f'dedup_{name}.json', log)
        a.write(directory / f'eligible_{name}_ids.json', [r['id'] for r in kept])
        summary[name] = {}
        for split in ('train', 'validation', 'test'):
            before = [r for r in rows if r['split'] == split and r[eligibility]]
            selected = [r for r in kept if r['split'] == split]
            summary[name][split] = dict(before_dedup_windows=len(before), after_dedup_windows=len(selected),
                after_dedup_groups=len({r['group'] for r in selected}),
                exact_windows=sum(r['category'] == 'exact_full_trio' for r in selected),
                transferred_windows=sum(r['category'] == 'transferred_full_trio' for r in selected))
        validation_groups = len({r['group'] for r in kept if r['split'] == 'validation'})
        summary[name]['validation_compositions_per_role_in_this_trio_only_pool'] = {r: validation_groups for r in ROLES}
        summary[name]['ten_validation_composition_gate_passed_for_this_pool'] = validation_groups >= 10
    for split in ('train', 'validation', 'test'):
        current = [r for r in rows if r['split'] == split]
        summary['splits'][split] = dict(input_windows=len(current),
            per_role_surviving_groups_before_trio_filter={role: len({r['group'] for r in current if r['roles'][role]['gate_passed']}) for role in ROLES},
            recheck_failures={role: dict(Counter(reason for r in current for reason in r['roles'][role]['gate_rejection_reasons'])) for role in ROLES})
    return summary


def inspection_selection(rows):
    buckets = defaultdict(list)
    for row in rows:
        if row['split'] != 'train': continue
        ratios = [r['retention'] for r in row['roles'].values()]
        if row['category'] == 'exact_full_trio': buckets['all_exact'].append(row)
        elif row['passes_without_legacy_threshold'] and all(r['notes'] is not None and r['notes'] >= .9 and r['sounding_coverage'] is not None and r['sounding_coverage'] >= .9 for r in ratios):
            buckets['high_retention_transfer'].append(row)
        elif any(r['notes'] is not None and .65 <= r['notes'] <= .85 for r in ratios):
            buckets['borderline_transfer'].append(row)
    chosen, used = [], set()
    for name in PROTOCOL['inspection']['buckets']:
        candidates = sorted(buckets[name], key=lambda r: r['id'])
        row = next((r for r in candidates if r['group'] not in used), candidates[0] if candidates else None)
        if row:
            used.add(row['group']); chosen.append(dict(bucket=name, row_id=row['id']))
    return dict(bucket_population={name: len(buckets[name]) for name in PROTOCOL['inspection']['buckets']}, selected=chosen,
                unavailable=[name for name in PROTOCOL['inspection']['buckets'] if not buckets[name]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--render-inspection', action='store_true')
    args = parser.parse_args()
    directory = args.output.resolve(); directory.mkdir(parents=True, exist_ok=True)
    protocol_path = directory / 'protocol.json'
    if protocol_path.exists() and read_json(protocol_path) != PROTOCOL: raise ValueError('Changed protocol requires new version')
    if not protocol_path.exists(): a.write(protocol_path, PROTOCOL)
    input_hash = a.digest(INPUT / 'windows.jsonl.gz')
    corpus_hash = a.digest(ROOT / 'assets/composition_v1/corpus.json.gz')
    if (directory / 'windows.json').exists():
        rows = read_json(directory / 'windows.json')
    else:
        windows = selected_windows()
        anomalies = anomaly_cache(windows, directory)
        rows = []
        for index, window in enumerate(windows):
            rows.append(inspect_window(window, anomalies[window['song']]))
            if (index + 1) % 50 == 0: print(f'Projection audited {index+1}/{len(windows)} windows; no training', flush=True)
        a.write(directory / 'windows.json', rows)
    summary = summarize(rows, directory)
    summary['input_audit_unchanged'] = a.digest(INPUT / 'windows.jsonl.gz') == input_hash
    summary['original_corpus_unchanged'] = a.digest(ROOT / 'assets/composition_v1/corpus.json.gz') == corpus_hash
    summary['production_projection_agreement'] = True
    summary['protocol_sha256'] = a.digest(protocol_path)
    a.write(directory / 'summary.json', summary)
    selection = inspection_selection(rows)
    a.write(directory / 'inspection_selection.json', selection)
    write_report(rows, summary, directory)
    print((directory / 'REPORT.md').read_text(encoding='utf-8'), flush=True)
    if args.render_inspection: render_inspection(rows, selection, directory)


def write_report(rows, summary, directory):
    text = ['# Playable projection survival', '', 'No training, reward fitting or replacement of the original corpus.',
        f'Audited {len(rows)} full-trio windows: {summary["transferred_input_windows"]} transfers and {summary["exact_input_windows"]} exact.',
        'Primary projection exactly matches the existing offline rules, including onset-only cropping. Raw incoming sustains are counted before projection.', '',
        '| Condition | Train windows / groups | Validation windows / groups | Test windows / groups |', '|---|---:|---:|---:|']
    for label, name in [('Post-gate, no eight-note threshold, deduplicated', 'without_legacy'), ('Post-gate plus eight-note threshold, deduplicated', 'with_legacy')]:
        text.append('| ' + label + ' | ' + ' | '.join(str(summary[name][s]['after_dedup_windows']) + ' / ' + str(summary[name][s]['after_dedup_groups']) for s in ('train', 'validation', 'test')) + ' |')
    text += ['', 'The validation gate is ten distinct compositions per target role. This trio-only subset does not certify the untouched broader corpus.',
        'Gate without legacy threshold: ' + str(summary['without_legacy']['ten_validation_composition_gate_passed_for_this_pool']),
        '', '## Main changes and failed rechecks', '']
    for role in ROLES:
        changes = Counter(); failures = Counter()
        for row in rows:
            changes.update(row['roles'][role]['changes']); failures.update(row['roles'][role]['gate_rejection_reasons'])
        text.extend([f'### {role}', '', f'Changes across windows: {dict(changes)}', '', f'Failed gates: {dict(failures)}', ''])
    text += ['## Interpretation', '',
        'The prior raw audit already omitted unfinished and zero-duration notes. They did not inflate its sounding coverage. No missing note-offs are guessed or repaired into new notes.',
        'The offline duration cap is 3840 ticks (eight beats), not four beats. Recovered same-pitch overlaps may still be shortened by canonicalization.',
        'A null polyphonic-time retention ratio means the source had no polyphonic time; it is not zero retention.',
        'Raw source and projected targets use different timing precision. Quantization, duration limits and boundary-sustain omission are explicit data transformations, not new agent decisions.',
        'Deduplication preserves role identity and aligned onset positions. A single global transposition is normalized; leading silence and relative instrumental registers are preserved.',
        'Whole lower-priority composition groups with detected cross-split overlaps are excluded from this candidate pool. No source group is moved to another split.',
        'Qualified windows are not automatically musically good. Inspection examples are training-only diagnostic data rendered on target patches, not agent-generated compositions.',
        'PPO remains gated; no DPO is started. Transfer-loss weights are not applied.']
    (directory / 'REPORT.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def render_notes(raw_parts, role=None):
    """Integer-tick rendering view, including explicitly retriggered carry-ins."""
    notes = []
    for name, source in raw_parts.items():
        if role and name != role: continue
        for n in source:
            left = max(0, round(n['start']))
            right = min(END, round(n['start'] + n['duration']))
            if right > left:
                notes.append(['note', left, right-left, ROLES.index(name), n['pitch'], n['velocity']])
    return sorted(notes, key=lambda n: (n[1], n[3], n[4]))


def inspection_midi(path, notes):
    """Preserve same-pitch overlapping source notes using extra render channels.

    This is renderer plumbing, not new notes or training-target modification.
    A round-trip maps channels back to the three logical roles and checks notes.
    """
    pools = {0: [0,3,6,10,13], 1: [1,4,7,11,14], 2: [2,5,8,12,15]}
    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack(); midi.tracks.append(meta)
    meta.append(mido.MetaMessage('set_tempo', tempo=600000))
    meta.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
    meta.append(mido.MetaMessage('end_of_track', time=END))
    channel_roles = {}
    for role, patch, volume in [(0,19,127), (1,48,64), (2,42,80)]:
        track = mido.MidiTrack(); midi.tracks.append(track)
        events, occupied = [], {}
        for n in [n for n in notes if n[3] == role]:
            channel = next((c for c in pools[role] if occupied.get((c,n[4]), -1) <= n[1]), None)
            if channel is None: raise ValueError('More than five overlapping identical source pitches; do not silently prune raw audio')
            occupied[channel,n[4]] = n[1]+n[2]
            channel_roles[channel] = role
            events.extend([(n[1],1,mido.Message('note_on',channel=channel,note=n[4],velocity=n[5])),
                           (n[1]+n[2],0,mido.Message('note_off',channel=channel,note=n[4],velocity=0))])
        for channel in pools[role]:
            track.append(mido.Message('program_change',channel=channel,program=patch))
            track.append(mido.Message('control_change',channel=channel,control=7,value=volume))
        last = 0
        for tick, _, message in sorted(events, key=lambda e:(e[0],e[1])):
            message.time=tick-last; track.append(message); last=tick
    midi.save(path)
    tick, active, actual = 0, {}, []
    for msg in mido.merge_tracks(mido.MidiFile(path).tracks):
        tick += msg.time
        if msg.type == 'note_on' and msg.velocity:
            if (msg.channel,msg.note) in active: raise ValueError('Ambiguous render-channel overlap')
            active[msg.channel,msg.note] = tick,msg.velocity
        elif msg.type == 'note_off' or (msg.type == 'note_on' and not msg.velocity):
            start, velocity = active.pop((msg.channel,msg.note))
            actual.append(['note',start,tick-start,channel_roles[msg.channel],msg.note,velocity])
    if active or sorted(actual) != sorted(notes): raise ValueError('Inspection MIDI round-trip failed')
    return dict(notes=len(notes), same_pitch_overlaps_preserved=True, roundtrip_exact=True)


def loudness(path):
    run = subprocess.run(['ffmpeg','-v','info','-i',str(path),'-af',
        'loudnorm=I=-18:TP=-1.5:LRA=50:print_format=json','-f','null','-'],capture_output=True,text=True,check=True)
    obj, _ = json.JSONDecoder().raw_decode(run.stderr[run.stderr.rfind('{'):])
    integrated, peak = float(obj['input_i']), float(obj['input_tp'])
    return integrated, peak


def render_pair(job):
    row, bucket, stem, directory = job
    audio = directory / 'audio'; audio.mkdir(exist_ok=True)
    check_path = audio / f'{row["id"]}_{stem}_verification.json'
    if check_path.exists(): return read_json(check_path)
    raw = raw_window(source_parts(row), row['start_tick'])
    variants = dict(raw=render_notes(raw, None if stem == 'mix' else stem),
                    projected=[n for n in row['projected_notes'] if stem == 'mix' or n[3] == ROLES.index(stem)])
    fluid = next((ROOT/'tools/fluidsynth').rglob('fluidsynth.exe'))
    measures, midi_checks = {}, {}
    for variant, notes in variants.items():
        name = audio / f'{row["id"]}_{variant}_{stem}'
        midi = name.with_suffix('.mid')
        midi_checks[variant] = inspection_midi(midi, notes)
        wave = name.with_name(name.name+'_render.wav')
        subprocess.run([str(fluid),'-ni','-F',str(wave),'-r','44100','-g','.6',str(ROOT/'assets/soundfont.sf2'),str(midi)],capture_output=True,check=True)
        trimmed = name.with_name(name.name+'_trim.wav')
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(wave),'-af','apad,atrim=duration=12,afade=t=out:st=11.5:d=0.5','-c:a','pcm_f32le',str(trimmed)],check=True)
        data, _ = sf.read(trimmed)
        if not np.isfinite(data).all(): raise ValueError('Nonfinite render')
        silent = not notes or float(np.max(np.abs(data))) < 1e-7
        integrated, peak = loudness(trimmed) if not silent else (None,None)
        measures[variant] = dict(silent=silent, integrated=integrated, peak=peak, temporary=str(trimmed))
        wave.unlink()
    desired = -18 if stem == 'mix' else -22
    feasible = [desired] + [-1.5-(m['peak']-m['integrated']) for m in measures.values() if not m['silent']]
    common_target = min(feasible)
    final_checks = {}
    for variant, measure in measures.items():
        name = audio / f'{row["id"]}_{variant}_{stem}'
        gain = 0 if measure['silent'] else common_target-measure['integrated']
        final = name.with_suffix('.wav')
        subprocess.run(['ffmpeg','-y','-v','error','-i',measure['temporary'],'-af',f'volume={gain:.8f}dB','-c:a','pcm_s24le',str(final)],check=True)
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(final),'-q:a','2',str(name.with_suffix('.mp3'))],check=True)
        data, sr = sf.read(final)
        if not np.isfinite(data).all() or np.max(np.abs(data)) >= 1: raise ValueError('Invalid/clipped final render')
        final_checks[variant] = dict(silent=measure['silent'], constant_gain_db=gain,
            target_lufs=None if measure['silent'] else common_target,
            measured_output_lufs=loudness(final)[0] if not measure['silent'] else None,
            seconds=len(data)/sr, sample_peak=float(np.max(np.abs(data))), file=final.with_suffix('.mp3').name,
            midi=midi_checks[variant])
        Path(measure['temporary']).unlink()
    result = dict(id=row['id'], bucket=bucket, stem=stem, variants=final_checks, renderer='FluidSynth + unchanged assets/soundfont.sf2',
                  loudness_matching='Pair-specific shared LUFS target, constant gain only; silence explicitly excluded from loudness matching',
                  raw_boundary_note_retrigger_for_inspection=True)
    a.write(check_path,result)
    print(f'Inspection rendered {bucket}: {stem}',flush=True)
    return result


def render_inspection(rows, selection, directory):
    by_id = {r['id']:r for r in rows}
    jobs = [(by_id[item['row_id']],item['bucket'],stem,directory) for item in selection['selected'] for stem in ('mix',*ROLES)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        checks = list(pool.map(render_pair,jobs))
    a.write(directory/'render_verification.json',checks)
    write_inspection_page(by_id, selection, directory)


def write_inspection_page(by_id, selection, directory):
    import html
    items = []
    names = dict(all_exact='Exact instruments',high_retention_transfer='Transfer with high retention',borderline_transfer='Transfer with noticeable note loss')
    for item in selection['selected']:
        row = by_id[item['row_id']]
        stats = ''.join('<tr><td>'+role+'</td><td>'+r['provenance']+' · GM '+str(r['source_program'])+'</td><td>'+str(r['before']['note_events'])+' → '+str(r['after']['note_events'])+'</td><td>'+f'{r["before"]["duty"]:.0%} → {r["after"]["duty"]:.0%}'+'</td></tr>' for role,r in row['roles'].items())
        items.append(f'''<section data-id="{row['id']}"><h2>{names[item['bucket']]}</h2>
<p>Training-set excerpt · source {html.escape(row['song'])} · bars {row['start_tick']//1920+1}–{row['start_tick']//1920+4}.</p>
<table><thead><tr><th>Role</th><th>Source label</th><th>Notes: raw → projected</th><th>Sounding coverage</th></tr></thead><tbody>{stats}</tbody></table>
<label>Listen to <select class="stem"><option value="mix">Full trio</option><option>organ</option><option>strings</option><option>cello</option></select></label>
<div class="pair"><div><h3>Raw recovered</h3><p>Source notes on our target instrument sounds.</p><audio controls preload="metadata" data-variant="raw" src="audio/{row['id']}_raw_mix.mp3"></audio></div>
<div><h3>Playable projected</h3><p>The existing training projection.</p><audio controls preload="metadata" data-variant="projected" src="audio/{row['id']}_projected_mix.mp3"></audio></div></div>
<label>Does the raw transfer arrangement work? <select data-rating="raw_fit"><option value="">Choose</option><option value="good">Yes</option><option value="poor">No</option><option value="uncertain">Unsure</option></select></label>
<label>What did projection do? <select data-rating="projection_effect"><option value="">Choose</option><option value="worse">Made it worse</option><option value="same">About the same</option><option value="better">Made it better</option><option value="uncertain">Unsure</option></select></label>
<p>Post-projection behavior gates: {'pass' if row['passes_without_legacy_threshold'] else 'fail'}. Legacy eight-note condition: {'pass' if row['passes_with_legacy_threshold'] else 'fail'}.</p></section>''')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Inspect trio data projection</title>
<style>body{font:18px system-ui;line-height:1.5;background:#101724;color:#e9edf4;max-width:1100px;margin:30px auto;padding:20px}section{background:#1d2b40;padding:24px;border-radius:14px;margin:24px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}audio{width:100%}table{border-collapse:collapse;width:100%;font-size:15px}th,td{text-align:left;padding:8px;border-bottom:1px solid #43536c}label{display:block;margin:16px 0}select,button{font:inherit;padding:8px}a{color:#9fd6ff}@media(max-width:700px){.pair{grid-template-columns:1fr}table{font-size:12px}}</style>
<h1>Does projection damage our training music?</h1><p><strong>Dataset inspection — these are not new agent compositions.</strong> Three predetermined training excerpts help separate role-transfer problems from note-pruning problems.</p>
<p>Both versions use the same organ, strings and cello patches, tempo and renderer. Each pair has matched loudness using constant gain. Incoming source sustains are retriggered at the excerpt boundary for the raw preview; the existing onset-only projection may omit them.</p>
<p><a href="REPORT.md">Survival report</a> · <a href="inspection_selection.json">Selection protocol</a> · <a href="summary.json">Counts and gate results</a></p>'''+''.join(items)+'''
<button id="download">Download inspection notes</button><p id="message" role="status"></p>
<script>
const storageKey='projected-role-inspection-v1';let ratings={};try{ratings=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch{}
for(const section of document.querySelectorAll('section')){
 const id=section.dataset.id;
 section.querySelector('.stem').onchange=e=>{for(const audio of section.querySelectorAll('audio')){audio.pause();audio.src=`audio/${id}_${audio.dataset.variant}_${e.target.value}.mp3`;audio.load()}};
 for(const select of section.querySelectorAll('[data-rating]')){select.value=ratings[id]?.[select.dataset.rating]||'';select.onchange=()=>{(ratings[id]??={})[select.dataset.rating]=select.value;try{localStorage.setItem(storageKey,JSON.stringify(ratings))}catch{}}}
}
for(const audio of document.querySelectorAll('audio'))audio.addEventListener('play',()=>{for(const other of document.querySelectorAll('audio'))if(other!==audio)other.pause()});
document.getElementById('download').onclick=async()=>{try{const summary=await(await fetch('summary.json')).json();const selection=await(await fetch('inspection_selection.json')).json();const blob=new Blob([JSON.stringify({schema:'projected-role-inspection-v1',protocol_sha256:summary.protocol_sha256,selection,ratings},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='projection-inspection.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);document.getElementById('message').textContent='Inspection notes downloaded.'}catch(e){document.getElementById('message').textContent='Download failed: '+e.message}};
</script></html>'''
    (directory/'listen.html').write_text(page,encoding='utf-8')


if __name__ == '__main__':
    main()
