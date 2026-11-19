"""Read-only functional-role coverage audit. No training or corpus relabelling.

Recover track/channel provenance from the original cached MIDI archives: the
existing corpus raw notes do not retain source track identity. Exact instruments
and heuristic role transfers stay distinct. Thresholds describe a candidate data
selection protocol, never minimum-activity rules for generated music.
"""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

import mido
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/functional_role_audit_v1'
ROLES = ('organ', 'strings', 'cello')
EXACT = {'organ': set(range(16, 21)), 'strings': {40, 41, 48, 49}, 'cello': {42}}
TRANSFER = {'organ': {0, 1, 2, 3, 6, 21, 22, 23, 89, 90, 95},
            'strings': {50, 51, 68, 71, 73}, 'cello': {32, 35, 43, 70}}
RANGES = {'organ': (48, 88), 'strings': (55, 91), 'cello': (36, 67)}
PROTOCOL = dict(version=1, bars=4, ticks_per_beat=480, window_hop_bars=4,
    source='The 1806 existing composition_v1 records; no new downloads or split changes',
    identity='Source track + channel + program; assignments exclusive by physical source track',
    format_zero='One physical source track may fill at most one role, even if multitimbral; report this conservative limitation',
    exact_programs={r: sorted(v) for r, v in EXACT.items()},
    transfer_programs={r: sorted(v) for r, v in TRANSFER.items()},
    program_numbering='Zero-based GM; patch names alone do not establish musical function',
    common_transfer=dict(minimum_target_range_fraction=.80),
    organ=dict(minimum_duty=.50, minimum_mean_duration_beats=1.,
               minimum_polyphonic_sounding_fraction=.15, minimum_polyphony=2,
               maximum_onset_groups_per_bar=6., pulse_minimum_onset_groups=4,
               pulse_maximum_ioi_cv=.25, pulse_median_ioi_beats=[.5, 4.],
               median_pitch_range=[48, 76]),
    strings=dict(minimum_duty=.25, minimum_mean_duration_beats=.5,
                 minimum_median_pitch=58, minimum_contour_transitions=2,
                 maximum_small_interval_semitones=4, minimum_small_interval_fraction=.60,
                 sustained_alternative_mean_duration_beats=2., sustained_alternative_duty=.50),
    cello=dict(minimum_mean_duration_beats=.75, median_pitch_range=[36, 62],
               maximum_polyphonic_sounding_fraction=.10,
               prune_lowest_voice=False),
    joint_constraint='When either mapped strings/cello part is transferred, strings median must exceed cello median',
    assignment='Maximum total eligibility weight; exact=2, transfer=1, plus .01*duty. Missing=0. Deterministic ties.',
    exact_gates='Known patch plus at least one sounding in-range note; no transfer behavior gates',
    suggested_future_loss_weights={'exact': 1., 'transfer': .35, 'missing': 0.},
    loss_weights_applied=False, trainable_dataset_created=False,
    limits=['Heuristic weak labels, not verified orchestration or taste.',
            'Exact source records and qualified simultaneous windows are different counts.',
            'No threshold changes selected after seeing test counts.',
            'No per-role minimum-note rule is applied to generation.',
            'No lowest-voice pruning or pitch transposition; canonical playable projection is deferred.',
            'Track/channel/program segmentation can split instruments with program changes; physical track exclusivity remains.',
            'Comparing composer IDs or source records does not establish unseen-work independence.'])


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def note_identity(notes):
    rows = sorted((n['start'], n['duration'], n['pitch'], n['velocity'], n['program']) for n in notes)
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def parse_with_tracks(blob):
    midi = mido.MidiFile(file=io.BytesIO(blob))
    if midi.type == 2:
        raise ValueError('Asynchronous type-2 MIDI is unsupported')
    events = []
    names = {}
    for track_id, track in enumerate(midi.tracks):
        tick = 0
        for index, message in enumerate(track):
            tick += message.time
            events.append((tick, track_id, index, message))
            if message.type == 'track_name':
                names[track_id] = message.name
    programs = defaultdict(int)
    active = defaultdict(deque)
    notes = []
    unmatched = 0
    scale = 480 / midi.ticks_per_beat
    for tick, track, _, msg in sorted(events, key=lambda row: row[:3]):
        if msg.type == 'program_change':
            programs[msg.channel] = msg.program
        elif msg.type == 'note_on' and msg.velocity:
            active[msg.channel, msg.note].append((tick, msg.velocity, programs[msg.channel], track))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and not msg.velocity):
            queue = active[msg.channel, msg.note]
            if not queue:
                unmatched += 1
                continue
            start, velocity, program, onset_track = queue.popleft()
            if tick > start and msg.channel != 9:
                notes.append(dict(start=start * scale, duration=(tick - start) * scale,
                                  pitch=msg.note, velocity=velocity, program=program,
                                  source_track=onset_track, source_channel=msg.channel,
                                  source_name=names.get(onset_track, '')))
    return dict(notes=notes, midi_type=midi.type, tracks=len(midi.tracks),
                unmatched_note_offs=unmatched, unfinished_notes=sum(map(len, active.values())))


def interval_coverage(notes, start, stop):
    events = defaultdict(int)
    for n in notes:
        a, b = max(start, n['start']), min(stop, n['start'] + n['duration'])
        if b > a:
            events[a] += 1
            events[b] -= 1
    previous, active, sounding, polyphonic, maximum = start, 0, 0., 0., 0
    for tick in sorted(events):
        duration = tick - previous
        sounding += duration * (active > 0)
        polyphonic += duration * (active >= 2)
        active += events[tick]
        maximum = max(maximum, active)
        previous = tick
    return sounding / (stop - start), polyphonic / max(sounding, 1), maximum


def statistics(notes, start, stop):
    duty, polyphonic, maximum = interval_coverage(notes, start, stop)
    onset_groups = defaultdict(list)
    for n in notes:
        if start <= n['start'] < stop:
            onset_groups[n['start']].append(n['pitch'])
    # Top voice per onset approximates contour; it is not recovered voice identity.
    contour = [max(onset_groups[t]) for t in sorted(onset_groups)]
    intervals = np.abs(np.diff(contour))
    ioi = np.diff(sorted(onset_groups)) / 480
    pitches = [n['pitch'] for n in notes]
    return dict(note_events=len(notes), onset_note_events=sum(len(v) for v in onset_groups.values()),
                onset_groups=len(onset_groups), onset_groups_per_bar=len(onset_groups) / 4,
                duty=float(duty), polyphonic_sounding_fraction=float(polyphonic), max_polyphony=int(maximum),
                mean_duration_beats=float(np.mean([min(stop, n['start'] + n['duration']) - max(start, n['start']) for n in notes]) / 480),
                median_pitch=float(np.median(pitches)),
                contour_transitions=len(intervals),
                small_interval_fraction=float(np.mean(intervals <= 4)) if len(intervals) else None,
                ioi_cv=float(np.std(ioi) / np.mean(ioi)) if len(ioi) and np.mean(ioi) > 0 else None,
                median_ioi_beats=float(np.median(ioi)) if len(ioi) else None,
                pitch_diversity=len(set(pitches)))


def qualify(role, notes, start, stop):
    """Return explicit reasons even when a known transfer patch fails behavior."""
    program = notes[0]['program']
    lo, hi = RANGES[role]
    ranged = [n for n in notes if lo <= n['pitch'] <= hi]
    if not ranged:
        return None, ['no_in_range_sounding_notes'], None
    stats = statistics(ranged, start, stop)
    stats['target_range_fraction'] = len(ranged) / len(notes)
    if program in EXACT[role]:
        return 'exact', [], stats
    if program not in TRANSFER[role]:
        return None, ['program_not_eligible'], stats
    # Behavior must be assessed BEFORE range filtering. Otherwise an unsuitable
    # high melody could masquerade as a suitable bass after dropping most notes.
    source = statistics(notes, start, stop)
    stats['source_behavior'] = source
    bad = []
    if stats['target_range_fraction'] < .80:
        bad.append('target_range_fraction_below_0.80')
    if role == 'organ':
        if source['duty'] < .50: bad.append('duty_below_0.50')
        if source['mean_duration_beats'] < 1: bad.append('mean_duration_below_1_beat')
        if not 48 <= source['median_pitch'] <= 76: bad.append('median_outside_48_76')
        if source['onset_groups_per_bar'] > 6: bad.append('onset_density_above_6_groups_per_bar')
        chordal = source['polyphonic_sounding_fraction'] >= .15 and source['max_polyphony'] >= 2
        pulse = source['onset_groups'] >= 4 and source['ioi_cv'] is not None and source['ioi_cv'] <= .25 and .5 <= source['median_ioi_beats'] <= 4
        if not (chordal or pulse): bad.append('neither_chordal_nor_regular_pulse')
    elif role == 'strings':
        if source['duty'] < .25: bad.append('duty_below_0.25')
        if source['mean_duration_beats'] < .5: bad.append('mean_duration_below_0.5_beat')
        if source['median_pitch'] < 58: bad.append('median_below_58')
        moving = source['contour_transitions'] >= 2 and source['small_interval_fraction'] >= .60
        sustained = source['mean_duration_beats'] >= 2 and source['duty'] >= .5
        if not (moving or sustained): bad.append('neither_smooth_contour_nor_sustained_texture')
    else:
        if source['mean_duration_beats'] < .75: bad.append('mean_duration_below_0.75_beat')
        if not 36 <= source['median_pitch'] <= 62: bad.append('median_outside_36_62')
        if source['polyphonic_sounding_fraction'] > .10: bad.append('polyphonic_sounding_fraction_above_0.10')
    return (None if bad else 'transfer'), bad, stats


def matching(candidates):
    """Exact maximum-weight assignment, including strings/cello compatibility.

    Enumerate the coupled strings/cello choices; the best remaining organ edge
    is optimal for each pair. Dummy missing nodes carry zero weight.
    """
    options = {r: sorted(candidates[r], key=lambda c: (-c['weight'], c['source_id'])) for r in ROLES}
    best_value, best = -1., {r: None for r in ROLES}
    collisions, register_conflicts = 0, 0
    for strings in [*options['strings'], None]:
        for cello in [*options['cello'], None]:
            if strings and cello:
                if strings['source_track'] == cello['source_track']:
                    collisions += 1
                    continue
                if ('transfer' in (strings['provenance'], cello['provenance']) and
                    strings['stats']['median_pitch'] <= cello['stats']['median_pitch']):
                    register_conflicts += 1
                    continue
            used = {part['source_track'] for part in (strings, cello) if part}
            organ = next((c for c in options['organ'] if c['source_track'] not in used), None)
            value = sum(c['weight'] for c in (organ, strings, cello) if c)
            if value > best_value:
                best_value = value
                best = dict(organ=organ, strings=strings, cello=cello)
    return best, dict(colliding_string_cello_combinations=collisions,
                      incompatible_register_combinations=register_conflicts)


def recover_job(job):
    metadata, blob, cache = job
    path = Path(cache)
    try:
        if path.exists():
            with gzip.open(path, 'rt', encoding='utf-8') as f: parsed = json.load(f)
        else:
            parsed = parse_with_tracks(blob)
            parsed['midi_sha256'] = hashlib.sha256(blob).hexdigest()
            parsed['notes_identity'] = note_identity(parsed['notes'])
            with gzip.open(path, 'wt', encoding='utf-8') as f: json.dump(parsed, f)
        if parsed['notes_identity'] != metadata['raw_identity']:
            raise ValueError('Recovered note content differs from original cached raw notes')
        return metadata, parsed, None
    except Exception as error:
        return metadata, None, str(error)


def audit_record(metadata, parsed, windows_file, rejections_file):
    notes = parsed['notes']
    counters = Counter()
    groups = defaultdict(list)
    for n in notes:
        groups[(n['source_track'], n['source_channel'], n['program'])].append(n)
    end = max((n['start'] + n['duration'] for n in notes), default=0)
    # Full four-bar windows only, same alignment for all source parts.
    categories = set()
    for start in range(0, int(end) - 7680 + 1, 7680):
        stop = start + 7680
        candidates = {r: [] for r in ROLES}
        for (track, channel, program), stream in sorted(groups.items()):
            current = [n for n in stream if n['start'] < stop and n['start'] + n['duration'] > start]
            if not current:
                continue
            for role in ROLES:
                if program not in EXACT[role] | TRANSFER[role]:
                    counters['ineligible_program_role_pairs'] += 1
                    continue
                provenance, reasons, stats = qualify(role, current, start, stop)
                identity = f'track={track};channel={channel};program={program}'
                if reasons:
                    for reason in reasons: counters['rejected:' + role + ':' + reason] += 1
                    rejections_file.write(json.dumps(dict(song=metadata['song'], group=metadata['group'], split=metadata['split'],
                        start_tick=start, source_id=identity, role=role, source_name=current[0]['source_name'],
                        reasons=reasons, stats=stats), allow_nan=False) + '\n')
                    continue
                counters['eligible:' + role + ':' + provenance] += 1
                candidates[role].append(dict(source_id=identity, source_track=track, source_channel=channel,
                    source_program=program, source_name=current[0]['source_name'], provenance=provenance, stats=stats,
                    weight=(2 if provenance == 'exact' else 1) + .01 * stats['duty']))
        assigned, conflicts = matching(candidates)
        counters.update(conflicts)
        provenance = {role: value['provenance'] if value else 'missing' for role, value in assigned.items()}
        filled = [v for v in assigned.values() if v]
        assert len({v['source_track'] for v in filled}) == len(filled)
        if len(filled) == 3:
            category = 'exact_full_trio' if all(v['provenance'] == 'exact' for v in filled) else 'transferred_full_trio'
        else:
            category = 'partial_or_missing'
        counters['windows:' + category] += 1
        categories.add(category)
        exact, _ = matching({r: [c for c in cs if c['provenance'] == 'exact'] for r, cs in candidates.items()})
        if all(exact.values()): counters['windows:exact_only_matching_full_trio'] += 1
        windows_file.write(json.dumps(dict(song=metadata['song'], group=metadata['group'], source=metadata['source'],
            split=metadata['split'], start_tick=start, end_tick=stop, midi_sha256=parsed['midi_sha256'],
            role_provenance=provenance, assignment=assigned, category=category), allow_nan=False) + '\n')
    return counters, categories


def jobs_from_archives(records, cache):
    lookup = {r['song']: {k: r[k] for k in ('song', 'group', 'source', 'split', 'available_roles')} |
              {'raw_identity': note_identity(r['raw'])} for r in records}
    found = set()
    for source, archive in [('pdmx', ROOT / 'assets/composition_v1/mid.tar.gz'),
                            ('slakh', ROOT / 'assets/trio_ensemble/slakh_midi.tar.gz')]:
        if not archive.exists():
            raise FileNotFoundError(f'Cached source archive required; no automatic download: {archive}')
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith('.mid'):
                    continue
                if source == 'pdmx':
                    song = 'pdmx:' + Path(member.name).stem
                else:
                    track = next((p for p in Path(member.name).parts if p.startswith('Track')), '')
                    song = 'slakh:' + track
                if song not in lookup or song in found:
                    continue
                found.add(song)
                path = cache / (hashlib.sha256(song.encode()).hexdigest() + '.json.gz')
                blob = None if path.exists() else tar.extractfile(member).read()
                yield lookup[song], blob, str(path)
    missing = sorted(set(lookup) - found)
    if missing:
        raise ValueError('Source MIDI not found: ' + str(missing[:10]))


def report(summary, directory):
    rows = ['# Functional role audit', '',
        'Read-only audit. No model training, corpus replacement, or new note composition.',
        'Transfer labels are heuristic; accepted windows are not verified musical-quality examples.', '',
        '| Coverage measure | Train | Validation | Test |', '|---|---:|---:|---:|']
    for title, key in [('Original exact source records (old eligibility)', 'baseline_exact_records'),
                       ('Qualified exact full-trio windows', 'exact_full_trio_windows'),
                       ('Qualified transferred full-trio windows', 'transferred_full_trio_windows'),
                       ('Groups with an exact full-trio window', 'exact_full_trio_groups'),
                       ('Groups with a transferred full-trio window', 'transferred_full_trio_groups')]:
        rows.append('| ' + title + ' | ' + ' | '.join(str(summary[s][key]) for s in ('train', 'validation', 'test')) + ' |')
    rows.extend(['', 'Group rows can overlap: one work may contain exact and transferred windows.',
        'Window counts are not independent composition counts. Original record counts do not imply simultaneous eligible parts.',
        '', '## Provenance and assignment', '',
        f'Recovered source records: {summary["recovery"]["verified_records"]}. Failures: {len(summary["recovery"]["failures"])}.',
        f'Source MIDI types: {dict(summary["recovery"]["midi_types"])}. Unfinished source notes: {summary["recovery"]["unfinished_notes"]}; unmatched note-offs: {summary["recovery"]["unmatched_note_offs"]}.',
        'These event diagnostics also exist upstream of the old raw cache: recovery parity does not establish clean source MIDI. Unfinished notes are not invented/completed.',
        'Every recovered record must match the cached raw note content. Source track, channel, program and MIDI SHA256 are retained.',
        'One physical source track can occupy at most one role per window. This conservatively excludes multitimbral format-0 trio mapping.',
        'Transfers must pass behavioral thresholds before assignment; exact labels do not inherit those behavior gates.',
        'No source pitches are invented/transposed and no lowest-voice pruning is used in this audit.',
        'Training-set playable projection, post-projection coverage, deduplication of transferred windows and listener review are still required before training.',
        '', '## Most frequent behavioral rejections', ''])
    rejections = Counter()
    for split in ('train', 'validation', 'test'):
        rejections.update({k: v for k, v in summary[split]['counters'].items() if k.startswith('rejected:')})
    for reason, count in rejections.most_common(15): rows.append(f'- {reason}: {count}')
    rows.extend(['', 'Thresholds are an explicit first-pass hypothesis, not validated instrument equivalence.',
                 'The report must not be used to select thresholds based on held-out coverage or reward outcomes.',
                 'If mapping changes are needed, calibrate using training examples and create a new protocol version.'])
    (directory / 'REPORT.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    protocol_path = directory / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != PROTOCOL:
        raise ValueError('Existing protocol differs. Use a new audit version.')
    if (directory / 'summary.json').exists():
        print((directory / 'REPORT.md').read_text(encoding='utf-8'))
        return
    write(protocol_path, PROTOCOL)
    corpus_path = ROOT / 'assets/composition_v1/corpus.json.gz'
    before = digest(corpus_path)
    with gzip.open(corpus_path, 'rt') as f: records = json.load(f)
    cache = directory / 'source_tracks'
    cache.mkdir(exist_ok=True)
    summary = {s: dict(baseline_exact_records=sum(r['split'] == s and len(r['available_roles']) == 3 for r in records), counters=Counter()) for s in ('train', 'validation', 'test')}
    groups = {s: defaultdict(set) for s in summary}
    recovery = dict(verified_records=0, failures=[], midi_types=Counter(), unfinished_notes=0, unmatched_note_offs=0)
    with gzip.open(directory / 'windows.jsonl.gz', 'wt', encoding='utf-8') as windows, \
         gzip.open(directory / 'rejections.jsonl.gz', 'wt', encoding='utf-8') as rejects, \
         ProcessPoolExecutor(max_workers=max(1, min(args.workers, 4))) as pool:
        for index, (metadata, parsed, error) in enumerate(pool.map(recover_job, jobs_from_archives(records, cache), chunksize=4)):
            if error:
                recovery['failures'].append(dict(song=metadata['song'], reason=error))
                continue
            recovery['verified_records'] += 1
            recovery['midi_types'][str(parsed['midi_type'])] += 1
            recovery['unfinished_notes'] += parsed['unfinished_notes']
            recovery['unmatched_note_offs'] += parsed['unmatched_note_offs']
            counters, categories = audit_record(metadata, parsed, windows, rejects)
            split = metadata['split']
            summary[split]['counters'].update(counters)
            for category in categories: groups[split][category].add(metadata['group'])
            if (index + 1) % 100 == 0:
                print(f'Audited {index+1}/{len(records)} original records; no training', flush=True)
    for split in summary:
        counters = summary[split]['counters']
        for category in ('exact_full_trio', 'transferred_full_trio'):
            summary[split][category + '_windows'] = counters['windows:' + category]
            summary[split][category + '_groups'] = len(groups[split][category])
    summary['recovery'] = recovery
    summary['corpus_records'] = len(records)
    summary['source_corpus_sha256'] = before
    summary['protocol_sha256'] = digest(protocol_path)
    summary['original_corpus_unchanged'] = digest(corpus_path) == before
    summary['training_started'] = False
    if not summary['original_corpus_unchanged']:
        raise ValueError('Source corpus unexpectedly changed')
    write(directory / 'summary.json', summary)
    report(summary, directory)
    print((directory / 'REPORT.md').read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    main()
