"""Audit genre candidates against actual MIDI instruments before GPU allocation."""
import collections,hashlib,io,json,tarfile
from pathlib import Path
import mido

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/rock_training_v1'

def inspect(data):
    midi=mido.MidiFile(file=io.BytesIO(data));programs=collections.defaultdict(int)
    counts=collections.Counter()
    for msg in mido.merge_tracks(midi.tracks):
        if msg.type=='program_change':programs[msg.channel]=msg.program
        elif msg.type=='note_on' and msg.velocity:
            program=programs[msg.channel]
            family='drums' if msg.channel==9 else 'guitar' if 24<=program<=31 else 'bass' if 32<=program<=39 else 'keyboard' if program<=7 else 'other'
            counts[family]+=1
    return dict(counts)

def main():
    report=json.loads((OUT/'pdmx_rock_candidates.json').read_text())
    lookup={r['mid'].lstrip('./'):r for r in report['candidates']}
    found=[];errors=[]
    with tarfile.open(ROOT/'assets/composition_v1/mid.tar.gz','r:gz') as archive:
        for member in archive:
            key=member.name.lstrip('./')
            if key not in lookup:continue
            row=lookup[key].copy()
            try:
                with archive.extractfile(member) as f:data=f.read()
                row['instrument_note_counts']=inspect(data)
                row['midi_sha256']=hashlib.sha256(data).hexdigest()
                row['has_guitar_bass_drums']=all(row['instrument_note_counts'].get(k,0)>=16 for k in ['guitar','bass','drums'])
                found.append(row)
            except Exception as exc:errors.append(dict(id=row['id'],error=str(exc)))
    bands=[r for r in found if r['has_guitar_bass_drums']]
    positive=[r for r in bands if r['positive_tag']]
    result=dict(rock_metadata_candidates=len(lookup),archive_matches=len(found),parse_errors=errors,
        guitar_bass_drums_candidates=len(bands),positive_tag_band_candidates=len(positive),
        happy_mood_verified=False,rock_trainer_available=(ROOT/'train_rock_pilot.py').is_file(),training_allowed=False,
        reason='Candidate discovery only: run verified role/window preparation and composition-separated data gates before pilot training. Happy mood is not verified.',
        bands=bands,all_records=found)
    (OUT/'data_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['bands','all_records']},indent=2))

if __name__=='__main__':main()
