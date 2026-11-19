"""Verify original MIDI meter and byte identity before the new training run."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import mido

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'outputs/orchestra_learning_v1/data'


def inspect(job):
    song,blob=job
    try:
        midi=mido.MidiFile(file=io.BytesIO(blob))
        signatures=sorted({(m.numerator,m.denominator) for t in midi.tracks for m in t if m.type=='time_signature'})
        return song,dict(sha256=hashlib.sha256(blob).hexdigest(),signatures=signatures,
                         supported=all(pair==(4,4) for pair in signatures),implicit_default_4_4=not signatures)
    except Exception as error:return song,dict(supported=False,error=str(error))


def main():
    meter_file=DATA/'source_meter.json'
    if meter_file.exists():meter=json.loads(meter_file.read_text())
    else:
        sources=json.loads((DATA/'sources.json').read_text());wanted={r['song'] for r in sources}
        def jobs():
            found=set()
            for kind,path in [('pdmx',ROOT/'assets/composition_v1/mid.tar.gz'),('slakh',ROOT/'assets/trio_ensemble/slakh_midi.tar.gz')]:
                with tarfile.open(path,'r|gz') as tar:
                    for member in tar:
                        if not member.isfile() or not member.name.endswith('.mid'):continue
                        song='pdmx:'+Path(member.name).stem if kind=='pdmx' else 'slakh:'+next((p for p in Path(member.name).parts if p.startswith('Track')),'')
                        if song in wanted and song not in found:found.add(song);yield song,tar.extractfile(member).read()
            if found!=wanted:raise ValueError('Unverified source remains')
        meter={}
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i,(song,result) in enumerate(pool.map(inspect,jobs(),chunksize=4)):
                meter[song]=result
                if (i+1)%400==0:print('Verified meter and bytes',i+1,flush=True)
        meter_file.write_text(json.dumps(meter,indent=2))
    with gzip.open(DATA/'examples.json.gz','rt') as f:windows=json.load(f)
    kept=[]
    for w in windows:
        row=meter[w['song']]
        if not row['supported']:continue
        if row['sha256']!=w['source_midi_sha256']:raise ValueError('Source cache byte identity mismatch')
        kept.append(w)
    summary=json.loads((DATA/'summary.json').read_text());roles=list(summary['composition_groups']['train'])
    counts={s:{r:len({w['group'] for w in kept if w['split']==s and r in w['available_roles']}) for r in roles} for s in ['train','validation','test']}
    with gzip.open(DATA/'examples.json.gz','wt') as f:json.dump(kept,f)
    summary.update(examples=len(kept),splits={s:sum(w['split']==s for w in kept) for s in counts},composition_groups=counts,
                   meter_verified=True,meter_excluded_windows=len(windows)-len(kept),
                   source_byte_identity_verified=True,validation_gate_passed=all(v>=12 for v in counts['validation'].values()),
                   examples_sha256=hashlib.sha256((DATA/'examples.json.gz').read_bytes()).hexdigest())
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2))
    print('Meter-audited data:',summary['splits'],'minimum validation groups',min(counts['validation'].values()),flush=True)


if __name__=='__main__':main()
