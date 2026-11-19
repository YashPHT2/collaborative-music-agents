"""Audit ballot assets, participant blinding and rewards; no listener votes invented."""
import base64,hashlib,json,re,subprocess,zipfile
from band import ROOT,Brief,ensemble_reward
from band_reward import ensemble_reward_v2
from prepare_band_ballot import clip,alter

def main():
    root=ROOT/'outputs/band_team_ballot_v2'
    organizer=json.loads((root/'organizer_manifest.json').read_text())
    page=(root/'OPEN_ME.html').read_text(encoding='utf-8')
    pairs=json.loads(re.search(r'const pairs=(.*?),audio=',page,re.S).group(1))
    embedded=json.loads(re.search(r',audio=(.*?),key=',page,re.S).group(1))
    assert len(embedded)==40
    for digest,uri in embedded.items():
        assert hashlib.sha256(base64.b64decode(uri.split(',',1)[1])).hexdigest()==digest
    audio_reports=list((root/'organizer_audio').glob('*_audio.json'))
    assert len(audio_reports)==40
    for report in audio_reports:
        data=json.loads(report.read_text()); assert abs(data['seconds']-10.6)<.01 and .001<data['peak']<.999
    assert len(pairs)==30 and len({p['id'] for p in pairs})==30
    assert all(word not in page for word in ['reward_v2','bass_displacement','piano_bar_order','organizer_manifest','previous'])
    counts={}; scores=[]
    for case in organizer['pairs']:
        counts[case['kind']]=counts.get(case['kind'],0)+1
        for label,item in case['mapping'].items():
            path=root/item['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
            public=next(p for p in pairs if p['id']==case['id'])
            assert public[label]==item['sha256']
        if case['kind']=='reward_training':
            passage=int(case['passage'][1:])-1; source=passage//2; half=passage%2
            a=clip(json.loads((root/f'organizer_audio/{920001+source}_previous_events.json').read_text()),half*7680)
            b=clip(json.loads((root/f'organizer_audio/{920001+source}_reward_v2_events.json').read_text()),half*7680)
            assert a!=b, f'Identical MIDI comparison: {case["id"]}'
    assert counts==dict(reward_training=10,bass_displacement=10,piano_bar_order=10)
    assert sum(c['mapping']['A']['method']=='reward_v2' for c in organizer['pairs'])==5
    # Separate diagnostic-score audit, linked to opaque ballot IDs only in organizer files.
    for case in organizer['pairs']:
        passage=int(case['passage'][1:])-1; source=passage//2; half=passage%2
        tonic=[9,0,2,5,7][source]; brief=Brief(4,100,tonic,bool(source%2==0))
        for label,item in case['mapping'].items():
            method=item['method']; cache_method='reward_v2' if method=='reward_v2' else 'previous'
            notes=json.loads((root/f'organizer_audio/{920001+source}_{cache_method}_events.json').read_text())
            notes=clip(notes,half*7680)
            if method in ['bass_displacement','piano_bar_order']: notes=alter(notes,method)
            for version,fn in [('v1',ensemble_reward),('v2',ensemble_reward_v2)]:
                values=[fn(notes,t,t+1920,brief) for t in range(0,7680,1920)]
                scores.append(dict(id=case['id'],label=label,version=version,
                    score=sum(v[0] for v in values)/4,
                    components={k:sum(v[1][k] for v in values)/4 for k in values[0][1]}))
    (root/'reward_calibration_scores.json').write_text(json.dumps(scores,indent=2))
    script=re.search(r'<script>(.*?)</script>',page,re.S).group(1)
    js=root/'organizer_ballot_script.js'; js.write_text(script,encoding='utf-8')
    subprocess.run(['node','--check',str(js)],check=True,capture_output=True)
    with zipfile.ZipFile(root/'Band_Team_Ballot_v2.zip') as z:
        assert set(z.namelist())=={'Band_Team_Ballot_v2/OPEN_ME.html','Band_Team_Ballot_v2/READ_ME.txt'}
        assert z.read('Band_Team_Ballot_v2/OPEN_ME.html')==(root/'OPEN_ME.html').read_bytes()
    # A synthetic decoder test is explicitly labeled and kept outside participant ZIP.
    sample=root/'SYNTHETIC_TEST_ONLY.json'
    sample.write_text(json.dumps(dict(ballot_id='band-team-v2',participant='SYNTHETIC_TEST_ONLY',ratings=[
        dict(id=pairs[0]['id'],enjoyment='A',coherence='Tie',together='Neither good')])) )
    target=root/'SYNTHETIC_DECODE_TEST_ONLY.json'
    import sys
    subprocess.run([sys.executable,str(ROOT/'decode_band_votes.py'),str(sample),'--output',str(target)],check=True,capture_output=True)
    decoded=json.loads(target.read_text())['ratings'][0]
    case=next(c for c in organizer['pairs'] if c['id']==pairs[0]['id'])
    assert decoded['preferences']['enjoyment']==case['mapping']['A']['method']
    assert decoded['preferences']['coherence']=='Tie' and decoded['preferences']['together']=='Neither good'
    result=dict(pairs=30,unique_passages=10,independent_source_seeds=5,participant_methods_hidden=True,
                audio_hashes_verified=True,archive_verified=True,javascript_syntax_valid=True,decoder_test_passed=True,
                real_listener_votes_collected=0,
                limitation='Previous vs new-reward checkpoint also differs in training duration; it does not isolate a causal reward effect.')
    (root/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
