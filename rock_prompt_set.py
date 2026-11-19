"""Three predeclared composition groups, paired step-128 context probes."""
import copy,gzip,hashlib,json
from pathlib import Path
import rock_prompt_diagnostic as d

ROOT=d.ROOT;OUT=ROOT/'outputs/rock_prompt_set_v1';SEEDS=[9976301,9976302,9976303]

def prepare():
    OUT.mkdir(exist_ok=True);d.g.rock.configure()
    with gzip.open(ROOT/'outputs/rock_training_v1/rock_examples.json.gz','rt') as f:windows=json.load(f)
    old=json.loads((d.OUT/'protocol.json').read_text())
    def eligible(w,roles):
        return w['group']!=old['prompt_group'] and not any(n[1]<3840<n[1]+n[2] for n in w['notes']) and all(any(n[3]==r and n[1]<3840 for n in w['notes']) for r in roles)
    rank=lambda w:hashlib.sha256(w['id'].encode()).hexdigest()
    full=sorted([w for w in windows if w['split']=='validation' and eligible(w,range(5))],key=rank)
    selected=[];groups=set()
    for w in full:
        if w['group'] in groups:continue
        selected.append(w);groups.add(w['group'])
        if len(selected)==2:break
    test=sorted([w for w in windows if w['split']=='test' and eligible(w,[0,2,3]) and w['group'] not in groups],key=rank)
    if len(selected)!=2 or not test:raise ValueError('Need two new five-part validation groups and one separate core-band test group')
    selected.append(test[0]);assert len({w['group'] for w in selected})==3
    goal=old['goal'];entries=[]
    for index,(w,seed) in enumerate(zip(selected,SEEDS),1):
        key=f'prompt_{index:02d}';directory=OUT/key;directory.mkdir(exist_ok=True)
        prefix=d.check_prefix([copy.deepcopy(n) for n in w['notes'] if n[1]<3840])
        p=dict(schema='rock-prompt-set-v1',prompt_id=key,seed=seed,bars=6,prefix_bars=2,target_bars=4,bpm=w['bpm'],
            prompt_source_window=w['id'],prompt_group=w['group'],prompt_split=w['split'],prefix=prefix,goal=goal,
            checkpoint_sha256={'step128':d.g.sha(ROOT/'outputs/rock_training_v1/checkpoints/final128.pt')},new_training=False,
            prefix_roles=[r for r,s in d.g.rock.ROLES.items() if any(n[3]==s['id'] for n in prefix)])
        path=directory/'protocol.json'
        if path.exists():assert json.loads(path.read_text())==p
        d.g.write_json(path,p);d.g.write_json(directory/'reference_window.json',w)
        entries.append(dict(id=key,seed=seed,group=w['group'],split=w['split'],bpm=w['bpm'],protocol_sha256=d.g.sha(path),prefix_roles=p['prefix_roles']))
    collection=dict(schema='rock-prompt-set-v1',checkpoint_step=128,prompts=entries,
        selection='Two first-SHA-ordered unused five-part validation composition groups, then first-SHA-ordered eligible core-band test group. No quality scoring or output selection.',
        scope='Three composition groups and three paired seeds. One test group is now used for this diagnostic; do not reuse its outcomes to fit or calibrate a reward.',
        missing_prefix_parts='Genuine absent parts remain absent in the reference; all five role policies generate in both conditions.',
        checkpoint_status='Final experimental step 128; failed the earlier drum-attribute selection guard. Not promoted by this experiment.',new_training=False)
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==collection
    d.g.write_json(path,collection)
    print(json.dumps(entries,indent=2),flush=True)

if __name__=='__main__':prepare()
