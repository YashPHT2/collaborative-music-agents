"""A five-pair, single-question alternative to the full calibration ballot."""
import base64,hashlib,json,random,re,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    source=ROOT/'outputs/band_team_ballot_v2'; out=ROOT/'outputs/band_quick_vote'; out.mkdir(exist_ok=True)
    original=json.loads((source/'organizer_manifest.json').read_text())
    cases=[next(c for c in original['pairs'] if c['passage']==p and c['kind']=='reward_training')
           for p in ['p01','p03','p05','p07','p09']]
    random.Random(712).shuffle(cases)
    pairs=[]; private=[]
    for i,c in enumerate(cases):
        methods={m['method']:m for m in c['mapping'].values()}
        order=['previous','reward_v2'] if i%2==0 else ['reward_v2','previous']
        pair={'id':f'Q{i+1}'}; mapping={}
        for label,method in zip(['A','B'],order):
            item=methods[method]; data=(source/item['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest()==item['sha256']
            pair[label]='data:audio/mpeg;base64,'+base64.b64encode(data).decode()
            mapping[label]=item
        pairs.append(pair); private.append(dict(id=pair['id'],passage=c['passage'],mapping=mapping))
    page=PAGE.replace('__DATA__',json.dumps(pairs))
    (out/'OPEN_ME.html').write_text(page,encoding='utf-8')
    (out/'organizer_manifest.json').write_text(json.dumps(dict(ballot_id='band-quick-v1',pairs=private,
        note='Five distinct source seeds, one enjoyment judgment per seed. Methods differ in reward and additional training.'),indent=2))
    (out/'READ_ME.txt').write_text('''QUICK BAND VOTE: 2–3 minutes
Extract this ZIP on a laptop and open OPEN_ME.html. No installation or internet.
For each of five pairs, press Listen A then B. Just choose which version you
would rather keep listening to. Same and Neither are valid; no music knowledge
needed. Next opens the next pair; Back lets you change a vote.
Vote independently before discussing. At the end download your votes or copy
the text for WhatsApp. Nothing is sent automatically. Export before closing.
This is a separate ballot: use its Q1–Q5 labels rather than the earlier B labels.
''',encoding='utf-8')
    (out/'WHATSAPP_MESSAGE.txt').write_text('Quick version: just 5 music comparisons, about 2–3 minutes. Extract the ZIP on a laptop, open OPEN_ME.html, and press Listen A then B. Choose whichever you would rather keep listening to. Same or Neither is fine. No music knowledge needed. Vote independently, then send the exported votes or copied text.\n',encoding='utf-8')
    with zipfile.ZipFile(out/'Quick_Band_Vote.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['OPEN_ME.html','READ_ME.txt']: z.write(out/name,'Quick_Band_Vote/'+name)
    script=re.search('<script>(.*?)</script>',page,re.S).group(1)
    (out/'organizer_script.js').write_text(script,encoding='utf-8')
    subprocess.run(['node','--check',str(out/'organizer_script.js')],check=True)
    assert all(word not in page for word in ['reward_v2','previous','organizer_manifest'])
    print(json.dumps(dict(pairs=5,megabytes=(out/'Quick_Band_Vote.zip').stat().st_size/1024**2,path=str(out/'Quick_Band_Vote.zip'))))

PAGE='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Quick music vote</title>
<style>body{font:19px system-ui;background:#111a29;color:#eef4ff;max-width:650px;margin:40px auto;padding:20px}section{background:#22324a;border-radius:18px;padding:24px}button{font:inherit;padding:13px 18px;border:0;border-radius:9px;margin:6px;cursor:pointer}button:disabled{opacity:.35;cursor:default}button[aria-pressed=true]{outline:4px solid #75d9b6}.primary{background:#92e5c6;color:#102a23}audio{width:100%;margin:12px 0}input,textarea{font:inherit;padding:8px;box-sizing:border-box;max-width:100%}textarea{width:100%}small{color:#b8c9df}#playing{min-height:30px}nav{display:flex;justify-content:space-between;margin-top:16px}</style>
<h1>Just listen. Pick your favourite.</h1><p>Five pairs. About 2–3 minutes. No music knowledge needed.</p>
<section><small id="progress"></small><h2 id="heading"></h2><button id="both" class="primary">▶ Listen A then B</button><p id="playing" role="status">Ready to listen.</p><audio id="player" controls preload="none" hidden></audio><div><button id="a">Replay A</button><button id="b">Replay B</button></div>
<h3>Which would you rather keep listening to?</h3><div id="choices"></div><p><small>If you cannot hear a difference, choose Same. If you dislike both, choose Neither.</small></p><nav><button id="back">← Back</button><button id="next" class="primary">Next →</button></nav></section>
<section id="finished" hidden><h2>That’s it—thank you.</h2><label>Name or alias (optional) <input id="name" maxlength="60"></label><p><button id="download" class="primary">Download my votes</button><button id="copy">Copyable WhatsApp text</button></p><textarea id="text" rows="8" readonly hidden></textarea><small>Send the file or text yourself. Nothing is submitted automatically.</small></section>
<script>const pairs=__DATA__,key='band-quick-v1'+(location.search==='?qa=1'?'-qa':'');let votes={},index=0,sequence=0;try{votes=JSON.parse(localStorage.getItem(key)||'{}')}catch{}const player=document.getElementById('player'),playing=document.getElementById('playing');
function save(){try{localStorage.setItem(key,JSON.stringify(votes))}catch{}}
function stop(){sequence++;player.hidden=true;player.pause();player.onended=null;player.removeAttribute('src');player.load()}
async function play(label,chain=false){stop();const run=sequence;player.hidden=false;player.src=pairs[index][label];playing.textContent='Playing '+label;player.onended=()=>{if(sequence!==run)return;if(chain&&label==='A')play('B');else playing.textContent='Finished. You can replay either version.'};try{await player.play()}catch{playing.textContent='Press the audio play button to listen to '+label}}
function draw(){stop();document.getElementById('progress').textContent='Pair '+(index+1)+' of 5 · '+Object.values(votes).filter(Boolean).length+'/5 voted';document.getElementById('heading').textContent='Pair '+(index+1);playing.textContent='Ready to listen.';document.getElementById('back').disabled=index===0;document.getElementById('next').disabled=!votes[pairs[index].id];document.getElementById('next').textContent=index===4?'Finish ✓':'Next →';const group=document.getElementById('choices');group.replaceChildren();for(const choice of ['A','B','Same','Neither']){const b=document.createElement('button');b.textContent=choice;b.setAttribute('aria-pressed',votes[pairs[index].id]===choice);b.onclick=()=>{votes[pairs[index].id]=choice;save();draw()};group.append(b)}}
document.getElementById('both').onclick=()=>play('A',true);document.getElementById('a').onclick=()=>play('A');document.getElementById('b').onclick=()=>play('B');
document.getElementById('back').onclick=()=>{if(index>0)index--;document.getElementById('finished').hidden=true;draw()};document.getElementById('next').onclick=()=>{if(index<4){index++;draw()}else{stop();document.getElementById('finished').hidden=false;document.getElementById('finished').scrollIntoView({behavior:'smooth'})}};
function result(){return{ballot_id:'band-quick-v1',participant:document.getElementById('name').value.trim(),created_at:new Date().toISOString(),ratings:pairs.map(p=>({id:p.id,choice:votes[p.id]||null}))}}
document.getElementById('download').onclick=()=>{const blob=new Blob([JSON.stringify(result(),null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='quick-band-votes.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
document.getElementById('copy').onclick=()=>{const r=result(),t=document.getElementById('text');t.hidden=false;t.value=['Quick band vote v1','Voter: '+(r.participant||'Anonymous'),...r.ratings.map(p=>p.id+': '+(p.choice||'unrated'))].join('\\n');t.focus();t.select()};draw();</script></html>'''

if __name__=='__main__': main()
