"""Read-only live dashboard for actual orchestral training logs."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'outputs/orchestra_learning_v1'
DATA=RUN/'data/summary.json'
LOG=ROOT/'outputs/orchestra_training_v1.log'
GPU={};LAST=0


def snapshot():
    global GPU,LAST
    try:state=json.loads((RUN/'training_log.json').read_text())
    except (OSError,json.JSONDecodeError):state={}
    try:data=json.loads(DATA.read_text())
    except (OSError,json.JSONDecodeError):data={}
    if time.monotonic()-LAST>5:
        try:
            line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).strip().splitlines()[0]
            used,total,util=[int(s.strip()) for s in line.split(',')];GPU=dict(used=used,total=total,util=util)
        except Exception:GPU={}
        LAST=time.monotonic()
    try:
        tail=LOG.read_text(encoding='utf-8-sig',errors='replace')
        if '\x00' in tail:tail=LOG.read_text(encoding='utf-16',errors='replace')
        tail=tail.splitlines()[-10:]
    except OSError:tail=[]
    return dict(state=state,data=data,gpu=GPU,tail=tail)


PAGE='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Training the orchestra together</title><style>body{font:17px system-ui;background:#101724;color:#e9f1fc;max-width:1100px;margin:30px auto;padding:20px;line-height:1.5}section{background:#1e2d43;padding:24px;border-radius:14px;margin:20px 0}.stats{display:flex;gap:30px;flex-wrap:wrap}.value{font-size:27px;font-weight:650}progress,canvas{width:100%}canvas{height:250px}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #40516a}pre{white-space:pre-wrap;font-size:13px}.muted{color:#b2c4dc}</style><h1>Training the orchestra together</h1><p>19 instrument-role slots · 7 shared family LoRA adapters · causal phrase planner and partner conditioning. This is supervised training, not RL.</p><p id="data"></p><section><p id="status">Loading actual logs…</p><div class="stats"><div>Stage<div class="value" id="stage">—</div></div><div>Optimizer step<div class="value" id="step">—</div></div><div>Latest loss<div class="value" id="loss">—</div></div><div>Elapsed<div class="value" id="elapsed">—</div></div></div><progress id="progress" value="0" max="64"></progress><p id="batch"></p></section><section><div class="stats"><div>GPU memory<div class="value" id="memory">—</div></div><div>GPU utilization<div class="value" id="util">—</div></div></div></section><section><h2>Training loss</h2><p class="muted">Grey: actual updates. Green: 8-update moving average. Lower loss is not proof of better music.</p><canvas id="plot"></canvas></section><section><h2>Held-out role validation</h2><p class="muted">Event loss relative to this stage's baseline. Below 1 means lower loss. Composition groups remain separate; missing instruments receive no note loss.</p><table><thead><tr><th>Role</th><th>Validation groups</th><th>Event ratio</th><th>Attribute ratio</th></tr></thead><tbody id="roles"></tbody></table></section><section><h2>Actual latest output</h2><pre id="tail"></pre></section><script>
const el=id=>document.getElementById(id);let latest;
function plot(values){const c=el('plot'),scale=devicePixelRatio||1,w=c.clientWidth,h=250;c.width=w*scale;c.height=h*scale;const g=c.getContext('2d');g.scale(scale,scale);if(!values.length)return;const hi=Math.max(1,...values)*1.05,p=35;for(let i=0;i<5;i++){const y=h-p-i/4*(h-2*p);g.strokeStyle='#3d4f69';g.beginPath();g.moveTo(p,y);g.lineTo(w-p,y);g.stroke();g.fillStyle='#b2c4dc';g.fillText((hi*i/4).toFixed(2),0,y+4)}for(const [color,data] of [['#718298',values],['#7ae0b6',values.map((_,i)=>{const a=values.slice(Math.max(0,i-7),i+1);return a.reduce((x,y)=>x+y,0)/a.length})]]){g.strokeStyle=color;g.lineWidth=2;g.beginPath();data.forEach((v,i)=>{const x=p+i/Math.max(1,data.length-1)*(w-2*p),y=h-p-v/hi*(h-2*p);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()}}
function draw(d){latest=d;const s=d.state,g=d.gpu,last=s.updates?.at(-1),val=(s.validations||[]).filter(v=>v.stage===s.stage).at(-1);el('status').textContent=s.status||'Preparing data / model';el('stage').textContent=s.stage||'verification';el('step').textContent=(s.step||0)+' / '+(s.total||64);el('loss').textContent=last?last.loss.toFixed(4):'—';el('elapsed').textContent=((s.elapsed||0)/60).toFixed(1)+' min';el('progress').max=s.total||64;el('progress').value=s.step||0;el('batch').textContent='Microbatch capacity '+(s.batch?.capacity||128)+' · latest physical batch '+(last?.physical_microbatch??'pending')+' · latest beat examples '+(last?.examples??'pending');el('data').textContent='Verified phrases: '+(d.data.splits?.train||0)+' train / '+(d.data.splits?.validation||0)+' validation. All role slots have at least '+Math.min(...Object.values(d.data.composition_groups?.validation||{a:0}))+' independent validation groups.';el('memory').textContent=g.total?(g.used/1024).toFixed(2)+' / '+(g.total/1024).toFixed(2)+' GB':'Unavailable';el('util').textContent=g.util!=null?g.util+'%':'Unavailable';el('tail').textContent=d.tail.join(String.fromCharCode(10));plot((s.updates||[]).filter(v=>v.stage===s.stage).map(v=>v.loss));el('roles').replaceChildren();for(const [role,count] of Object.entries(d.data.composition_groups?.validation||{})){const tr=document.createElement('tr'),a=val?.scores[role],b=s.initial?.[role];for(const value of [role.replaceAll('_',' '),count,a&&b?(a.event/b.event).toFixed(3):'pending',a&&b&&a.attributes!=null?(a.attributes/b.attributes).toFixed(3):'pending']){const td=document.createElement('td');td.textContent=value;tr.append(td)}el('roles').append(tr)}}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});draw(await r.json())}catch(e){el('status').textContent='Retrying: '+e.message}setTimeout(refresh,2000)}refresh();window.onresize=()=>{if(latest)draw(latest)};
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/favicon.ico':self.send_response(204);self.end_headers();return
        if self.path not in ['/','/api/status']:self.send_error(404);return
        body=json.dumps(snapshot()).encode() if self.path=='/api/status' else PAGE.encode()
        self.send_response(200);self.send_header('Content-Type','application/json' if self.path=='/api/status' else 'text/html; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--port',type=int,default=8794);parser.add_argument('--log',type=Path,default=LOG)
    parser.add_argument('--data',type=Path,default=DATA);args=parser.parse_args()
    RUN=args.run.resolve();LOG=args.log.resolve();DATA=args.data.resolve()
    PAGE=PAGE.replace("s.batch?.capacity||128","s.physical_capacity||s.batch?.capacity||128")
    PAGE=PAGE.replace('<progress id="progress"','<p id="selected"></p><progress id="progress"')
    PAGE=PAGE.replace("el('progress').max=","el('selected').textContent='Current selected validation step: '+(s.selected_step||0)+'; checkpoints save periodically.';el('progress').max=")
    print(f'Live orchestral learning: http://127.0.0.1:{args.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
