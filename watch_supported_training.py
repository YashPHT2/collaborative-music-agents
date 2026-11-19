"""Live read-only dashboard for the verified supported-prefix supervised run."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'outputs/supported_ensemble_run_v1'
GPU={'time':0.,'value':{}}


def snapshot():
    try:state=json.loads((RUN/'training_log.json').read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError):state={}
    if 'status' not in state and state.get('updates'):
        state['status']='Training: '+state['updates'][-1]['stage']+' (validation runs every 16 updates)'
    try:
        text=(ROOT/'outputs/supported_ensemble_training_v1.log').read_text(encoding='utf-8-sig',errors='replace')
        if '\x00' in text:text=(ROOT/'outputs/supported_ensemble_training_v1.log').read_text(encoding='utf-16',errors='replace')
        tail=text.splitlines()[-12:]
    except OSError:tail=[]
    if time.monotonic()-GPU['time']>5:
        try:
            output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            used,total,util=map(lambda x:int(x.strip()),output.splitlines()[0].split(','))
            GPU['value']=dict(used_mib=used,total_mib=total,utilization=util)
        except Exception:GPU['value']={}
        GPU['time']=time.monotonic()
    return dict(state=state,gpu=GPU['value'],tail=tail)


PAGE='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Supported trio — live supervised training</title>
<style>body{font:17px system-ui;line-height:1.5;background:#101724;color:#e9eef5;max-width:1100px;margin:32px auto;padding:20px}section{background:#1c2b40;border-radius:14px;padding:24px;margin:20px 0}.stats{display:flex;gap:30px;flex-wrap:wrap}.value{font-size:28px;font-weight:650}progress,canvas{width:100%}canvas{height:250px}pre{white-space:pre-wrap;font-size:13px}a{color:#9ed8ff}.muted{color:#b3bfd2}</style>
<h1>Live supervised trio training</h1><p>Organ · strings · cello. Frozen Transformer backbone; trainable LoRA adapters, planner and partner conditioning. <strong>No RL in this run.</strong></p>
<p>Audited dataset: 355 training phrases / 83 groups · 56 validation phrases / 19 groups. Supported sustains remain context; prefix loss is masked.</p>
<section><p id="status">Loading actual logs…</p><div class="stats"><div>Stage<div id="stage" class="value">—</div></div><div>Optimizer step<div id="step" class="value">—</div></div><div>Latest loss<div id="loss" class="value">—</div></div><div>Elapsed<div id="elapsed" class="value">—</div></div></div><progress id="progress" value="0" max="128"></progress><p id="batch" class="muted"></p></section>
<section><div class="stats"><div>GPU memory<div id="memory" class="value">—</div></div><div>GPU utilization<div id="util" class="value">—</div></div></div></section>
<section><h2>Training objective</h2><p class="muted">Grey: each optimizer update. Green: moving average. Loss varies with the sampled composition and provenance weights; lower loss alone does not prove better music.</p><canvas id="lossplot"></canvas></section>
<section><h2>Held-out balanced event loss</h2><p class="muted">Relative to this stage's start. Organ: green · strings: blue · cello: amber. Below 1 means lower validation loss. Exact/transfer regressions are checked separately during selection.</p><canvas id="valplot"></canvas><p id="validation"></p></section>
<section><h2>Latest actual output</h2><pre id="tail"></pre></section>
<script>
const el=id=>document.getElementById(id);let latest;
function chart(id,series){const c=el(id),d=devicePixelRatio||1,w=c.clientWidth,h=250;c.width=w*d;c.height=h*d;const g=c.getContext('2d');g.scale(d,d);if(!series.some(s=>s.values.length))return;const vals=series.flatMap(s=>s.values),lo=Math.min(0,...vals),hi=Math.max(1,...vals)*1.05,p=35;g.font='12px system-ui';for(let j=0;j<5;j++){const v=lo+(hi-lo)*j/4,y=h-p-j/4*(h-2*p);g.strokeStyle='#3b4b62';g.beginPath();g.moveTo(p,y);g.lineTo(w-p,y);g.stroke();g.fillStyle='#b3bfd2';g.fillText(v.toFixed(2),0,y+4)}for(const s of series){g.strokeStyle=s.color;g.lineWidth=2;g.beginPath();s.values.forEach((v,i)=>{const x=p+i/Math.max(1,s.values.length-1)*(w-2*p),y=h-p-(v-lo)/(hi-lo)*(h-2*p);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()}}
function render(d){latest=d;const s=d.state,g=d.gpu,last=s.updates?.at(-1),stage=['participation','revision'][s.stage]||'finished';el('status').textContent=s.status||'Verifying loader and preparing training';el('stage').textContent=stage;el('step').textContent=(s.step||0)+' / '+(s.total||128);el('loss').textContent=last?last.loss.toFixed(4):'—';el('elapsed').textContent=((s.elapsed||0)/60).toFixed(1)+' min';el('progress').max=s.total||128;el('progress').value=s.step||0;el('batch').textContent='Microbatch capacity: '+(s.batch?.microbatch||128)+' · latest actual largest forward batch: '+(last?.largest_physical_microbatch??'not recorded')+' · up to '+(s.batch?.maximum_beat_examples_per_update||192)+' beat examples per update.';el('memory').textContent=g.total_mib?(g.used_mib/1024).toFixed(2)+' / '+(g.total_mib/1024).toFixed(2)+' GB':'Unavailable';el('util').textContent=g.utilization!=null?g.utilization+'%':'Unavailable';el('tail').textContent=d.tail.join(String.fromCharCode(10));const rows=(s.updates||[]).filter(r=>r.stage===stage),values=rows.map(r=>r.loss);chart('lossplot',[{color:'#697c93',values},{color:'#75dfb8',values:values.map((_,i)=>{const a=values.slice(Math.max(0,i-7),i+1);return a.reduce((a,b)=>a+b,0)/a.length})}]);const logs=(s.logs||[]).filter(r=>r.stage===stage),roles=['organ','strings','cello'];chart('valplot',roles.map((r,i)=>({color:['#75dfb8','#8dbfff','#f2c771'][i],values:s.initial?[1,...logs.map(v=>v.validation[r].all.event/s.initial[r].all.event)]:[]})));el('validation').textContent=logs.length?roles.map(r=>r+': '+logs.at(-1).validation[r].all.event.toFixed(4)).join(' · '):'Baseline / first validation checkpoint pending.'}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error(r.status);render(await r.json())}catch(e){el('status').textContent='Retrying connection: '+e.message}setTimeout(refresh,2000)}refresh();window.onresize=()=>{if(latest)render(latest)};
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/favicon.ico':self.send_response(204);self.end_headers();return
        if self.path not in ('/','/api/status'):self.send_error(404);return
        body=json.dumps(snapshot()).encode() if self.path=='/api/status' else PAGE.encode()
        self.send_response(200);self.send_header('Content-Type','application/json' if self.path=='/api/status' else 'text/html; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8789);args=parser.parse_args()
    print(f'Live verified supervised run: http://127.0.0.1:{args.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
