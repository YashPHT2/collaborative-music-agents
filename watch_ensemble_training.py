"""Read-only live view of real repair training logs and GPU use."""
import argparse
import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'ensemble_repair_v1'
PATTERN=re.compile(r'(participation|revision) (\d+)/(\d+): loss=([\d.eE+-]+)')
GPU_CACHE=dict(time=0,value={})


def snapshot():
    steps={};tail=[];latest_time=0;current=None
    for path in sorted((ROOT/'outputs').glob('ensemble_repair_training*.log'),key=lambda p:p.stat().st_mtime):
        text=path.read_text(encoding='utf-8-sig',errors='replace')
        if '\x00' in text:text=path.read_text(encoding='utf-16',errors='replace')
        current=None
        for line in text.splitlines():
            match=PATTERN.search(line)
            if match:
                stage,step,total,loss=match.groups()
                current=dict(stage=stage,step=int(step),total=int(total),loss=float(loss))
                steps[(stage,int(step))]=current
        tail=text.splitlines()[-18:]
        latest_time=max(latest_time,path.stat().st_mtime)
    state={}
    try:state=json.loads((OUT/'training_log.json').read_text())
    except (OSError,json.JSONDecodeError):pass
    if current is None and not state.get('complete'):
        stage='participation' if state.get('stage',0)==0 else 'revision'
        current=steps.get((stage,state.get('step',0)))
    if current is not None:
        rank=lambda stage:0 if stage=='participation' else 1
        steps={key:value for key,value in steps.items() if (rank(key[0]),key[1]) <= (rank(current['stage']),current['step'])}
    if time.monotonic()-GPU_CACHE['time']>5:
        try:
            flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
            output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3,creationflags=flags)
            used,total,util=[int(v.strip()) for v in output.splitlines()[0].split(',')]
            GPU_CACHE['value']=dict(used=used,total=total,utilization=util)
        except (OSError,subprocess.SubprocessError,ValueError):GPU_CACHE['value']={}
        GPU_CACHE['time']=time.monotonic()
    ordered=sorted(steps.values(),key=lambda r:(0 if r['stage']=='participation' else 1,r['step']))
    rl={}
    rl_root=ROOT/'outputs'/'joint_ensemble_rl_v1'
    for name,file in [('queue','queue_status.json'),('training','training_log.json')]:
        try:rl[name]=json.loads((rl_root/file).read_text())
        except (OSError,json.JSONDecodeError):pass
    return dict(steps=ordered,tail=tail[-18:],state=state,rl=rl,gpu=GPU_CACHE['value'],
                seconds_since_output=round(time.time()-latest_time) if latest_time else None)


PAGE='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Live trio training</title>
<style>body{background:#101724;color:#e8edf5;font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}h1{margin-bottom:6px}.muted{color:#a8b6cc}section{background:#1b293e;padding:20px;border-radius:12px;margin:18px 0}.stats{display:flex;gap:20px;flex-wrap:wrap}.stats div{min-width:155px}.big{font-size:27px;font-weight:650}canvas{width:100%;height:270px}pre{overflow:auto;font-size:14px;line-height:1.6;max-height:280px}table{width:100%;border-collapse:collapse}td,th{padding:9px;text-align:left;border-bottom:1px solid #34465e}progress{width:100%;accent-color:#73dfb6;height:15px}a{color:#a9d2ff}</style>
<h1>Live trio training</h1><p class="muted">Organ · strings · cello — refreshes every 2 seconds from the actual training logs.</p><p id="status">Connecting…</p>
<section><div class="stats"><div>Stage<div class="big" id="stage">—</div></div><div>Step<div class="big" id="step">—</div></div><div>Latest loss<div class="big" id="loss">—</div></div><div>CUDA microbatch<div class="big" id="batch">—</div></div></div><progress id="progress" max="512" value="0"></progress><p id="effective" class="muted"></p></section>
<section><div class="stats"><div>GPU memory<div class="big" id="memory">—</div></div><div>GPU utilization<div class="big" id="util">—</div></div><div>Recorded compute<div class="big" id="elapsed">—</div></div></div></section>
<section><b>Held-out validation — is each agent learning?</b><p class="muted">Relative to the start of this stage: 1.00 = initial loss; lower is better. Organ: green · strings: blue · cello: amber. This is imitation accuracy, not a listening score.</p><canvas id="validation"></canvas><p id="validationSummary"></p></section>
<section><b>Training loss by update</b><p class="muted">Grey: each update. Green: 16-step average. Different pieces produce different losses; training-objective and batch changes also affect this curve. Use held-out validation above to judge learning.</p><canvas id="chart"></canvas></section>
<section><b>Validation checkpoints</b><p class="muted">Balanced note/stop loss for each instrument. Selected candidates must also pass the note-attribute regression limit.</p><table><thead><tr><th>Stage / step</th><th>Organ</th><th>Strings</th><th>Cello</th><th>Eligible</th></tr></thead><tbody id="checks"></tbody></table></section>
<section><b>Latest training output</b><pre id="tail"></pre></section>
<script>let data;const el=id=>document.getElementById(id);function plot(rows){const c=el('chart'),dpr=devicePixelRatio||1;c.width=c.clientWidth*dpr;c.height=270*dpr;const g=c.getContext('2d');g.scale(dpr,dpr);const w=c.clientWidth,h=270,p=35;g.clearRect(0,0,w,h);if(!rows.length)return;const max=Math.max(1,...rows.map(r=>r.loss));g.font='12px system-ui';g.fillStyle='#a8b6cc';for(let i=0;i<5;i++){let y=p+(h-2*p)*i/4;g.strokeStyle='#34465e';g.beginPath();g.moveTo(p,y);g.lineTo(w-p,y);g.stroke();g.fillText((max*(1-i/4)).toFixed(1),0,y+4)}function line(values,color){g.strokeStyle=color;g.lineWidth=color==='#73dfb6'?2:1;g.beginPath();values.forEach((v,i)=>{const x=p+i/Math.max(1,values.length-1)*(w-2*p),y=h-p-v/max*(h-2*p);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()}line(rows.map(r=>r.loss),'#748399');line(rows.map((r,i)=>{const slice=rows.slice(Math.max(0,i-15),i+1);return slice.reduce((s,v)=>s+v.loss,0)/slice.length}),'#73dfb6');g.fillText('Step '+rows[0].step,p,h-5);g.fillText('Step '+rows.at(-1).step,w-90,h-5)}
function validation(s,stage){const c=el('validation'),dpr=devicePixelRatio||1;c.width=c.clientWidth*dpr;c.height=270*dpr;const g=c.getContext('2d');g.scale(dpr,dpr);const w=c.clientWidth,h=270,p=40,logs=(s.logs||[]).filter(r=>r.stage===stage),initial=s.initial;if(!initial||!logs.length)return;const end=Math.max(1,...logs.map(r=>r.step)),roles=['organ','strings','cello'],colors=['#73dfb6','#8bbcff','#f4c878'];const values=roles.flatMap(role=>logs.map(r=>r.validation[role].event/initial[role].event)),top=Math.max(1.1,...values)*1.02,bottom=Math.min(.6,...values)*.95;g.font='12px system-ui';g.fillStyle='#a8b6cc';for(let i=0;i<5;i++){const v=bottom+(top-bottom)*i/4,y=h-p-i/4*(h-2*p);g.strokeStyle='#34465e';g.beginPath();g.moveTo(p,y);g.lineTo(w-p,y);g.stroke();g.fillText(v.toFixed(2),0,y+4)}roles.forEach((role,index)=>{g.strokeStyle=colors[index];g.fillStyle=colors[index];g.lineWidth=2.5;const points=[{step:0,value:1},...logs.map(r=>({step:r.step,value:r.validation[role].event/initial[role].event}))];g.beginPath();points.forEach((r,i)=>{const x=p+r.step/end*(w-2*p),y=h-p-(r.value-bottom)/(top-bottom)*(h-2*p);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();points.forEach(r=>{g.beginPath();g.arc(p+r.step/end*(w-2*p),h-p-(r.value-bottom)/(top-bottom)*(h-2*p),4,0,Math.PI*2);g.fill()})});g.fillStyle='#a8b6cc';g.fillText('Stage start',p,h-8);g.fillText('Step '+end,w-90,h-8);el('validationSummary').textContent=roles.map(role=>{const improvement=100*(1-logs.at(-1).validation[role].event/initial[role].event);return role+': '+Math.abs(improvement).toFixed(1)+'% '+(improvement>=0?'lower':'higher')+' balanced event loss'}).join(' · ')}
async function refresh(){try{const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);data=await response.json();const s=data.state,last=data.steps.at(-1),gpu=data.gpu;el('status').textContent=s.complete?'Training complete. Listening evaluation is separate.':'Last training output '+data.seconds_since_output+' seconds ago. '+(data.seconds_since_output>30?'Validation, generation, or a paused update may be in progress.':'');if(last){el('stage').textContent=last.stage;el('step').textContent=last.step+' / '+last.total;el('loss').textContent=last.loss.toFixed(4);el('progress').max=last.total;el('progress').value=last.step;plot(data.steps.filter(r=>r.stage===last.stage));validation(s,last.stage)}el('batch').textContent=s.batch?.microbatch||1;el('effective').textContent='Up to '+(s.batch?.maximum_beat_examples_per_update||24)+' beat examples per optimizer update across the three agents.';el('memory').textContent=gpu.total?(gpu.used/1024).toFixed(2)+' / '+(gpu.total/1024).toFixed(2)+' GB':'Unavailable';el('util').textContent=gpu.utilization!=null?gpu.utilization+'%':'Unavailable';el('elapsed').textContent=((s.elapsed||0)/60).toFixed(1)+' min';el('tail').textContent=data.tail.join('\\n');el('checks').replaceChildren();(s.logs||[]).slice(-8).forEach(r=>{const tr=document.createElement('tr');[r.stage+' / '+r.step,...['organ','strings','cello'].map(k=>r.validation[k].event.toFixed(4)),r.eligible?'Yes':'No'].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});el('checks').append(tr)})}catch(e){el('status').textContent='Connection interrupted; retrying. '+e.message}setTimeout(refresh,2000)}refresh();window.addEventListener('resize',()=>{if(data?.steps.length){plot(data.steps.filter(r=>r.stage===data.steps.at(-1).stage));validation(data.state,data.steps.at(-1).stage)}});</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/favicon.ico':
            self.send_response(204);self.end_headers();return
        if self.path not in ('/','/api/status'):
            self.send_error(404);return
        panel='''<section><b>Joint multi-agent reinforcement learning</b><p id="rlStatus">Loading queue status…</p><p id="rlResult"></p><p id="rlLoss"></p><p class="muted">PPO trains three independent note policies from self-generated ensemble rollouts. Team reward is an engineering proxy. Supervised graphs below belong to the preceding foundation stage.</p><a href="http://127.0.0.1:8780/">RL listening results (available after rendering)</a></section>
<script>async function refreshRL(){try{const d=await(await fetch('/api/status',{cache:'no-store'})).json(),q=d.rl?.queue,t=d.rl?.training;document.getElementById('rlStatus').textContent=t?'Stage: joint PPO · update '+t.step+'/'+t.total+' · '+(t.elapsed/60).toFixed(1)+' min · '+(t.complete?'complete':t.budget_stopped?'budget stopped':'running'):'Queue: '+(q?.status||'not queued');if(t?.logs?.length){const last=t.logs.at(-1);document.getElementById('rlResult').textContent='Latest rollout team reward: '+last.rollout.team.toFixed(4)+' · selected update: '+(t.selected_step||0);document.getElementById('rlLoss').textContent=['organ','strings','cello'].map(r=>r+' PPO loss: '+last.losses[r].loss.toFixed(4)).join(' · ')}if(q?.error)document.getElementById('rlResult').textContent=q.error}catch(e){}setTimeout(refreshRL,5000)}refreshRL();</script>'''
        body=json.dumps(snapshot()).encode() if self.path=='/api/status' else PAGE.replace('<section>',panel+'<section>',1).encode()
        self.send_response(200);self.send_header('Content-Type','application/json' if self.path=='/api/status' else 'text/html; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8779);args=parser.parse_args()
    print(f'Live training: http://127.0.0.1:{args.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
