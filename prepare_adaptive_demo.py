"""Create a transparent listening view for the adaptive band artifact."""
from pathlib import Path
import json
from band import ROOT
from adaptive_performance import OUT

def main():
    root=OUT
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>Adaptive band · 30 seconds</title><style>body{font:18px system-ui;background:#101a29;color:#edf4ff;max-width:900px;margin:36px auto;padding:20px}section{background:#203149;border-radius:16px;padding:24px;margin:18px 0}button,select{font:inherit;padding:10px;margin:6px;border-radius:8px;border:0}audio{width:100%;margin:15px 0}canvas{width:100%;background:#0b121f;border-radius:8px}small{color:#b7c7dd}a{color:#9fdbff}#stage{font-weight:bold;color:#a3e8ce}</style>
    <h1>Three musicians. One changing idea.</h1><p>A 30-second piano, bass and drum performance. Listen for the change at <strong>14.4 seconds</strong>.</p>
    <section><label>Performance <select id="version"><option value="performance">Responsive band</option><option value="held_intent_control">Followers keep their earlier musical brief</option></select></label>
    <label>Part <select id="stem"><option value="mix">Full band</option><option value="piano">Piano</option><option value="bass">Bass</option><option value="drums">Drums</option></select></label>
    <audio id="player" controls preload="metadata" src="performance_mix.mp3"></audio><button id="change">Listen from the change</button><p id="stage">Opening: the piano introduces its motif.</p><canvas id="roll" width="850" height="220"></canvas><p><small>Blue: piano · Gold: bass · Purple: drums</small></p></section>
    <section><h2>What to listen for</h2><p>0–14.4s: a recurring piano idea, low bass, and restrained percussion.</p><p>14.4s: the piano moves up two semitones and develops the motif rhythm. It announces its bar plan to the followers.</p><p>The responsive bass and drums generate fresh parts from that plan. The control keeps the exact same piano while withholding the changed brief from the followers.</p><p><a href="performance_mix.mid">Download MIDI</a> · <a href="performance_mix.mp3">Download mix</a> · <a href="REPORT.md">Implementation and limits</a></p></section>
    <details><summary>How this performance was made</summary><p>Three independently trained residual policies share a frozen pretrained MIDI model. A rhythmic prior is fitted on human Groove MIDI Dataset training recordings. Phrase reprises, rhythmic development, role masks and the scheduled transition are arrangement scaffolding. This shows response to an explicit musical plan; learned superiority and fully emergent song structure have not been established.</p></details>
    <script>const audio=document.getElementById('player'),version=document.getElementById('version'),stem=document.getElementById('stem');let notes=[];let end=23040;
    async function loadNotes(){notes=await(await fetch(version.value+'_events.json')).json();end=Math.max(23040,...notes.map(n=>n[1]+n[2]))}loadNotes();
    function switchTrack(){const t=audio.currentTime;audio.pause();audio.src=version.value+'_'+stem.value+'.mp3';audio.addEventListener('loadedmetadata',()=>audio.currentTime=Math.min(t,audio.duration||0),{once:true});audio.load();loadNotes()};version.onchange=stem.onchange=switchTrack;
    document.getElementById('change').onclick=async()=>{try{if(audio.readyState<1)await new Promise((resolve,reject)=>{audio.addEventListener('loadedmetadata',resolve,{once:true});audio.addEventListener('error',reject,{once:true});audio.load()});audio.currentTime=14.4;await audio.play()}catch{document.getElementById('stage').textContent='Use the audio controls to start playback.'}};
    const canvas=document.getElementById('roll'),ctx=canvas.getContext('2d');function draw(){ctx.clearRect(0,0,850,220);for(const n of notes){ctx.fillStyle=n[3]===0?'#82cdff':n[3]===1?'#ffd180':'#d5a0ff';ctx.fillRect(n[1]/end*850,(90-n[4])*3,Math.max(2,n[2]/end*850),5)}ctx.fillStyle='#a3e8ce';ctx.fillRect(14.4/28.8*850,0,1,220);ctx.fillStyle='white';ctx.fillRect(audio.currentTime/28.8*850,0,2,220);document.getElementById('stage').textContent=audio.currentTime<14.4?'Opening: the piano introduces its motif.':audio.currentTime<24?'Change: piano develops; bass and drums receive its new bar plan.':'Ending: the piano resolves the phrase.';requestAnimationFrame(draw)}draw();</script></html>'''
    (root/'listen.html').write_text(page,encoding='utf-8')
    print(root/'listen.html')

if __name__=='__main__':main()
