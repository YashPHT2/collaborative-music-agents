"""Package a portable, offline team listening ballot without exposing localhost."""
import base64
import hashlib
import json
from pathlib import Path
import random
import re
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / "outputs" / "evaluation_v1"
    output = ROOT / "outputs" / "team_handoff"
    output.mkdir(parents=True, exist_ok=True)
    page = (source / "listen.html").read_text(encoding="utf-8")
    original = json.loads(re.search(r"const pairs=(.*?);\nconst storageKey", page, re.S).group(1))
    pairs, audit, excluded = [], [], []
    candidates = []
    for pair in original:
        directory = source / pair["id"] / f"seed{pair['seed']}"
        responding = json.loads((directory / "responding_notes.json").read_text())
        frozen = json.loads((directory / "frozen_notes.json").read_text())
        signature = lambda notes: sorted(tuple(note[1:]) for note in notes)
        if signature(responding) == signature(frozen):
            excluded.append(pair["id"])
        else:
            candidates.append(pair)
    # Balance A/B placement for this new team ballot. These labels can differ
    # from the development listening page; exports carry the exact mapping.
    placement = ["responding"] * (len(candidates)//2) + ["frozen"] * (len(candidates)-len(candidates)//2)
    random.Random(173).shuffle(placement)
    for number, (pair, a_method) in enumerate(zip(candidates, placement), 1):
        files = {pair["methods"]["A"]: pair["a"], pair["methods"]["B"]: pair["b"]}
        pair["methods"] = dict(A=a_method, B="frozen" if a_method == "responding" else "responding")
        mapping = {}
        for label in ("A", "B"):
            path = source / files[pair["methods"][label]]
            data = path.read_bytes()
            pair[label.lower()] = "data:audio/mpeg;base64," + base64.b64encode(data).decode("ascii")
            mapping[label] = dict(method=pair["methods"][label],sha256=hashlib.sha256(data).hexdigest())
        pair["number"] = number
        pairs.append(pair)
        audit.append(dict(number=number,case_id=pair["id"],seed=pair["seed"],mapping=mapping))
    page = re.sub(r"const pairs=.*?;\nconst storageKey", lambda _: "const pairs=" + json.dumps(pairs) + ";\nconst storageKey", page, flags=re.S)
    page = page.replace("rl-piano-evaluation-v1", "rl-piano-team-ballot-v1")
    page = page.replace("Piano accompaniment listening test", "Team piano listening vote")
    page = page.replace("<h1>Piano accompaniment: what sounds better?</h1>",
                        "<h1>Team vote: which accompaniment sounds better?</h1>"
                        "<p><b>7 pairs · about 5–10 minutes · headphones recommended.</b> "
                        "All audio is embedded in this file. No server, login, or installation needed.</p>"
                        "<p>Focus on the lower piano part after 9.6 seconds. Choose the version you prefer, "
                        "judging fit with the melody, smoothness and excessive busyness. "
                        "Choose Tie if you hear no meaningful difference, or Neither good if both sound poor. "
                        "Rate independently before discussing with the team or revealing labels.</p>"
                        "<p><label for='participant'>Your name or initials (optional): </label>"
                        "<input id='participant' maxlength='60' autocomplete='off'></p>")
    page = page.replace('<button id="export">Export my ratings</button>',
                        '<button id="export">Download my votes</button>'
                        '<button id="whatsapp">Show votes to send on WhatsApp</button>')
    page = page.replace('<p id="status" role="status"></p>',
                        '<p id="status" role="status"></p>'
                        '<textarea id="voteText" aria-label="Votes to copy into WhatsApp" rows="11" '
                        'readonly hidden style="width:100%;box-sizing:border-box"></textarea>')
    page = re.sub(r'<p><a href="REPORT.md">.*?</p>',
                  '<p class="muted">This is an exploratory baseline comparison; neither option uses our RL training. '
                  'The identical test pair was excluded. Votes are stored locally where supported and are only '
                  'shared when you send the downloaded file or copy your vote text. A/B labels belong to this team ballot.</p>',page)
    page = page.replace("pair[label.toLowerCase()]+'?playback=2'", "pair[label.toLowerCase()]")
    page = page.replace("title.textContent=(pair.id.startsWith('theme_a')?", "title.textContent=pair.number+'. '+(pair.id.startsWith('theme_a')?")
    page = page.replace("const storageKey='rl-piano-team-ballot-v1';let ratings={};",
                        "const storageKey='rl-piano-team-ballot-v1';let ratings={};")
    page = page.replace("created_at:new Date().toISOString(),ratings:",
                        "ballot_id:'team-v1',participant:document.getElementById('participant').value.trim(),created_at:new Date().toISOString(),ratings:")
    page = page.replace("a.download='listening-ratings.json';", "a.download='team-votes-'+(document.getElementById('participant').value.trim().replace(/[^a-zA-Z0-9_-]/g,'_')||'anonymous')+'.json';")
    page = page.replace("</script>", """
document.getElementById('whatsapp').onclick=()=>{
 const name=document.getElementById('participant').value.trim()||'Anonymous';
 const voted=pairs.filter(p=>ratings[p.id]).length;
 const text=['Piano team ballot v1','Voter: '+name,'Completed: '+voted+'/'+pairs.length,
 ...pairs.map(p=>p.number+'. '+p.id+': '+(ratings[p.id]||'Not rated'))].join('\\n');
 const area=document.getElementById('voteText');area.hidden=false;area.value=text;area.focus();area.select();
 document.getElementById('status').textContent='Copy the selected text and send it in WhatsApp.';
};
</script>""")
    (output / "OPEN_ME.html").write_text(page,encoding="utf-8")
    instructions = """PIANO PROJECT — TEAM LISTENING VOTE

1. Download the ZIP to a laptop/desktop and extract it completely.
2. Open OPEN_ME.html in Chrome, Edge, Safari or Firefox.
   The audio is inside the HTML file; no Python, server or installation needed.
3. Use headphones if available. There are seven A/B pairs, about 5–10 minutes.
4. Both versions have the same higher melody. Focus on the lower accompaniment
   AFTER 9.6 seconds. Use Start at change under A, then under B.
5. Judge fit with the melody, smoothness and whether the part is too busy.
   A or B = clear preference; Tie = equal/no audible difference;
   Neither good = both poor. Do not force a winner or reward change alone.
6. Vote independently. Do not reveal the labels or discuss preferences first.
7. Add your name/initials, then click Download my votes and send the JSON file
   in WhatsApp. Alternatively click Show votes to send on WhatsApp, copy the
   selected text, and send that. Votes are NOT automatically sent anywhere.

One identical pair was excluded. Neither candidate uses our project's RL
training; this vote checks the pretrained accompaniment baseline. A/B labels
belong to THIS ballot and may differ from earlier listening-page labels.

If the HTML does not open in WhatsApp's preview, extract it on your laptop and
open it directly in your normal browser. Export your votes before closing.
"""
    (output / "READ_ME.txt").write_text(instructions,encoding="utf-8")
    message = """Hey team! Please help evaluate our piano accompaniment baseline 🎹

Download the attached ZIP on a laptop, extract it, and open OPEN_ME.html in your browser. No installation needed.

There are 7 A/B comparisons (~5–10 min). Click “Start at change” for each version and focus on the LOWER piano part after 9.6 seconds.

Judge: Does it fit the melody? Does it flow smoothly? Is it too busy?

Vote A / B / Tie / Neither good. Tie is completely valid if you cannot hear a meaningful difference. Please vote independently and don't reveal labels or discuss choices until finished.

Add your name, click “Download my votes”, and send the downloaded file here. Or click “Show votes to send on WhatsApp” and paste the text here.

These are pretrained baseline samples, before our RL training. We're finding what actually needs improvement.
"""
    (output / "WHATSAPP_MESSAGE.txt").write_text(message,encoding="utf-8")
    archive = output / "Piano_Team_Vote_v1.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as bundle:
        for name in ("OPEN_ME.html", "READ_ME.txt"):
            bundle.write(output/name, "Piano_Team_Vote/"+name)
    # Keep the mapping outside the participant ZIP for analysis of returned votes.
    (output / "organizer_manifest.json").write_text(json.dumps(dict(ballot_id="team-v1",pairs=audit,excluded_identical=excluded),indent=2)+"\n")
    print(json.dumps(dict(archive=str(archive),megabytes=round(archive.stat().st_size/1024**2,2),
                          pairs=len(pairs),audio_clips=len(pairs)*2,excluded=excluded),indent=2))


if __name__ == "__main__":
    main()
