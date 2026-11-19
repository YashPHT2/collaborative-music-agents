"""Cache frozen-backbone teacher features for the partner-response curriculum."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import note_trio as nt
import supervised_ensemble as se
import partner_response as pr
import ensemble_data as ed

OUT=nt.ROOT/'outputs'/'partner_response_v1'
SOURCE=se.OUT/'latest.pt'


def prepare(output=OUT,source=SOURCE):
    output.mkdir(parents=True,exist_ok=True)
    recipe=dict(version=1,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                audit_sha256=hashlib.sha256((ed.DATA/'audit.json').read_bytes()).hexdigest(),
                train_windows=64,validation_windows=12,train_beats=256,validation_beats=48)
    cache_path=output/'teacher_cache.pt'
    if cache_path.exists():
        saved=torch.load(cache_path,map_location='cpu',weights_only=False)
        if saved['recipe']!=recipe:raise ValueError('Cached training data differs from this source/recipe')
        return saved['data'],saved['cache'],recipe
    torch.set_num_threads(4);torch.manual_seed(20261018)
    data=pr.response_windows()
    model,adapters,agents,critic=se.load_bundle(source)
    cache=pr.build_teacher_cache(model,adapters,data)
    payload=dict(recipe=recipe,data=data,cache=cache)
    temporary=cache_path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(cache_path)
    manifest=dict(recipe=recipe,source=str(source.resolve()),
        windows={split:{role:[dict(song=w['song'],window_start=w['window_start'],available_roles=w['available_roles'],
                                  notes_sha256=pr.fingerprint(w['notes'])) for w in ws] for role,ws in roles.items()} for split,roles in data.items()},
        usable_beats={s:{r:len(pairs) for r,pairs in roles.items()} for s,roles in cache.items()},
        cached_source_songs={s:{r:sorted({x['correct']['song'] for x in pairs}) for r,pairs in roles.items()} for s,roles in cache.items()},
        notes='Missing roles remain absent. Fixed partners are dataset parts, not agent outputs. No current/future partner notes are exposed. Final joint music is generated entirely by policies.')
    (output/'data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest['usable_beats'],indent=2),flush=True)
    return data,cache,recipe


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--source',type=Path,default=SOURCE)
    args=p.parse_args();prepare(args.output,args.source)
