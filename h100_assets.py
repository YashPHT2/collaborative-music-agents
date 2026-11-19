"""Verify portable payload, fetch pinned model assets, and export run results."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()


def verify():
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
    for name,expected in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or sha(path)!=expected:raise ValueError('Package hash mismatch: '+name)
    print('Verified',len(manifest['files']),'package files')
    return manifest


def download():
    from huggingface_hub import hf_hub_download
    pins=json.loads((ROOT/'ASSET_PINS.json').read_text())
    for item in pins['downloads']:
        path=ROOT/item['destination']
        if not path.exists() or sha(path)!=item['sha256']:
            path.parent.mkdir(parents=True,exist_ok=True)
            downloaded=Path(hf_hub_download(item['repository'],item['filename'],revision=item['revision']))
            if sha(downloaded)!=item['sha256']:raise ValueError('Remote asset differs from pinned bytes')
            import shutil
            shutil.copyfile(downloaded,path)
        print('Asset verified:',item['destination'])


def export(run,destination):
    run=Path(run).resolve();destination=Path(destination).resolve()
    if destination.is_relative_to(run):raise ValueError('Results ZIP must be outside the run folder')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(run.rglob('*')):
            if path.is_file() and not path.name.endswith('.tmp'):bundle.write(path,arcname=path.relative_to(run))
    print('Results saved:',destination,'bytes',destination.stat().st_size)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','download','export'])
    p.add_argument('--run',type=Path);p.add_argument('--destination',type=Path);args=p.parse_args()
    if args.action=='verify':verify()
    elif args.action=='download':download()
    else:
        if not args.run or not args.destination:p.error('export requires --run and --destination')
        export(args.run,args.destination)


if __name__=='__main__':main()
