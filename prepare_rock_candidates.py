"""Rebuild the rock metadata candidate manifest used by the instrument audit."""
import argparse,csv,json,re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def collect(path):
    counts=Counter();candidates=[]
    with Path(path).open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if 'rock' not in re.split('[,;| ]+',row.get('genres','').lower()):continue
            counts['genre_rock']+=1
            positive=bool(re.search(r'\b(happy|uplifting|upbeat|joyful)\b',row.get('tags','').lower()))
            counts['rock_with_positive_tag']+=positive
            counts['local_rock_midi']+=0
            candidates.append(dict(id=row['path'],genre=row['genres'],tags=row['tags'],title=row['title'],mid=row['mid'],
                local_midi=None,positive_tag=positive,license=row['license'],license_conflict=row['license_conflict'],tracks=row['tracks']))
    return dict(counts=dict(counts),candidates=candidates,note='Metadata tags are candidate evidence, not confirmed mood or ensemble labels. No training data is approved by this audit.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,default=ROOT/'assets/composition_v1/PDMX.csv')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/rock_training_v1');p.add_argument('--download',action='store_true');a=p.parse_args()
    if a.download:
        from composition_data import fetch
        fetch('PDMX.csv');fetch('mid.tar.gz')
    if not a.csv.is_file():raise FileNotFoundError('Supply --csv or use --download to fetch the pinned dataset metadata/archive')
    a.output.mkdir(parents=True,exist_ok=True);target=a.output/'pdmx_rock_candidates.json';report=collect(a.csv)
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8'))!=report:raise ValueError('Existing candidate manifest differs; preserve the run and choose another output')
    else:target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['counts'],indent=2))

if __name__=='__main__':main()
