"""Decode listener choices without mixing held-out seed groups into training."""
import argparse,hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('file',type=Path)
    p.add_argument('--run',type=Path,default=ROOT/'outputs/preference_holdout_v1')
    p.add_argument('--model-identities-disclosed',action='store_true',
        help='Record that the listener had access to the checkpoint mapping before ballot submission.')
    args=p.parse_args()
    raw=args.file.read_bytes();payload=json.loads(raw)
    if payload.get('schema') not in ['preference-holdout-v1','preference-holdout-v2']:raise ValueError('Unknown ballot schema')
    root=args.run;mapping=json.loads((root/'organizer_mapping.json').read_text())
    generation=json.loads((root/'generation.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    if payload['schema']!=protocol['schema']:raise ValueError('Ballot does not match comparison run')
    rows=[];seen=set()
    for row in payload['rows']:
        pair=row['pair'];choice=row['choice']
        if pair not in mapping or pair in seen or choice not in ['A','B','Same','Neither']:raise ValueError('Invalid or duplicate choice')
        seen.add(pair)
        if pair not in generation['pairs']:raise ValueError('Pair was not generated')
        for variant in ['A','B']:
            digest=hashlib.sha256((root/pair/(variant+'.mp3')).read_bytes()).hexdigest()
            if digest!=generation['pairs'][pair][variant]['sha256']:raise ValueError('Audio identity changed')
        selected=mapping[pair][choice] if choice in ['A','B'] else None
        rows.append(dict(pair=pair,seed=mapping[pair]['seed'],choice=choice,
            preferred_checkpoint_sha256=protocol['source_sha256'][selected] if selected is not None else None,
            preferred_step=[128,224][selected] if selected is not None else None,split='heldout_listener_evaluation'))
    decisive=[r for r in rows if r['preferred_step'] is not None];wins=sum(r['preferred_step']==128 for r in decisive);n=len(decisive)
    # Exact two-sided binomial test; no multiplication of trials by rating categories.
    pvalue=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,n-wins)+1))/2**n) if n else None
    report=dict(rows=rows,step128_wins=wins,step224_wins=n-wins,decisive_seed_groups=n,
        comparison_schema=protocol['schema'],
        model_identities_disclosed_before_submission=args.model_identities_disclosed,
        blinding_status='mapping disclosed before submission; vote timing unknown' if args.model_identities_disclosed else 'not established',
        exact_two_sided_binomial_p=pvalue,reward_validated=False,rl_training_allowed=False,
        interpretation='Small held-out checkpoint preference test; not a trained reward validation. Do not reuse these groups as fitting data.')
    out=ROOT/'outputs/listener_preferences_v3/heldout';out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(raw).hexdigest();(out/(digest+'.original.json')).write_bytes(raw)
    (out/(digest+'.analysis.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
