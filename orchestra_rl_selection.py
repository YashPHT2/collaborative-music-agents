"""Paired fixed-seed selection; participation guards are not taste rewards."""
import numpy as np
import orchestra_learning as l

VALIDATION_SEEDS=list(range(9950001,9950013))
TEST_SEEDS=list(range(9960001,9960013))

def measurements(episode):
    beats=episode['brief']['bars']*4;notes=episode['notes'];roles={}
    for r,s in l.ROLES.items():
        own=[n for n in notes if n[3]==s['id']]
        active=np.zeros(beats*16,dtype=bool)
        for n in own:active[n[1]//30:min(len(active),(n[1]+n[2])//30)]=True
        roles[r]=dict(sounding=float(active.mean()),onsets_per_beat=len(own)/beats)
    return dict(seed=episode['seed'],reward=episode['steps'][-1]['reward'],roles=roles)

def select(candidate,baseline,best_mean):
    if [r['seed'] for r in candidate]!=VALIDATION_SEEDS or [r['seed'] for r in baseline]!=VALIDATION_SEEDS:
        raise ValueError('Selection requires every predetermined validation seed in order')
    diff=np.array([a['reward']-b['reward'] for a,b in zip(candidate,baseline)])
    if not np.isfinite(diff).all():raise ValueError('Nonfinite validation rewards')
    draws=np.random.default_rng(9951).choice(diff,(2000,len(diff)),replace=True).mean(1)
    lower=float(np.quantile(draws,.025));role_checks={}
    for role in l.ROLES:
        before=np.mean([r['roles'][role]['sounding'] for r in baseline]);after=np.mean([r['roles'][role]['sounding'] for r in candidate])
        # Guard against disappearance OR flooding, allowing natural rests. This
        # compares each role to its own baseline, never rewards maximum density.
        role_checks[role]=dict(before=float(before),after=float(after),passed=abs(after-before)<=.2)
    mean=float(np.mean([r['reward'] for r in candidate]))
    return dict(passed=lower>0 and mean>best_mean and all(v['passed'] for v in role_checks.values()),
        mean_reward=mean,paired_improvement=float(diff.mean()),paired_bootstrap_lower95=lower,
        roles=role_checks,interpretation='Validation-selected proxy candidate; musical preference is still unknown')
