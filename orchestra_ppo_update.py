"""Device-aware PPO with whole-beat action ratios and explicit stability metrics."""
import torch
from ppo_adapter import gae


def update(actors,critic,optimizers,critic_optimizer,episodes):
    samples=[]
    for ep in episodes:
        steps=ep['steps'];adv,ret=gae(torch.tensor([s['reward'] for s in steps]),torch.tensor([s['value'] for s in steps]))
        samples.extend(zip(steps,adv,ret))
    advantage=torch.stack([s[1] for s in samples]);advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-6)
    logs={}
    for role,actor in actors.items():
        device=next(actor.parameters()).device;rows=[];groups=[];old=[];targets=[]
        for i,(step,_,_) in enumerate(samples):
            records=step['records'][role]
            if not records:continue
            groups.extend([len(old)]*len(records));rows.extend(records)
            old.append(sum(r['logp'] for r in records));targets.append(advantage[i])
        if not rows:
            logs[role]=dict(skipped=True,reason='No unconstrained decisions; sustained notes/rest constraints are allowed');continue
        obs=torch.stack([r['observation'] for r in rows]).to(device);refs=torch.stack([r['reference'] for r in rows]).to(device)
        acts=torch.tensor([r['action'] for r in rows],device=device);groups=torch.tensor(groups,device=device)
        old=torch.tensor(old,device=device);targets=torch.stack(targets).to(device);history=[]
        for epoch in range(4):
            logps=[];kls=[];entropies=[]
            for ix in torch.arange(len(rows),device=device).split(256):
                d=actor.distribution(obs[ix],refs[ix]);logps.append(d.log_prob(acts[ix]));entropies.append(d.entropy())
                kls.append((d.probs*(d.logits.clamp_min(-1e9)-refs[ix].clamp_min(1e-30).log())).sum(-1))
            logp=torch.zeros(len(old),device=device).index_add(0,groups,torch.cat(logps));delta=logp-old
            ratio=delta.exp();kl=((ratio-1)-delta).mean()
            if not torch.isfinite(ratio).all():raise ValueError('PPO ratio overflow; retain prior complete checkpoint')
            if epoch and float(kl.detach())>.02:break
            reference_kl=torch.cat(kls).mean();entropy=torch.cat(entropies).mean()
            loss=-torch.minimum(ratio*targets,ratio.clamp(.8,1.2)*targets).mean()+.05*reference_kl-.005*entropy
            if not torch.isfinite(loss):raise ValueError('Nonfinite actor loss')
            optimizers[role].zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),.5,error_if_nonfinite=True);optimizers[role].step()
            history.append(dict(loss=float(loss.detach()),old_kl=float(kl.detach()),reference_kl=float(reference_kl.detach()),
                entropy=float(entropy.detach()),clip_fraction=float(((ratio-1).abs()>.2).float().mean()),gradient_norm=float(norm)))
        logs[role]={**history[-1], 'tokens':len(rows),'beats':len(old),'epochs':len(history),'device':str(device)}
    device=next(critic.parameters()).device
    states=torch.stack([s[0]['critic_state'] for s in samples]).to(device);returns=torch.stack([s[2] for s in samples]).to(device)
    for _ in range(4):
        values=critic(states);loss=.5*(values-returns).square().mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite critic loss')
        critic_optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(),.5,error_if_nonfinite=True);critic_optimizer.step()
    logs['critic_loss']=float(loss.detach())
    with torch.no_grad():
        variance=returns.var(unbiased=False)
        logs['critic_explained_variance']=float(1-(returns-critic(states)).var(unbiased=False)/variance) if variance>1e-8 else None
    return logs
