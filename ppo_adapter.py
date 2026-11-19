"""Small on-policy PPO residual policy over a frozen symbolic-music generator.

Only note-vs-wait and pitch tokens are adapted. Other tokens use the unchanged
SkyTNT sampler. Exact sampling log probabilities and fixed reference support
are retained for PPO; no post-adapter top-k/top-p filtering is applied.
"""
import math
import torch
from torch import nn

OBSERVATIONS = 64
ACTIONS = 33  # note/wait followed by MIDI pitches 36..66


class ResidualPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(OBSERVATIONS,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.actor = nn.Linear(64,ACTIONS)
        self.critic = nn.Linear(64,1)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def distribution(self, observations, reference):
        hidden = self.encoder(observations)
        residual = 2.0 * torch.tanh(self.actor(hidden))
        logits = torch.where(reference>0, reference.clamp_min(1e-12).log()+residual,
                             torch.full_like(reference,float('-inf')))
        return torch.distributions.Categorical(logits=logits), self.critic(hidden).squeeze(-1)


def reference_kl(distribution, reference):
    log_reference = reference.clamp_min(1e-12).log()
    return (distribution.probs * (distribution.logits.clamp_min(-1e9)-log_reference)).sum(-1)


def gae(rewards, values, gamma=0.99, lam=0.95):
    """Complete episodes only: zero bootstrap at the terminal melody boundary."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros((),device=rewards.device)
    next_value = torch.zeros((),device=rewards.device)
    for index in range(len(rewards)-1,-1,-1):
        delta = rewards[index] + gamma*next_value-values[index]
        running = delta + gamma*lam*running
        advantages[index] = running
        next_value = values[index]
    return advantages, advantages+values


def ppo_update(policy, optimizer, episodes, generator, epochs=4, clip=0.2, target_kl=0.025):
    samples = []
    for episode in episodes:
        if not episode:
            continue
        values = torch.tensor([s['value'] for s in episode])
        rewards = torch.tensor([s['reward'] for s in episode])
        advantages,returns = gae(rewards,values)
        for sample,advantage,target in zip(episode,advantages,returns):
            samples.append((sample,float(advantage),float(target)))
    observations = torch.stack([s[0]['observation'] for s in samples])
    reference = torch.stack([s[0]['reference'] for s in samples])
    actions = torch.tensor([s[0]['action'] for s in samples])
    old_logp = torch.tensor([s[0]['logp'] for s in samples])
    advantages = torch.tensor([s[1] for s in samples])
    advantages = (advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
    returns = torch.tensor([s[2] for s in samples])
    losses = []
    early_stop = False
    for epoch in range(epochs):
        for indices in torch.randperm(len(samples),generator=generator).split(128):
            distribution,values = policy.distribution(observations[indices],reference[indices])
            logp = distribution.log_prob(actions[indices])
            log_ratio = logp-old_logp[indices]
            ratio = log_ratio.exp()
            surrogate = torch.minimum(ratio*advantages[indices],ratio.clamp(1-clip,1+clip)*advantages[indices])
            value_loss = 0.5*(values-returns[indices]).square().mean()
            kl = reference_kl(distribution,reference[indices]).mean()
            loss = -surrogate.mean()+0.5*value_loss+0.05*kl-0.005*distribution.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),0.5)
            if not torch.isfinite(grad_norm):
                raise RuntimeError('Non-finite PPO gradient')
            optimizer.step()
            approx_kl = ((ratio-1)-log_ratio).mean().detach().item()
            losses.append(dict(loss=loss.item(),reference_kl=kl.item(),approx_old_policy_kl=approx_kl,
                               clip_fraction=((ratio-1).abs()>clip).float().mean().item()))
            if approx_kl > target_kl:
                early_stop = True
                break
        if early_stop:
            break
    return dict(decisions=len(samples),early_stop=early_stop,
                **{key:sum(row[key] for row in losses)/len(losses) for key in losses[0]})


def features(reference, melody_pitch, last_pitch, notes_in_window, tick, end_tick, is_pitch, history):
    observation = torch.zeros(OBSERVATIONS)
    observation[:ACTIONS] = reference
    observation[33+melody_pitch%12] = 1
    observation[45] = (last_pitch-51)/15 if last_pitch is not None else 0
    observation[46] = min(notes_in_window/8,1)
    phase = (tick/480)%4 * math.pi/2
    observation[47:49] = torch.tensor([math.sin(phase),math.cos(phase)])
    observation[49] = max(0,1-tick/end_tick)
    observation[50+int(is_pitch)] = 1
    for pitch in history[-16:]:
        observation[52+pitch%12] += 1/max(1,len(history[-16:]))
    return observation
