"""Version 3: phrase imitation, explicit event diagnostics, causal bar plans.

Warm-build is a plan control, not an emotion classifier or a note arranger.
"""
import copy
from collections import Counter
import numpy as np
import torch
from torch import nn
import note_trio as nt
import offline_ensemble as oe
import supervised_ensemble as se

VERSION = 3
PLAN_SIZE = 48
ROLES = oe.ROLES
CATEGORIES = ('first_event', 'additional_event', 'pitch', 'onset', 'duration', 'velocity')


def beat_kind(w, role, beat):
    if role not in w['available_roles']:
        return 'excluded_low_support' if role in w.get('source_roles', []) else 'missing'
    ns = [n for n in w['notes'] if n[3] == ROLES.index(role)]
    if any(beat*480 <= n[1] < (beat+1)*480 for n in ns):
        return 'onset'
    if any(n[1] < (beat+1)*480 and n[1]+n[2] > beat*480 for n in ns):
        return 'sustain'
    return 'silence'


def description(notes, bar, total=8):
    base = oe.description(notes, bar, total)
    rhythm = torch.zeros(16)
    for n in notes:
        if bar*1920 <= n[1] < (bar+1)*1920:
            rhythm[(n[1] % 1920)//120] += 1
    rhythm /= rhythm.sum().clamp_min(1)
    return torch.cat([base, rhythm])


class Planner(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(PLAN_SIZE*2+1, 128, 2, batch_first=True)
        self.head = nn.Linear(128, PLAN_SIZE)

    def forward(self, x, hidden=None):
        y, h = self.gru(x, hidden)
        return self.head(y).sigmoid(), h


class Conditions(oe.Conditions):
    def __init__(self):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(PLAN_SIZE+131, 256), nn.Tanh(), nn.Linear(256, 1024))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)


def upgrade(saved, device):
    """Copy v2 weights exactly, zero only new input columns."""
    planner, conditions = Planner().to(device), Conditions().to(device)
    oldp, oldc = saved['planner'], saved['conditions']
    p, c = planner.state_dict(), conditions.state_dict()
    for k in p:
        if p[k].shape == oldp[k].shape:
            p[k] = oldp[k]
    p['gru.weight_ih_l0'].zero_()
    p['gru.weight_ih_l0'][:, :32] = oldp['gru.weight_ih_l0'][:, :32]
    p['gru.weight_ih_l0'][:, 48:80] = oldp['gru.weight_ih_l0'][:, 32:64]
    p['gru.weight_ih_l0'][:, -1] = oldp['gru.weight_ih_l0'][:, -1]
    p['head.weight'][:32] = oldp['head.weight']
    p['head.bias'][:32] = oldp['head.bias']
    p['head.weight'][32:].zero_()
    p['head.bias'][32:] = -2.70805  # initial 1/16 onset mass
    for k in c:
        if c[k].shape == oldc[k].shape:
            c[k] = oldc[k]
    c['out.0.weight'].zero_()
    c['out.0.weight'][:, :32] = oldc['out.0.weight'][:, :32]
    c['out.0.weight'][:, 48:] = oldc['out.0.weight'][:, 32:]
    planner.load_state_dict(p)
    conditions.load_state_dict(c)
    return planner, conditions


def plan_segment(planner, ledger, start, bars, previous=None, hidden=None, profile=None, quantiles=None):
    """Identical two-bar scheduling for teaching, free rollout and response probes."""
    device = next(planner.parameters()).device
    previous = torch.zeros(PLAN_SIZE, device=device) if previous is None else previous
    summary = description(ledger, start//4-1, bars).to(device) if start else torch.zeros_like(previous)
    plans = []
    for bar in range(start//4, min(bars, start//4+2)):
        phase = bar/max(1, bars-1)
        pred, hidden = planner(torch.cat([previous, summary, previous.new_tensor([phase])])[None, None], hidden)
        previous = pred[0, 0]
        if profile == 'warm_build':
            if quantiles is None:
                raise ValueError('warm_build needs training-only quantiles')
            previous = previous.clone()
            for index in range(12, 21):
                low, high = quantiles[str(index)]
                previous[index] = low+(high-low)*phase
        elif profile is not None:
            raise ValueError('Unknown arrangement profile')
        plans.append(previous)
    return plans, previous, hidden


def plan_sequence(planner, notes, bars):
    plans, previous, hidden = [], None, None
    for start in range(0, bars*4, 8):
        part, previous, hidden = plan_segment(planner, nt.visible(notes, start), start, bars, previous, hidden)
        plans.extend(part)
    return torch.stack(plans)


def category_losses(model, example, training=False):
    """Keep event stopping distinct from note attributes and first onset."""
    x = example['x'].to(model.device)[None]
    hidden = model(x)[0, example['positions'].to(model.device)]
    y = example['labels'].to(model.device)
    logits = model.forward_token(hidden, y[:, :-1]).float()
    support = example['support'].to(model.device)
    return category_losses_from_logits(logits, y, support, training)


def batch_category_losses(model, examples, deltas, training=False,return_pitch=False):
    """Right-pad causal contexts; gather only real positions before token decoding."""
    if not examples or len(examples) != len(deltas):
        raise ValueError('A condition is required for every microbatch example')
    size = max(len(ex['x']) for ex in examples)
    x = torch.full((len(examples), size, 8), model.tokenizer.pad_id, dtype=torch.long, device=model.device)
    for i, ex in enumerate(examples):
        x[i, :len(ex['x'])] = ex['x'].to(model.device)
    # Right padding is strictly after every selected causal position, so no real
    # token can attend to padding even though the backbone has no padding API.
    states = model(x)
    hidden, labels, sizes = [], [], []
    for i, (ex, delta) in enumerate(zip(examples, deltas)):
        positions = ex['positions'].to(model.device)
        if positions.min() < 0 or positions.max() >= len(ex['x']):
            raise ValueError('Selected position points into padding')
        hidden.append(states[i, positions]+delta.to(states.dtype))
        labels.append(ex['labels'].to(model.device)); sizes.append(len(positions))
    y = torch.cat(labels)
    logits = model.forward_token(torch.cat(hidden), y[:, :-1]).float()
    result = []; offset = 0;pitch_stats=[]
    for ex, size in zip(examples, sizes):
        result.append(category_losses_from_logits(logits[offset:offset+size], y[offset:offset+size],
                                                  ex['support'].to(model.device), training))
        if return_pitch:
            pitch_ids=model.tokenizer.parameter_ids['pitch']
            ids=torch.tensor([pitch_ids[p] for p in range(128)],device=model.device)
            support=ex['support'].to(model.device)[:,5,ids]
            valid=support.any(-1)
            chosen=logits[offset:offset+size,5][:,ids][valid]
            probabilities=torch.softmax((chosen/.9).masked_fill(~support[valid],-torch.inf),-1)
            pitches=torch.arange(128,device=model.device,dtype=torch.float32)
            prediction=(probabilities*pitches).sum(-1)
            labels=y[offset:offset+size,5][valid]
            target=(labels[:,None]==ids[None,:]).float().argmax(-1).float()
            pitch_stats.append((prediction,target))
        offset += size
    return (result,pitch_stats) if return_pitch else result


def category_losses_from_logits(logits, y, support, training=False):
    valid = support.any(-1)
    # Avoid NaN gradients on deterministic/padded positions with no legal mask.
    safe = support.clone()
    safe[..., 0] |= ~valid
    log_probabilities = torch.log_softmax((logits/.9).masked_fill(~safe, -torch.inf), -1)
    if training:
        # The exploration floor otherwise kills gradients on confident wrong rests.
        nll = -log_probabilities.gather(-1, y[..., None]).squeeze(-1)
    else:
        probabilities = .98*log_probabilities.exp()+.02*safe/safe.sum(-1, keepdim=True)
        nll = -probabilities.gather(-1, y[..., None]).squeeze(-1).clamp_min(1e-30).log()
    result = {}
    for category, column in [('pitch', 5), ('onset', 2), ('duration', 7), ('velocity', 6)]:
        if valid[:, column].any():
            result[category] = nll[:, column][valid[:, column]].mean()
    if valid[0, 0]:
        result['first_event'] = nll[0, 0]
    if valid[1:, 0].any():
        result['additional_event'] = nll[1:, 0][valid[1:, 0]].mean()
    return result


def teacher(model, conditions, w, role, beat, plans, drafts=None):
    if role not in w['available_roles']:
        return None
    start = beat//8*3840
    partners = [n for n in (drafts or []) if n[3] != ROLES.index(role) and start <= n[1] < start+3840]
    delta = conditions(plans[beat//4], oe.draft_features(partners, role, start).to(model.device), role)
    wrapped = oe.ConditionedModel(model, delta)
    with oe.limits():
        example = se.teacher_example(wrapped, role, {**w, 'beat': beat, 'sampler_type': oe.Sampler})
    return None if example is None else category_losses(wrapped, example)


def phrase_choice(w, role, rng):
    strata = [[b for b in range(w['bars']*4) if (beat_kind(w, role, b) == 'onset') == active]
              for active in (True, False)]
    choices = [s for s in strata if s]
    return rng.choice(rng.choice(choices))//8*8


def phrase_weights(w, role, start):
    kinds = [beat_kind(w, role, b) == 'onset' for b in range(start, min(start+8, w['bars']*4))]
    counts = Counter(kinds)
    return [1/(len(counts)*counts[k]) for k in kinds]


def aggregate(rows):
    """Macro averages: equal compositions and onset/non-onset strata."""
    result = {}
    for category in CATEGORIES:
        buckets = {}
        for row in rows:
            if category in row['losses']:
                key = (row['group'], row['kind'] == 'onset')
                buckets.setdefault(key, []).append(row['losses'][category])
        per_group = {}
        for (group, active), values in buckets.items():
            per_group.setdefault(group, []).append(sum(values)/len(values))
        values = [sum(v)/len(v) for v in per_group.values()]
        result[category] = sum(values)/len(values) if values else None
    events = [result[k] for k in CATEGORIES[:2] if result[k] is not None]
    attributes = [result[k] for k in CATEGORIES[2:] if result[k] is not None]
    result['event'] = sum(events)/len(events) if events else None
    result['attributes'] = sum(attributes)/len(attributes) if attributes else None
    result['strata_counts'] = dict(Counter(r['kind'] for r in rows))
    result['by_kind'] = {kind: {k: float(np.mean([r['losses'][k] for r in rows if r['kind'] == kind and k in r['losses']]))
                                for k in CATEGORIES if any(r['kind'] == kind and k in r['losses'] for r in rows)}
                         for kind in ('onset', 'sustain', 'silence')}
    return result


def eligible(scores, initial):
    return all(scores[r]['event'] is not None and scores[r]['attributes'] is not None
               and scores[r]['event'] < initial[r]['event']
               and scores[r]['attributes'] <= initial[r]['attributes']*1.02 for r in ROLES)


@torch.no_grad()
def compose(model, adapters, agents, planner, conditions, bars=8, seed=9619101, mode='planned',
            profile=None, quantiles=None, order=None, prefix=None):
    if mode not in ('planned', 'revised'):
        raise ValueError('v3 mode must be planned or revised')
    order = order or ROLES
    if sorted(order) != sorted(ROLES):
        raise ValueError('Each role must appear once')
    brief = nt.Brief(bars)
    ledger, decisions, traces, allplans = copy.deepcopy(prefix or []), [], [], []
    initial = 8 if prefix is not None else 0
    if any(n[1] >= initial*480 for n in ledger):
        raise ValueError('Prefix contains future notes')
    previous, hidden = None, None
    device = model.device
    # Seeded prefixes are used only for training/probes, never the listening panel.
    with oe.limits():
        for start in range(0, bars*4, 8):
            plans, previous, hidden = plan_segment(planner, nt.visible(ledger, start), start, bars, previous, hidden, profile, quantiles)
            allplans.extend([p.cpu().tolist() for p in plans])
            if start < initial:
                continue
            def segment(drafts, revision):
                actions, records = {}, []
                for role in order:
                    adapters.activate(role)
                    own, made = copy.deepcopy(ledger), []
                    partner = [n for n in drafts if n[3] != ROLES.index(role)]
                    features = oe.draft_features(partner, role, start*480).to(device)
                    for beat in range(start, min(start+8, bars*4)):
                        delta = conditions(plans[(beat-start)//4], features, role)
                        history = nt.visible(own, beat)
                        sampler = oe.Sampler(oe.ConditionedModel(model, delta), role,
                                             seed+beat*1009+ROLES.index(role)*100003+revision*700001, brief, beat, history)
                        available = bool(sampler.onsets())
                        notes, samples = sampler.propose(agents[role], nt.state_features(brief, beat, history))
                        records.append(dict(role=role, beat=beat, legal_onset=available, proposed=len(notes),
                                            stopped=any(int(r['action']) == 1 and r['observation'][1056] == 1 for r in samples)))
                        own = nt.commit(own, {r: notes if r == role else [] for r in ROLES}, beat, brief)
                        made.extend(notes)
                    actions[role] = made
                return sorted(sum(actions.values(), []), key=lambda n: (n[1], n[3], n[4])), records
            draft, draft_records = segment([], 0)
            final, final_records = segment(draft, 1) if mode == 'revised' else (draft, draft_records)
            for beat in range(start, min(start+8, bars*4)):
                actions = {r: [n for n in final if n[3] == i and n[1]//480 == beat] for i, r in enumerate(ROLES)}
                ledger = nt.commit(ledger, actions, beat, brief)
                for role, ns in actions.items():
                    decisions.extend(dict(role=role, beat=beat, note=n, origin='v3_'+mode) for n in ns)
            traces.append(dict(start_beat=start, draft=draft, committed=final, draft_records=draft_records, records=final_records))
    assert sorted(ledger) == sorted([d['note'] for d in decisions]+list(prefix or []))
    return dict(notes=ledger, decisions=decisions, trace=traces, plans=allplans, brief=nt.asdict(brief),
                seed=seed, dataset_prefix=prefix or [], profile=profile, mode=mode, format_version=VERSION)


@torch.no_grad()
def fixed_response(model, adapters, agents, planner, conditions, w, role, seed, context='correct'):
    if context not in ('correct', 'missing', 'mismatched'):
        raise ValueError('Unknown context')
    channel = ROLES.index(role)
    ledger = nt.visible(w['notes'], 8)
    fixed = [copy.deepcopy(n) for n in w['notes'] if n[3] != channel and 3840 <= n[1] < 7680]
    observed = copy.deepcopy(ledger)
    def intervention(notes):
        result = copy.deepcopy(notes)
        if context == 'missing':
            result = [n for n in result if n[3] == channel]
        elif context == 'mismatched':
            for n in result:
                if n[3] != channel:
                    n[4] = min(127, n[4]+6)
        return result
    _, prev, hidden = plan_segment(planner, [], 0, w['bars'])
    plans, _, _ = plan_segment(planner, intervention(observed), 8, w['bars'], prev, hidden)
    partners = intervention(fixed)
    features = oe.draft_features(partners, role, 3840).to(model.device)
    made = []
    adapters.activate(role)
    with oe.limits():
        for beat in range(8, 16):
            history = intervention(nt.visible(ledger, beat))
            delta = conditions(plans[(beat-8)//4], features, role)
            sampler = oe.Sampler(oe.ConditionedModel(model, delta), role, seed+beat*1009, nt.Brief(w['bars']), beat, history)
            notes, _ = sampler.propose(agents[role], nt.state_features(nt.Brief(w['bars']), beat, history))
            actions = {r: [n for n in fixed if n[3] == i and n[1]//480 == beat] for i, r in enumerate(ROLES)}
            actions[role] = notes
            ledger = nt.commit(ledger, actions, beat, nt.Brief(w['bars']))
            made.extend(notes)
    assert sorted(n for n in ledger if n[3] != channel and n[1] >= 3840) == sorted(fixed)
    return ledger, made


def participation(notes, bars=8):
    result = {}
    for i, role in enumerate(ROLES):
        own = [n for n in notes if n[3] == i]
        occupied = np.zeros(bars*64, dtype=bool)
        for n in own:
            occupied[max(0,n[1]//30):min(len(occupied),(n[1]+n[2])//30)] = True
        longest = running = 0
        for active in occupied:
            running = 0 if active else running+1
            longest = max(longest, running)
        phrases = [sorted((n[1]-b*1920, n[2], n[4], n[5]) for n in own if b*1920 <= n[1] < (b+1)*1920) for b in range(bars)]
        nonempty = [p for p in phrases if p]
        exact = sum(bool(phrases[b]) and phrases[b] == phrases[b-1] for b in range(1,bars))
        intervals = [[p[j+1][2]-p[j][2] for j in range(len(p)-1)] for p in nonempty]
        motifs = [tuple(p[j:j+3]) for p in intervals for j in range(len(p)-2)]
        result[role] = dict(notes=len(own), onsets_per_beat=len(own)/(bars*4), sounding_beats=float(occupied.sum()/16),
                            longest_silence_beats=longest/16, exact_adjacent_bar_loops=exact,
                            recurring_interval_trigrams=len(motifs)-len(set(motifs)))
    return result


def load(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('format_version') != VERSION or saved.get('plan_size') != PLAN_SIZE:
        raise ValueError('Expected version-3 repair checkpoint')
    if saved['constraints'] != dict(strings_cap=4, max_duration_ticks=3840):
        raise ValueError('Incompatible sampling constraints')
    model = nt.load_model()
    adapters = se.InstrumentAdapters(model)
    adapters.load(saved['adapters'])
    planner, conditions = Planner().to(model.device), Conditions().to(model.device)
    planner.load_state_dict(saved['planner']); conditions.load_state_dict(saved['conditions'])
    planner.eval(); conditions.eval()
    return model, adapters, {r: oe.Musician().eval() for r in ROLES}, planner, conditions
