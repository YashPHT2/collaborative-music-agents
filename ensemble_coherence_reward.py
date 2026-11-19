"""Temporal, role-specific MIDI preference head trained on structural contrasts.

Labels are synthetic (unaltered score preferred to a disrupted part), not human
taste labels. Absolute score is uncalibrated. Missing roles receive no loss.
"""
import copy
import hashlib
import numpy as np
import torch
from torch import nn

ROLES = ('organ', 'strings', 'cello')
TICKS = 480
FEATURES = 19
KINDS = ('detune', 'desync', 'scramble')


def seed_for(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16)


def disrupt(notes, role, kind, end, seed):
    result = copy.deepcopy(notes)
    rng = np.random.default_rng(seed)
    selected = [n for n in result if n[3] == role]
    if kind == 'detune':
        shift = int(rng.choice([-3, -2, -1, 1, 2, 3]))
        for n in selected:
            n[4] = (n[4] + shift - 24) % 72 + 24
    elif kind == 'desync':
        shift = int(rng.choice([120, 240, 360]))
        for n in selected:
            n[1] = (n[1] + shift) % end
            n[2] = min(n[2], end - n[1])
    elif kind == 'scramble':
        # Bijection retains pitch support and exact same-pitch overlap pattern.
        # Unequal pitch frequencies can change: not a histogram-preserving edit.
        pitches = sorted({n[4] for n in selected})
        order = list(rng.permutation(pitches))
        if order == pitches and len(order) > 1:
            order = order[1:] + order[:1]
        mapping = dict(zip(pitches, order))
        for n in selected:
            n[4] = int(mapping[n[4]])
    else:
        raise ValueError(kind)
    return sorted(result, key=lambda n: (n[1], n[3], n[4]))


def sequence(notes, bars=8, transpose=0):
    frames = bars * 16
    x = np.zeros((frames, 3, FEATURES), dtype=np.float32)
    for role in range(3):
        ns = [n for n in notes if n[3] == role]
        if not ns:
            continue
        x[:, role, 18] = 1  # known present, distinct from a rest
        for frame in range(frames):
            tick = frame * 120
            sounding = [n for n in ns if n[1] < tick + 120 and n[1] + n[2] > tick]
            onsets = [n for n in ns if tick <= n[1] < tick + 120]
            if sounding:
                for n in sounding:
                    x[frame, role, (n[4] + transpose) % 12] += 1 / len(sounding)
                x[frame, role, 14] = (np.mean([n[4] for n in sounding]) - 60) / 24
                x[frame, role, 15] = np.mean([n[5] for n in sounding]) / 127
                x[frame, role, 16] = np.mean([min(n[2] / 1920, 2) for n in sounding])
                x[frame, role, 17] = 1
            x[frame, role, 12] = min(len(onsets), 4) / 4
            x[frame, role, 13] = min(len(sounding), 4) / 4
    return x.reshape(frames, -1)


class CoherenceRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(3 * FEATURES, 64), nn.GELU())
        self.temporal = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self.head = nn.Linear(128, 3)

    def forward(self, x):
        h, _ = self.temporal(self.project(x))
        return self.head(h.mean(dim=1))


def pairwise_loss(model, positive, negative, role):
    pos = model(positive).gather(1, role[:, None]).squeeze(1)
    neg = model(negative).gather(1, role[:, None]).squeeze(1)
    return torch.nn.functional.softplus(-(pos - neg)).mean()
