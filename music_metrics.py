"""Transparent symbolic diagnostics. No aggregate musical-quality score."""
from collections import Counter
import math

TPB = 480


def overlap(a, b, c, d):
    return max(0, min(b, d)-max(a, c))


def diagnostics(lead, accompaniment, harmony, start, end):
    if end <= start:
        raise ValueError("Scoring window must have positive length")
    notes = [n for n in accompaniment if overlap(n[1], n[1]+n[2], start, end)]
    onsets = [n for n in accompaniment if start <= n[1] < end]
    duration = sum(overlap(n[1], n[1]+n[2], start, end) for n in notes)
    fit = dict(scale=0, chord=0)
    for note in notes:
        for segment in harmony:
            weight = overlap(max(start,note[1]), min(end,note[1]+note[2]),
                             segment["start"], segment["end"])
            for kind in fit:
                if note[4] % 12 in segment[kind]:
                    fit[kind] += weight
    pair_duration, rough_pair_duration = 0, 0
    for note in notes:
        for melody_note in lead:
            weight = overlap(max(start,note[1]), min(end,note[1]+note[2]),
                             melody_note[1], melody_note[1]+melody_note[2])
            pair_duration += weight
            if abs(note[4]-melody_note[4]) % 12 in (1,6,11):
                rough_pair_duration += weight
    boundaries = sorted({start,end} | {max(start,n[1]) for n in notes} |
                        {min(end,n[1]+n[2]) for n in notes})
    covered, peak_polyphony = 0, 0
    for left, right in zip(boundaries,boundaries[1:]):
        count = sum(n[1] < right and n[1]+n[2] > left for n in notes)
        peak_polyphony = max(peak_polyphony,count)
        covered += right-left if count else 0
    counts = Counter(n[4] for n in onsets)
    entropy = -sum((count/len(onsets))*math.log2(count/len(onsets)) for count in counts.values())
    lead_onsets = [n[1] for n in lead if start <= n[1] < end]
    aligned = sum(any(abs(n[1]-tick)<=30 for tick in lead_onsets) for n in onsets)
    longest, run, previous = 0,0,None
    for note in sorted(onsets,key=lambda n:(n[1],n[4])):
        run = run+1 if note[4]==previous else 1
        longest, previous = max(longest,run),note[4]
    ratio = lambda a,b: round(a/b,6) if b else None
    return dict(note_count=len(onsets), notes_per_beat=round(len(onsets)/((end-start)/TPB),6),
                sounding_coverage=ratio(covered,end-start), max_polyphony=peak_polyphony,
                scale_fit_duration=ratio(fit["scale"],duration),
                intended_chord_fit_duration=ratio(fit["chord"],duration),
                rough_melody_interval_fraction=ratio(rough_pair_duration,pair_duration),
                melody_onset_alignment=ratio(aligned,len(onsets)),
                dominant_pitch_share=ratio(max(counts.values(),default=0),len(onsets)),
                pitch_entropy_bits=round(entropy,6), longest_same_pitch_run=longest)


def adversarial_examples(end=15360):
    """Simple failures that expose why a single harmony score is unsafe."""
    make = lambda t,p,d=120: ["note",t,d,1,p,64]
    scale = [48,50,52,53,55,57,59,60]
    return {
        "silence": [],
        "repeated_C": [make(t,48,420) for t in range(0,end,480)],
        "ascending_scale": [make(t,scale[i%8],210) for i,t in enumerate(range(0,end,240))],
        "dense_chromatic": [make(t,p,120) for t in range(0,end,120) for p in range(36,67)],
    }
