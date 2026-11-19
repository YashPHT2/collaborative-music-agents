"""Small, versioned, scripted probes; not a representative music benchmark."""
from dataclasses import dataclass

TPB = 480
BAR = TPB * 4
CHANGE = BAR * 4
END = BAR * 8
SUITE_VERSION = "1.0"


@dataclass
class Case:
    id: str
    family: str
    split: str
    condition: str
    description: str
    melody: list
    harmony: list
    seeds: tuple


def build_cases():
    patterns = {
        "theme_a": ([[72,76,79,76], [69,72,77,76], [71,74,79,77], [76,74,72,72]],
                    [{0,4,7}, {5,9,0}, {7,11,2}, {0,4,7}], "development", (11,22,33)),
        "theme_b": ([[76,81,79,76], [77,81,84,81], [79,74,71,74], [72,79,76,72]],
                    [{9,0,4}, {5,9,0}, {7,11,2}, {0,4,7}], "reserved_evaluation", (44,55,66)),
    }
    descriptions = {
        "steady": "Original phrase repeats in C major.",
        "key_shift": "Melody and intended harmony move to D major at bar 5; no key signal is supplied.",
        "melody_change": "Pitch order changes at bar 5; rhythm and intended harmony stay the same.",
        "rhythm_change": "Syncopated melody onsets from bar 5; pitches and intended harmony stay the same.",
        "combined_change": "D-major shift and syncopated melody onsets from bar 5.",
    }
    cases = []
    for family, (phrase, chords, split, seeds) in patterns.items():
        for condition, description in descriptions.items():
            lead, harmony = [], []
            for bar in range(8):
                post = bar >= 4
                shift = 2 if post and condition in ("key_shift", "combined_change") else 0
                pitches = phrase[bar % 4].copy()
                if post and condition == "melody_change":
                    pitches = [pitches[2], pitches[0], pitches[3], pitches[1]]
                syncopated = post and condition in ("rhythm_change", "combined_change")
                offsets = [0,360,720,1320] if syncopated else [0,480,960,1440]
                durations = [240,240,480,480] if syncopated else [420]*4
                for pitch, offset, duration in zip(pitches, offsets, durations):
                    lead.append(["note", bar*BAR+offset, duration, 0, pitch+shift, 90])
                harmony.append(dict(start=bar*BAR, end=(bar+1)*BAR,
                                    chord=sorted((pc+shift)%12 for pc in chords[bar%4]),
                                    scale=sorted((pc+shift)%12 for pc in [0,2,4,5,7,9,11])))
            cases.append(Case(f"{family}_{condition}", family, split, condition, description,
                              lead, harmony, seeds))
    return cases
