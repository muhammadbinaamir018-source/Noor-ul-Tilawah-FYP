"""Controlled letter-substitution probe for the pronunciation layer.

A learner who says ه where ح is expected produces audio of ه against a
reference of ح. We reproduce exactly that without recording anyone: take a
correct recitation, find a letter the reciter pronounced as X, and change the
reference at that spot to the focus letter learners confuse with X. The
scorer should flag it. Scoring the unmodified reference gives the false-alarm
rate on correct focus letters. Sweeping the threshold picks PRON_THRESHOLD.

Usage: python tools/letter_probe.py [--cache testset/cache] [--surahs 1 112 113 114]
"""

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import audio, quran  # noqa: E402
from app.arabic import strip_diacritics  # noqa: E402
from app import models  # noqa: E402
from app.pronunciation import FOCUS_LETTERS, silent_clusters  # noqa: E402

# Letter actually said -> focus letter the learner was aiming for (common confusions).
SAID_INSTEAD_OF = {
    "ه": ["ح"], "ك": ["ق"], "س": ["ص", "ث"], "د": ["ض"], "ت": ["ط"], "ز": ["ظ", "ذ"],
    "ا": ["ع"], "أ": ["ع"], "ح": ["خ", "ه"], "خ": ["غ"], "ء": ["ع"],
}
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.35, 0.5, 0.7, 0.9]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="testset/cache")
    ap.add_argument("--surahs", type=int, nargs="+", default=[1, 112, 113, 114])
    args = ap.parse_args()

    # Local model, or app.model_service when MODEL_SERVICE_URL is set.
    scorer = models.load_scorer()
    correct_scores: list[float] = []                      # focus letters, unmodified reference
    swapped: dict[str, list[float]] = defaultdict(list)   # "said->expected" -> scores of expected
    clips = sorted(Path(args.cache).glob("*/*.mp3"))
    for clip in clips:
        s, a = int(clip.stem[:3]), int(clip.stem[3:])
        if s not in args.surahs:
            continue
        words = quran.reference(s, a, a)
        logp = scorer.log_probs(audio.load(clip))
        for scores in scorer.score_logp(logp, words).values():
            correct_scores += [l.score for l in scores if l.letter in FOCUS_LETTERS]
        for w in words:
            silent = silent_clusters(w.clusters, first_in_segment=w.index == 0)
            for ci, cl in enumerate(w.clusters):
                said = strip_diacritics(cl)[:1]
                if ci in silent or said not in SAID_INSTEAD_OF:
                    continue
                for expected in SAID_INSTEAD_OF[said]:
                    marks = cl[1:]
                    variant = replace(w, clusters=w.clusters[:ci] + [expected + marks] + w.clusters[ci + 1:])
                    probe = [variant if x.index == w.index else x for x in words]
                    got = scorer.score_logp(logp, probe).get(w.index, [])
                    hit = next((l.score for l in got if l.cluster == ci), None)
                    if hit is not None:
                        swapped[f"{said}->{expected}"].append(hit)
        print(f"{clip.parent.name}/{clip.name}: done", flush=True)

    all_swapped = np.array([x for v in swapped.values() for x in v])
    correct = np.array(correct_scores)
    print(f"\nfocus letters recited correctly: {len(correct)}   substitution probes: {len(all_swapped)}\n")
    print(f"{'threshold':>9}  {'recall (probes flagged)':>24}  {'false alarms (correct flagged)':>31}")
    for t in THRESHOLDS:
        print(f"{t:>9.2f}  {np.mean(all_swapped < t):>24.1%}  {np.mean(correct < t):>31.1%}")
    t = float(os.getenv("PRON_THRESHOLD", "0.35"))
    print(f"\nper confusion at threshold {t}:")
    for k, v in sorted(swapped.items(), key=lambda kv: -len(kv[1])):
        v = np.array(v)
        print(f"  said {k.split('->')[0]} for {k.split('->')[1]}:  {np.mean(v < t):6.1%} flagged  (n={len(v)}, median score {np.median(v):.2f})")


if __name__ == "__main__":
    main()
