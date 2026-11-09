"""Speaker-wise train/test split.

How it works: recordings are grouped by speaker_id and whole speakers are
assigned to the test side until `test_fraction` of the speakers is there. A
speaker is never on both sides, so a model or threshold tuned on the train side
is measured on voices it has never heard. `assert_no_overlap` is the guard to
call before any evaluation that uses the split.
"""

import random
from collections import defaultdict
from typing import Sequence, TypeVar

T = TypeVar("T")


class SpeakerOverlap(ValueError):
    pass


def split_by_speaker(rows: Sequence[T], test_fraction: float, seed: int = 0) -> tuple[list[T], list[T]]:
    by_speaker: dict[str, list[T]] = defaultdict(list)
    for r in rows:
        by_speaker[r.speaker_id].append(r)
    speakers = sorted(by_speaker)
    random.Random(seed).shuffle(speakers)
    n_test = max(1, round(len(speakers) * test_fraction)) if len(speakers) > 1 else 0
    test_speakers = set(speakers[:n_test])
    train = [r for s in speakers if s not in test_speakers for r in by_speaker[s]]
    test = [r for s in speakers if s in test_speakers for r in by_speaker[s]]
    assert_no_overlap(train, test)
    return train, test


def assert_no_overlap(train: Sequence, test: Sequence) -> None:
    shared = {r.speaker_id for r in train} & {r.speaker_id for r in test}
    if shared:
        raise SpeakerOverlap(f"speakers on both sides of the split: {sorted(shared)}")
