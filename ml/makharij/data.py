"""Labelled single-letter clips for the makharij classifier.

How it works: a labels CSV (speaker_id, audio_path, letter) lists recordings
of one letter each; `letter` must be one of the eight confusable letters and
is mapped to its pair. `publisher_dataset()` builds the same structure from
the Qaida reference clips (letter names on pages 1–2, harakat syllables on
pages 7–9) using the page id as a pseudo speaker, so the speaker-wise split
code runs; it is a reference-audio sanity check only, never training data
for a real result.
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

PAIRS = {"س": "س/ص", "ص": "س/ص", "د": "د/ض", "ض": "د/ض", "ك": "ك/ق", "ق": "ك/ق", "ه": "ه/ح", "ح": "ه/ح"}
LETTERS = set(PAIRS)
_MARKS = "ًٌٍَُِّْ"


@dataclass(frozen=True)
class Clip:
    speaker_id: str
    audio_path: str
    letter: str

    @property
    def pair(self) -> str:
        return PAIRS[self.letter]


def read_labels(path: Path) -> list[Clip]:
    path = Path(path)
    out = []
    with path.open(encoding="utf-8", newline="") as f:
        for n, rec in enumerate(csv.DictReader(f), start=2):
            letter = rec["letter"].strip()
            if letter not in LETTERS:
                raise ValueError(f"{path}:{n}: letter {letter!r} is not one of {sorted(LETTERS)}")
            audio = Path(rec["audio_path"].strip())
            if not audio.is_absolute():
                audio = path.parent / audio
            if not audio.exists():
                raise ValueError(f"{path}:{n}: audio not found: {audio}")
            out.append(Clip(rec["speaker_id"].strip(), str(audio), letter))
    if not out:
        raise ValueError(f"{path}: no rows")
    return out


def publisher_dataset(pages=("p01", "p02", "p07", "p08", "p09")) -> list[Clip]:
    """Reference clips whose item is a single confusable letter; pseudo speaker = page."""
    from app.config import settings
    from app.services.content_store import store

    if not store.manifest.lessons:
        store.load()
    clips = []
    for page_id in pages:
        for r in store.load_page(page_id).regions:
            if not r.expected:
                continue
            letter = None
            if r.expected.mode == "letter-name":
                letter = next((v for v in r.expected.variants if len(v) == 1), None)
            elif r.expected.mode == "syllable":
                bare = "".join(c for c in r.expected.text if c not in _MARKS)
                letter = bare if len(bare) == 1 else None
            if letter in LETTERS:
                clips.append(Clip(f"publisher-{page_id}", str(settings.audio_dir / r.audio), letter))
    return clips
