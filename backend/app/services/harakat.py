"""Secondary vowel-mark (harakat) score.

How it works: the main verdict strips tashkeel because Whisper's diacritics
are unreliable, so بَ and بِ look the same to it. This score keeps them: the
bare letters of what was heard are aligned with the expected text (difflib),
and on every aligned letter where *both* sides carry a vowel mark (fatha,
damma, kasra, sukun, tanween) the marks are compared. Score = agreeing /
compared × 100. Letters Whisper left unmarked are "unknown", not wrong, and
substituted or extra letters are the main score's business, so they are not
counted. None means nothing could be compared (the item has no tashkeel, or
Whisper wrote none), and then the signal must not be used. It never changes
the verdict by itself; it is a soft signal for the eval harness and, when
enabled, an extra field in the API response.
"""

from difflib import SequenceMatcher

from ..schemas import Expected
from .matching import _ALEF_FORMS

FATHATAN, DAMMATAN, KASRATAN = "ً", "ٌ", "ٍ"
FATHA, DAMMA, KASRA, SHADDA, SUKUN = "َ", "ُ", "ِ", "ّ", "ْ"
VOWEL_MARKS = {FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA, KASRA, SUKUN}  # shadda is doubling, not a vowel


def _tokens(text: str) -> list[tuple[str, str | None]]:
    """(letter, first vowel mark on it or None) for every Arabic letter."""
    out: list[tuple[str, str | None]] = []
    for ch in text.translate(_ALEF_FORMS):
        if "ء" <= ch <= "ي":          # ء..ي: a letter
            out.append((ch, None))
        elif ch in VOWEL_MARKS and out and out[-1][1] is None:
            out[-1] = (out[-1][0], ch)
    return out


def harakat_score(transcription: str, expected_text: str) -> float | None:
    exp, heard = _tokens(expected_text), _tokens(transcription)
    if not any(m for _, m in exp) or not any(m for _, m in heard):
        return None
    matcher = SequenceMatcher(None, [c for c, _ in exp], [c for c, _ in heard], autojunk=False)
    compared = agree = 0
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            e_mark, h_mark = exp[block.a + k][1], heard[block.b + k][1]
            if e_mark is not None and h_mark is not None:
                compared += 1
                agree += e_mark == h_mark
    return round(100.0 * agree / compared, 1) if compared else None


def score_against(transcription: str, expected: Expected) -> float | None:
    """Best harakat score over the item's expected text and its variants."""
    scores = [s for s in (harakat_score(transcription, c) for c in [expected.text, *expected.variants]) if s is not None]
    return max(scores) if scores else None
