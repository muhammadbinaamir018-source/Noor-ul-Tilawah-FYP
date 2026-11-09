"""Arabic text normalization and recitation-verdict logic.

Diacritics are stripped before comparison because Whisper's diacritic output
is unreliable; harakat correctness is covered by per-item expected variants.
"""

import re

from rapidfuzz import fuzz

from ..config import settings
from ..schemas import Expected

# Tashkeel + Quranic annotation marks: fathatan..wavy hamza, superscript alef,
# and the U+06D6-06ED range of small/high Quranic marks.
_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_NON_ARABIC = re.compile(r"[^ء-ي\s]")

_ALEF_FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})


def normalize(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = text.translate(_ALEF_FORMS)
    text = _NON_ARABIC.sub("", text)
    tokens = text.split()
    # Whisper loops on very short clips ("د د د د د"); a run of one repeated
    # token carries no more information than the token itself.
    if len(set(tokens)) == 1:
        tokens = tokens[:1]
    return " ".join(tokens)


_SHORT_MODES = ("letter-name", "syllable")


def _core_letter(candidates: list[str]) -> str:
    """The single-character variant of a letter-name item ("د" for دال)."""
    for cand in candidates:
        norm = normalize(cand)
        if len(norm) == 1:
            return norm
    return ""


def evaluate_match(transcription: str, expected: Expected) -> dict:
    """Compare a Whisper transcription against an item's expected text/variants."""
    norm_trans = normalize(transcription)
    threshold = settings.threshold_for(expected.mode)

    candidates = [expected.text, *expected.variants]
    best_score = 0.0
    best_variant = candidates[0]

    # The model is tuned on continuous recitation, so an isolated letter name
    # often comes back with a hallucinated extra token ("ر وعد" for را).
    # For short modes each token competes on its own as well.
    units = [norm_trans]
    if expected.mode in _SHORT_MODES:
        units += norm_trans.split()

    for candidate in candidates:
        norm_cand = normalize(candidate)
        if not norm_cand:
            continue
        for unit in units:
            score = 100.0 if unit and unit == norm_cand else fuzz.ratio(unit, norm_cand)
            if score > best_score:
                best_score, best_variant = score, candidate

    # Initial-consonant credit: the decisive phoneme of a letter name is its
    # first consonant. A short transcription starting with the target letter
    # ("شهيد" for شين, "ده" for دال) is the right letter plus a decoder tail.
    # The length cap keeps full hallucinated words ("والمؤمنين") failing.
    if expected.mode in _SHORT_MODES and best_score < threshold and norm_trans:
        core = _core_letter(candidates)
        first = norm_trans.split()[0]
        if core and first[0] != core and len(first) > 1 and first.startswith("و"):
            first = first[1:]  # hallucinated "wa-" conjunction prefix
        if core and len(first) <= 4 and first[0] == core:
            best_score, best_variant = 85.0, core

    if not norm_trans:
        verdict = "unclear"
    elif best_score >= threshold:
        verdict = "correct"
    else:
        verdict = "incorrect"

    out = {
        "verdict": verdict,
        "normalizedTranscription": norm_trans,
        "bestVariant": best_variant,
        "score": round(best_score, 1),
        "threshold": threshold,
    }
    if settings.harakat_signal_enabled:
        from .harakat import score_against  # lazy: harakat imports this module's alef table

        out["harakatScore"] = score_against(transcription, expected)
    return out
