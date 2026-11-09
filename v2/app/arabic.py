"""Arabic text normalisation shared by ASR matching and pronunciation scoring."""

import re
import unicodedata

def _char_class(*ranges: tuple[int, int], negate: bool = False, extra: str = "") -> str:
    body = "".join(re.escape(chr(a)) + (f"-{re.escape(chr(b))}" if b != a else "") for a, b in ranges)
    return f"[{'^' if negate else ''}{body}{extra}]"


# Harakat, tanween, shadda, sukun, superscript alef, Quranic annotation marks, tatweel.
DIACRITICS = re.compile(_char_class((0x0610, 0x061A), (0x064B, 0x065F), (0x0670, 0x0670),
                                    (0x06D6, 0x06ED), (0x0640, 0x0640)))
# Anything that is not an Arabic base letter (hamza .. ya) or whitespace.
_NON_LETTER = re.compile(_char_class((0x0621, 0x064A), negate=True, extra=r"\s"))

_LETTER_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه",
})


def strip_diacritics(text: str) -> str:
    return DIACRITICS.sub("", unicodedata.normalize("NFC", text))


def normalize(text: str) -> str:
    """Aggressive normalisation for word matching: no marks, unified letter variants."""
    text = strip_diacritics(text).translate(_LETTER_MAP)
    text = _NON_LETTER.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return [w for w in normalize(text).split(" ") if w]


def clusters(word: str) -> list[str]:
    """Split a diacritised word into grapheme clusters: base letter + its marks."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFC", word):
        if out and DIACRITICS.fullmatch(ch):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """1.0 for identical normalised words, 0.0 for completely different ones."""
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))
