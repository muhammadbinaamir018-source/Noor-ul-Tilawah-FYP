"""Learner feedback after a recitation check.

How it works: the text verdict only says correct / incorrect. This turns the
match result into something a learner can act on: what was heard against
what was expected (both normalised the way the matcher sees them), the first
letter that differs (difflib opcodes on the two strings: a substitution gives
both letters, a deletion the expected one, an insertion the extra one), and a
message. When the check itself is shaky, the message says "Try again" rather
than blaming the learner: Whisper heard nothing ("unclear"), or the score sits
within `feedback_confidence_margin` points of the pass threshold, where a
noisy recording flips the verdict either way. If the madd check ran and
failed, its hint rides along, including on a correct text verdict.
"""

from difflib import SequenceMatcher

from ..config import settings
from ..schemas import Expected, Feedback, MaddInfo
from .matching import normalize


def differing_letter(expected: str, heard: str) -> tuple[str | None, str | None]:
    """(expected letter, heard letter) at the first difference; None on the
    side that has nothing there; (None, None) when the strings agree."""
    e, h = normalize(expected).replace(" ", ""), normalize(heard).replace(" ", "")
    for tag, i1, _i2, j1, _j2 in SequenceMatcher(None, e, h, autojunk=False).get_opcodes():
        if tag == "replace":
            return e[i1], h[j1]
        if tag == "delete":
            return e[i1], None
        if tag == "insert":
            return None, h[j1]
    return None, None


def build_feedback(result: dict, expected: Expected, madd: MaddInfo | None) -> Feedback:
    verdict, score, threshold = result["verdict"], result["score"], result["threshold"]
    heard = result["normalizedTranscription"]
    best = normalize(result.get("bestVariant") or expected.text)
    madd_hint = madd.hint if madd is not None and not madd.ok else None

    if verdict == "correct":
        return Feedback(message="Correct, and one thing to work on: the madd." if madd_hint else "Correct.",
                        heard=heard, expected=best, maddHint=madd_hint)
    if verdict == "unclear":
        return Feedback(message="Try again: we couldn't hear you clearly. Speak a little closer to the microphone.",
                        heard="", expected=best, lowConfidence=True, maddHint=madd_hint)

    low = abs(score - threshold) < settings.feedback_confidence_margin
    exp_letter, heard_letter = differing_letter(best, heard)
    if low:
        message = "Try again: that was close, but not clear enough to be sure."
    elif exp_letter and heard_letter:
        message = f"Not quite: expected «{exp_letter}» but heard «{heard_letter}». Listen, then try again."
    elif exp_letter:
        message = f"Not quite: the «{exp_letter}» was not heard. Listen, then try again."
    elif heard_letter:
        message = f"Not quite: an extra «{heard_letter}» was heard. Listen, then try again."
    else:
        message = "Not quite. Listen, then try again."
    return Feedback(message=message, heard=heard, expected=best, expectedLetter=exp_letter,
                    heardLetter=heard_letter, lowConfidence=low, maddHint=madd_hint)
