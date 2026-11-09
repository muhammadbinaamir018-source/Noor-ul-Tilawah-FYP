"""Learner feedback builder: pure logic tests."""

from app.config import settings
from app.schemas import Expected, MaddInfo
from app.services.feedback import build_feedback, differing_letter


def match(verdict="incorrect", transcription="ضال", norm=None, best="دال", score=50.0, threshold=75.0):
    return {"verdict": verdict, "normalizedTranscription": transcription if norm is None else norm,
            "bestVariant": best, "score": score, "threshold": threshold}


DAL = Expected(text="دال", variants=["د"], mode="letter-name")


def madd(ok, hint="Hold the vowel longer: it should be 2 counts, you held it about 1.1."):
    return MaddInfo(measured_s=0.22, reference_s=0.2, reference_is_default=False, counts=1.1,
                    expected_counts=2, verdict="too-short", ok=ok, confidence=0.5, hint=hint, clipped=False)


# ---- differing letter -------------------------------------------------------------

def test_differing_letter_substitution():
    assert differing_letter("دال", "ضال") == ("د", "ض")


def test_differing_letter_missing_and_extra():
    assert differing_letter("بسم الله", "بسم") == ("ا", None)      # expected letter not heard
    assert differing_letter("بسم", "بسم الله") == (None, "ا")      # extra letter heard


def test_differing_letter_none_when_same():
    assert differing_letter("دال", "دال") == (None, None)


# ---- feedback ---------------------------------------------------------------------------

def test_clear_miss_names_the_letters():
    fb = build_feedback(match(), DAL, None)
    assert fb.lowConfidence is False
    assert fb.expectedLetter == "د" and fb.heardLetter == "ض"
    assert fb.heard == "ضال" and fb.expected == "دال"
    assert "د" in fb.message and "ض" in fb.message
    assert "wrong" not in fb.message.lower()


def test_low_confidence_near_threshold_says_try_again():
    fb = build_feedback(match(score=settings.threshold_letter_name - 3), DAL, None)
    assert fb.lowConfidence is True
    assert fb.message.lower().startswith("try again")


def test_unclear_is_low_confidence():
    fb = build_feedback(match(verdict="unclear", transcription="", best="دال", score=0.0), DAL, None)
    assert fb.lowConfidence is True
    assert "hear" in fb.message.lower()
    assert fb.expectedLetter is None


def test_margin_comes_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "feedback_confidence_margin", 0.0)
    assert build_feedback(match(score=74.0), DAL, None).lowConfidence is False
    monkeypatch.setattr(settings, "feedback_confidence_margin", 30.0)
    assert build_feedback(match(score=50.0), DAL, None).lowConfidence is True


def test_madd_hint_is_included_when_madd_not_ok():
    fb = build_feedback(match(verdict="correct", transcription="دا", best="دَا", score=100.0),
                        Expected(text="دَا", variants=[], mode="syllable"), madd(ok=False))
    assert fb.maddHint and "longer" in fb.maddHint
    assert fb.expectedLetter is None
    assert "correct" in fb.message.lower()


def test_madd_hint_absent_when_madd_ok():
    fb = build_feedback(match(), DAL, madd(ok=True))
    assert fb.maddHint is None
