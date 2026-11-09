"""Madd (vowel-length) module tests.

Precise classification cases use synthetic vowels of known length, so they are
deterministic. Publisher clips are fixtures only for a loose sanity check (a
madd clip's longest voiced stretch is longer than the short-vowel clip of the
same letter); they contain Urdu explanations, so nothing finer is asserted on
them (reference-audio sanity check, not an evaluation).
"""

import io
import wave

import numpy as np
import pytest

from app.config import settings
from tajweed.madd import (
    DEFAULT_HARAKAH_S,
    MaddResult,
    analyse,
    classify,
    expected_counts_for,
    measure,
)

SR = 16000


def vowel(seconds: float, amp: float = 0.5, f0: float = 150.0) -> np.ndarray:
    """A voiced vowel-like tone (f0 + harmonics) with 20 ms fade in/out."""
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3, 4))
    x = x / np.abs(x).max() * amp
    ramp = int(0.02 * SR)
    env = np.ones_like(x)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return (x * env).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def wav_bytes(samples: np.ndarray) -> bytes:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def clip_with_reference(vowel_s: float, ref_s: float = 0.2) -> np.ndarray:
    """A short reference syllable, a pause, then the vowel under test."""
    return np.concatenate([silence(0.2), vowel(ref_s), silence(0.25), vowel(vowel_s), silence(0.2)])


# ---- measure: voiced segments -----------------------------------------------

def test_measure_finds_both_segments_and_picks_longest_and_reference():
    m = measure(clip_with_reference(0.5, 0.2), SR)
    assert len(m.segments) == 2
    assert m.longest_s == pytest.approx(0.5, abs=0.05)
    assert m.reference_s == pytest.approx(0.2, abs=0.05)
    assert not m.reference_is_default


def test_measure_single_segment_falls_back_to_default_reference():
    m = measure(np.concatenate([silence(0.2), vowel(0.5), silence(0.2)]), SR)
    assert len(m.segments) == 1
    assert m.reference_s == DEFAULT_HARAKAH_S and m.reference_is_default


def test_measure_bridges_tiny_gaps_inside_a_vowel():
    x = np.concatenate([silence(0.2), vowel(0.25), silence(0.03), vowel(0.25), silence(0.2)])
    assert len(measure(x, SR).segments) == 1


# ---- classify: counts -> verdict -------------------------------------------------

@pytest.mark.parametrize("vowel_s,expected,verdict,ok", [
    (0.45, 2, "2-count", True),
    (0.45, 4, "2-count", False),
    (0.85, 4, "4-count", True),
    (0.85, 2, "4-count", False),
    (0.22, 2, "too-short", False),
    (1.30, 2, "too-long", False),
    (1.30, 4, "too-long", False),
])
def test_classify_against_a_200ms_reference(vowel_s, expected, verdict, ok):
    r = classify(longest_s=vowel_s, reference_s=0.2, expected_counts=expected)
    assert r.verdict == verdict and r.ok is ok
    assert r.counts == pytest.approx(vowel_s / 0.2)
    assert 0.0 <= r.confidence <= 1.0


def test_confidence_is_higher_mid_class_than_at_a_boundary():
    mid = classify(0.45, 0.2, 2).confidence       # 2.25 counts, inside [1.5, 3)
    edge = classify(0.31, 0.2, 2).confidence      # 1.55 counts, just over the line
    assert mid > edge


def test_hint_mentions_the_direction_of_the_error():
    assert "longer" in classify(0.22, 0.2, 2).hint.lower()
    assert "shorter" in classify(1.3, 0.2, 2).hint.lower()
    assert classify(0.45, 0.2, 2).hint


# ---- analyse: bytes in, result out ---------------------------------------------

def test_analyse_two_count_from_wav_bytes():
    r = analyse(wav_bytes(clip_with_reference(0.45)), expected_counts=2)
    assert isinstance(r, MaddResult)
    assert r.verdict == "2-count" and r.ok
    assert r.measured_s == pytest.approx(0.45, abs=0.06)
    assert r.expected_counts == 2


def test_analyse_four_count_from_wav_bytes():
    r = analyse(wav_bytes(clip_with_reference(0.85)), expected_counts=4)
    assert r.verdict == "4-count" and r.ok


def test_analyse_silence_is_unclear():
    r = analyse(wav_bytes(silence(1.0)), expected_counts=2)
    assert r.verdict == "unclear" and not r.ok and r.confidence == 0.0
    assert r.hint


def test_analyse_very_short_audio_is_unclear():
    r = analyse(wav_bytes(vowel(0.05)), expected_counts=2)
    assert r.verdict == "unclear"


def test_analyse_clipped_audio_still_measures_and_is_flagged():
    loud = clip_with_reference(0.45) * 4.0  # overdriven: clipped to ±1 in the wav
    r = analyse(wav_bytes(loud), expected_counts=2)
    assert r.verdict == "2-count"
    assert r.clipped is True
    assert "clipp" in r.hint.lower()


def test_analyse_without_reference_segment_has_low_confidence():
    r = analyse(wav_bytes(np.concatenate([silence(0.2), vowel(0.45), silence(0.2)])), expected_counts=2)
    assert r.verdict == "2-count"
    assert r.reference_is_default and r.confidence <= 0.5


# ---- expected counts from the item's text --------------------------------------

@pytest.mark.parametrize("text,counts", [
    ("بَا", 2),          # fatha + alif
    ("نُوْحُ", 2),       # damma + waw sakin
    ("دِيْنِيْ", 2),     # kasra + ya sakin
    ("هٰذَا", 2),        # dagger alif
    ("جَآءَ", 4),        # madd sign
    ("بَ", None),        # short vowel only
    ("دَرَسَ", None),
    ("با", None),        # no tashkeel: cannot tell
])
def test_expected_counts_for(text, counts):
    assert expected_counts_for(text) == counts


# ---- publisher clips: reference-audio sanity check ----------------------------

@pytest.mark.parametrize("madd_clip,short_clip", [
    ("a04004.mp3", "a03005.mp3"),  # بَا vs بَ
    ("a04006.mp3", "a03007.mp3"),  # ثَا vs ثَ
    ("a04007.mp3", "a03008.mp3"),  # جَا vs جَ
])
def test_publisher_madd_clip_longest_segment_exceeds_short_vowel_clip(madd_clip, short_clip):
    madd = analyse((settings.audio_dir / madd_clip).read_bytes(), expected_counts=2)
    short = analyse((settings.audio_dir / short_clip).read_bytes(), expected_counts=2)
    assert madd.verdict != "unclear" and short.verdict != "unclear"
    assert madd.measured_s > short.measured_s
