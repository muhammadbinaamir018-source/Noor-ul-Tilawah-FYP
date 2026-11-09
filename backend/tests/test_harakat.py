"""Harakat (vowel mark) secondary score: pure logic tests."""

import pytest

from app.schemas import Expected
from app.services.harakat import harakat_score, score_against


@pytest.mark.parametrize("heard,expected,score", [
    ("بَ", "بَ", 100.0),
    ("بِ", "بَ", 0.0),                 # kasra for fatha
    ("بُ", "بَ", 0.0),
    ("دَرَسُ", "دَرَسَ", pytest.approx(66.7, abs=0.1)),   # 2 of 3 vowels right
    ("دُرِسَ", "دَرَسَ", pytest.approx(33.3, abs=0.1)),
    ("بِسْمِ اللَّهِ", "بِسْمِ اللَّهِ", 100.0),
    ("نُوْحُ", "نُوْحُ", 100.0),       # sukun counts as a mark
    ("بًا", "بًا", 100.0),              # tanween counts as a mark
])
def test_vowel_agreement(heard, expected, score):
    assert harakat_score(heard, expected) == score


def test_none_when_expected_has_no_tashkeel():
    assert harakat_score("بَ", "ب") is None
    assert harakat_score("بَسْمِ", "بسم الله") is None


def test_none_when_whisper_wrote_no_tashkeel():
    assert harakat_score("درس", "دَرَسَ") is None


def test_partial_marks_only_compare_marked_letters():
    # Whisper marked only the first letter; the other two are unknown, not wrong
    assert harakat_score("دَرس", "دَرَسَ") == 100.0
    assert harakat_score("دِرس", "دَرَسَ") == 0.0


def test_letter_errors_are_not_vowel_errors():
    # a substituted letter is the main score's business; its vowel is simply not compared
    assert harakat_score("دَلَسَ", "دَرَسَ") == 100.0
    # an extra hallucinated word does not dilute the score
    assert harakat_score("دَرَسَ وَعَدَ", "دَرَسَ") == 100.0


def test_alef_forms_and_shadda_do_not_break_alignment():
    assert harakat_score("أَحَدٌ", "اَحَدٌ") == 100.0
    assert harakat_score("ثُمَّ", "ثُمَّ") == 100.0


def test_best_over_text_and_variants():
    exp = Expected(text="بسم الله الرحمن الرحيم",
                   variants=["بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"], mode="phrase")
    assert score_against("بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ", exp) == 100.0
    assert score_against("بَسْمُ اللَّهِ الرَّحْمَنِ الرَّحِيمِ", exp) < 100.0
    assert score_against("بسم الله", Expected(text="بسم الله", variants=[], mode="phrase")) is None
