import pytest

from app.schemas import Expected
from app.services.matching import evaluate_match, normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("بَ", "ب"),  # fatha stripped
        ("اَلْحَمْدُ", "الحمد"),  # full tashkeel stripped
        ("أَ", "ا"),  # hamza-on-alef unified
        ("إِ", "ا"),
        ("آ", "ا"),
        ("مُوسَىٰ", "موسي"),  # superscript alef stripped, alef maqsura -> ya
        ("رَحْمَة", "رحمه"),  # taa marbuta -> ha
        ("بـــا", "با"),  # tatweel stripped
        ("  با   تا ", "با تا"),  # whitespace collapsed
        ("ba!", ""),  # non-Arabic removed
        ("د د د د د د", "د"),  # Whisper repetition loop collapsed
        ("با تا با", "با تا با"),  # mixed tokens kept
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_exact_match_is_correct() -> None:
    result = evaluate_match("بَا", Expected(text="با", mode="letter-name"))
    assert result["verdict"] == "correct"
    assert result["score"] == 100.0


def test_variant_match_is_correct() -> None:
    result = evaluate_match("باء", Expected(text="با", variants=["باء", "ب"], mode="letter-name"))
    assert result["verdict"] == "correct"


def test_wrong_letter_is_incorrect() -> None:
    result = evaluate_match("جيم", Expected(text="با", variants=["باء"], mode="letter-name"))
    assert result["verdict"] == "incorrect"


def test_similar_letter_name_is_incorrect() -> None:
    # alif transcribed as "الم" shares two characters with "دال" (ratio 66.7)
    # and must not pass the letter-name threshold
    result = evaluate_match("الم", Expected(text="دال", variants=["د"], mode="letter-name"))
    assert result["verdict"] == "incorrect"


# The cases below are real Whisper outputs taken from content/eval_log —
# the model appends hallucinated tails to correct letter names.


@pytest.mark.parametrize(
    ("heard", "text", "variants"),
    [
        ("شهِيدٌ", "شين", ["شش", "ش"]),  # sheen + tail
        ("ر وَعْد", "را", ["رى", "راء", "ر"]),  # correct token + extra token
        ("ده", "دال", ["د"]),  # dal + tail
        ("ذَرْ", "ذال", ["ذ"]),  # dhal + tail
        ("زه", "زا", ["زى", "زاء", "ز"]),  # za + tail
        ("جس", "جيم", ["ج"]),  # jeem + tail
        ("بى", "با", ["باء", "ب", "بائ"]),  # ba + alef maqsura
        ("تَى", "تا", ["طه", "تاء", "ت", "تائ"]),  # ta + alef maqsura
        ("وَسُيِّئَتْ", "سين", ["س"]),  # hallucinated wa- prefix on seen
    ],
)
def test_letter_with_decoder_tail_is_correct(heard: str, text: str, variants: list[str]) -> None:
    result = evaluate_match(heard, Expected(text=text, variants=variants, mode="letter-name"))
    assert result["verdict"] == "correct", result


@pytest.mark.parametrize(
    ("heard", "text", "variants"),
    [
        ("ج", "با", ["باء", "ب", "بائ"]),  # plainly the wrong letter
        ("سِيق", "شين", ["شش", "ش"]),  # seen is not sheen
        ("وَالْمُؤْمِنِينَ", "تا", ["طه", "تاء", "ت", "تائ"]),  # full hallucinated word
        ("فَسَبِّرُوا", "الف", ["الم", "ألف", "أ", "ا"]),  # full hallucinated word
        ("وَالْمُسْلِمُ", "با", ["باء", "ب", "بائ"]),  # wa-strip must not rescue long words
    ],
)
def test_wrong_or_hallucinated_word_stays_incorrect(heard: str, text: str, variants: list[str]) -> None:
    result = evaluate_match(heard, Expected(text=text, variants=variants, mode="letter-name"))
    assert result["verdict"] == "incorrect", result


def test_empty_transcription_is_unclear() -> None:
    result = evaluate_match("", Expected(text="با", mode="letter-name"))
    assert result["verdict"] == "unclear"


def test_word_mode_uses_stricter_threshold() -> None:
    # One-letter difference in a 4-letter word: ratio 75 -> passes word threshold exactly
    result = evaluate_match("درس", Expected(text="درسا", mode="word"))
    assert result["threshold"] == 75.0


def test_phrase_close_match() -> None:
    result = evaluate_match(
        "بسم الله الرحمن الرحيم",
        Expected(text="بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ", mode="phrase"),
    )
    assert result["verdict"] == "correct"
