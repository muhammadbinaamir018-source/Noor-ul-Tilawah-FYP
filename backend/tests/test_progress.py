"""Per-learner progress: pure logic tests plus a file round trip."""

import pytest

from app.config import settings
from app.services import progress as P

ITEMS = {
    "p01-r004": {"pageId": "p01", "text": "الف", "letter": "ا"},
    "p01-r005": {"pageId": "p01", "text": "با", "letter": "ب"},
    "p07-r003": {"pageId": "p07", "text": "اَ", "letter": "ا"},
}


def attempts(seq):
    """seq: list of (region_id, verdict, score, letter)."""
    state = P.empty()
    for region_id, verdict, score, letter in seq:
        state = P.apply_attempt(state, region_id, verdict, score, letter, when="2026-10-25 10:00:00")
    return state


# ---- apply_attempt ----------------------------------------------------------------

def test_attempt_counts_and_pass():
    s = attempts([("p01-r004", "incorrect", 40.0, "ا"), ("p01-r004", "correct", 100.0, "ا")])
    item = s["items"]["p01-r004"]
    assert item["attempts"] == 2 and item["fails"] == 1 and item["passed"] is True
    assert item["lastVerdict"] == "correct" and item["lastScore"] == 100.0 and item["lastTime"]


def test_unclear_counts_as_a_fail_and_passed_stays_once_earned():
    s = attempts([("p01-r004", "correct", 100.0, "ا"), ("p01-r004", "unclear", 0.0, "ا")])
    item = s["items"]["p01-r004"]
    assert item["passed"] is True and item["fails"] == 1 and item["attempts"] == 2


def test_letter_stats_follow_the_letter_given():
    s = attempts([("p01-r005", "incorrect", 30.0, "ب"), ("p01-r005", "incorrect", 30.0, "ب"),
                  ("p01-r004", "correct", 100.0, "ا")])
    assert s["letters"]["ب"] == {"attempts": 2, "fails": 2}
    assert s["letters"]["ا"] == {"attempts": 1, "fails": 0}


def test_apply_attempt_does_not_mutate_input():
    s0 = P.empty()
    P.apply_attempt(s0, "p01-r004", "correct", 100.0, "ا")
    assert s0["items"] == {}


# ---- summarise -------------------------------------------------------------------------

def test_summary_counts_weak_letters_and_revise_list():
    s = attempts([
        ("p01-r004", "incorrect", 40.0, "ا"), ("p01-r004", "incorrect", 45.0, "ا"),   # never passed
        ("p01-r005", "incorrect", 30.0, "ب"), ("p01-r005", "correct", 100.0, "ب"),    # passed after one fail
        ("p07-r003", "correct", 100.0, "ا"),
    ])
    summary = P.summarise("ali", s, ITEMS)
    assert summary["learnerId"] == "ali"
    assert summary["itemsAttempted"] == 3 and summary["itemsPassed"] == 2 and summary["attempts"] == 5
    assert summary["weakLetters"][0] == {"letter": "ا", "attempts": 3, "fails": 2}
    assert [r["regionId"] for r in summary["revise"]] == ["p01-r004"]
    assert summary["revise"][0] == {"regionId": "p01-r004", "pageId": "p01", "text": "الف", "fails": 2, "lastScore": 45.0}


def test_revise_list_is_hardest_first_and_capped():
    seq = [(f"p01-r{n:03d}", "incorrect", 10.0, "ا") for n in range(10, 30)]
    seq += [("p01-r015", "incorrect", 10.0, "ا")] * 3
    items = {f"p01-r{n:03d}": {"pageId": "p01", "text": "x", "letter": "ا"} for n in range(10, 30)}
    summary = P.summarise("x", attempts(seq), items, limit=5)
    assert len(summary["revise"]) == 5 and summary["revise"][0]["regionId"] == "p01-r015"


def test_weak_letters_only_lists_letters_with_fails():
    summary = P.summarise("x", attempts([("p01-r004", "correct", 100.0, "ا")]), ITEMS)
    assert summary["weakLetters"] == [] and summary["revise"] == []


# ---- files ---------------------------------------------------------------------------------

def test_load_save_round_trip_and_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "progress_dir", tmp_path)
    assert P.load("sara") == P.empty()
    s = P.apply_attempt(P.load("sara"), "p01-r004", "correct", 100.0, "ا")
    P.save("sara", s)
    assert P.load("sara") == s and (tmp_path / "sara.json").exists()
    P.reset("sara")
    assert P.load("sara") == P.empty()


@pytest.mark.parametrize("bad", ["../x", "a/b", "", "a b"])
def test_learner_ids_are_safe_file_names(bad):
    with pytest.raises(ValueError):
        P.check_id(bad)
    assert P.check_id("S01-ali_2") == "S01-ali_2"
