"""Integration test of POST /api/evaluate with the madd check switched on.

ASR is replaced by a fake that returns the item's expected text, so the test
needs no model; the audio is a synthetic vowel (see test_madd.py)."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services import asr

from test_madd import clip_with_reference, wav_bytes  # tests/ is on sys.path under pytest

ITEM = "p10-r003"  # ءَا: first madd item on takhti 4


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "asr_preload", False)      # no model load in tests
    monkeypatch.setattr(settings, "eval_log_enabled", False)  # do not record test audio
    monkeypatch.setattr(settings, "progress_dir", tmp_path / "progress")
    monkeypatch.setattr(asr, "is_loaded", lambda: True)
    monkeypatch.setattr(asr, "transcribe", lambda audio_bytes: "ءا")
    with TestClient(app) as c:
        yield c


def post(client, seconds=0.45):
    return client.post("/api/evaluate",
                       files={"audio": ("rec.wav", wav_bytes(clip_with_reference(seconds)), "audio/wav")},
                       data={"region_id": ITEM})


def test_madd_in_response_when_enabled(client, monkeypatch):
    monkeypatch.setattr(settings, "madd_enabled", True)
    body = post(client).json()
    assert body["verdict"] == "correct"
    assert body["madd"]["verdict"] == "2-count" and body["madd"]["ok"] is True
    assert body["madd"]["expected_counts"] == 2
    assert body["madd"]["hint"]


def test_madd_too_short_is_reported_but_does_not_change_text_verdict(client, monkeypatch):
    monkeypatch.setattr(settings, "madd_enabled", True)
    body = post(client, seconds=0.22).json()
    assert body["verdict"] == "correct"
    assert body["madd"]["verdict"] == "too-short" and body["madd"]["ok"] is False


def test_feedback_on_a_miss_names_the_letters(client, monkeypatch):
    monkeypatch.setattr(settings, "madd_enabled", False)
    monkeypatch.setattr(asr, "transcribe", lambda audio_bytes: "دا")   # heard د for ء
    body = post(client).json()
    assert body["verdict"] == "incorrect"
    fb = body["feedback"]
    assert fb["expectedLetter"] == "ء" and fb["heardLetter"] == "د"
    assert fb["heard"] == "دا" and fb["lowConfidence"] is False
    assert "wrong" not in fb["message"].lower()


def test_feedback_absent_on_a_clean_pass(client, monkeypatch):
    monkeypatch.setattr(settings, "madd_enabled", False)
    assert post(client).json()["feedback"] is None


def test_progress_is_recorded_per_learner_and_can_be_reset(client, monkeypatch):
    monkeypatch.setattr(settings, "madd_enabled", False)
    monkeypatch.setattr(asr, "transcribe", lambda audio_bytes: "دا")   # a miss
    for _ in range(2):
        client.post("/api/evaluate", files={"audio": ("rec.wav", wav_bytes(clip_with_reference(0.45)), "audio/wav")},
                    data={"region_id": ITEM, "learner_id": "ali"})
    monkeypatch.setattr(asr, "transcribe", lambda audio_bytes: "ءا")   # then a pass
    client.post("/api/evaluate", files={"audio": ("rec.wav", wav_bytes(clip_with_reference(0.45)), "audio/wav")},
                data={"region_id": ITEM, "learner_id": "ali"})
    s = client.get("/api/progress/ali").json()
    assert s["itemsAttempted"] == 1 and s["itemsPassed"] == 1 and s["attempts"] == 3
    assert s["weakLetters"][0]["letter"] == "ء" and s["weakLetters"][0]["fails"] == 2
    assert s["revise"] == []  # passed in the end
    assert client.get("/api/progress/nobody").json()["attempts"] == 0
    assert client.delete("/api/progress/ali").status_code == 200
    assert client.get("/api/progress/ali").json()["attempts"] == 0
    assert client.get("/api/progress/../x").status_code in (404, 422)


def test_evaluate_without_learner_id_records_nothing(client, monkeypatch):
    monkeypatch.setattr(settings, "madd_enabled", False)
    post(client)
    assert not list((settings.progress_dir).glob("*.json")) if settings.progress_dir.exists() else True


def test_madd_absent_when_disabled(client, monkeypatch):
    monkeypatch.setattr(settings, "madd_enabled", False)
    assert post(client).json()["madd"] is None
