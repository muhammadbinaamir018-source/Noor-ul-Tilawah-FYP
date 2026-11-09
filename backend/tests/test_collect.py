"""Integration tests for /api/collect with a temporary recordings folder."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "asr_preload", False)
    monkeypatch.setattr(settings, "recordings_dir", tmp_path / "recordings")
    with TestClient(app) as c:
        yield c


def start(client, **over):
    body = {"consent": True, "ageGroup": "adult", "native": False, **over}
    return client.post("/api/collect/speakers", json=body)


def test_items_are_scope_pages_in_reading_order(client):
    items = client.get("/api/collect/items").json()
    assert items[0]["id"] == "p01-r002" and items[0]["text"]
    pages = [i["pageId"] for i in items]
    assert pages == sorted(pages) and set(pages) == {f"p{n:02d}" for n in range(1, 13)}
    assert all(i["mode"] in ("letter-name", "syllable", "word", "phrase") for i in items)


def test_consent_is_required(client):
    assert start(client, consent=False).status_code == 422


def test_speaker_ids_increment(client):
    assert start(client).json()["speakerId"] == "S01"
    assert start(client, ageGroup="child", native=True).json()["speakerId"] == "S02"
    assert (settings.recordings_dir / "speakers.csv").read_text(encoding="utf-8").count("\n") == 3


def test_record_rerecord_skip_and_progress(client):
    sid = start(client).json()["speakerId"]

    def rec(item, data=b"webm-bytes"):
        return client.post("/api/collect/recordings",
                           files={"audio": ("rec.webm", data, "audio/webm")},
                           data={"speaker_id": sid, "item_id": item})

    assert rec("p01-r004").status_code == 200
    assert rec("p01-r004", b"second take, longer").status_code == 200  # re-record overwrites
    assert (settings.recordings_dir / sid / "p01-r004.webm").read_bytes() == b"second take, longer"
    assert client.post("/api/collect/skip", json={"speakerId": sid, "itemId": "p01-r005"}).status_code == 200
    assert client.get(f"/api/collect/progress/{sid}").json()["status"] == {"p01-r004": "recorded", "p01-r005": "skipped"}
    meta = (settings.recordings_dir / "metadata.csv").read_text(encoding="utf-8").splitlines()
    assert len(meta) == 3  # header + one row per item, not per take


def test_rejects_unknown_speaker_item_and_empty_audio(client):
    sid = start(client).json()["speakerId"]
    post = lambda s, i, b=b"x": client.post("/api/collect/recordings", files={"audio": ("r.webm", b, "audio/webm")},
                                             data={"speaker_id": s, "item_id": i})
    assert post("S99", "p01-r004").status_code == 404
    assert post(sid, "p13-r003").status_code == 404   # outside the scope
    assert post(sid, "p01-r001").status_code == 404   # title row, not recitable
    assert post(sid, "p01-r004", b"").status_code == 422


def test_export_is_a_labels_template_the_eval_harness_reads(client):
    from eval.labels import LabelError, read_labels

    sid = start(client).json()["speakerId"]
    for item in ("p01-r004", "p01-r005"):
        client.post("/api/collect/recordings", files={"audio": ("r.webm", b"x", "audio/webm")},
                    data={"speaker_id": sid, "item_id": item})
    client.post("/api/collect/skip", json={"speakerId": sid, "itemId": "p01-r006"})
    res = client.get("/api/collect/export")
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/csv")
    lines = res.text.strip().splitlines()
    assert lines[0] == "speaker_id,item_id,audio_path,label,error_type"
    assert lines[1:] == [f"{sid},p01-r004,{sid}/p01-r004.webm,,", f"{sid},p01-r005,{sid}/p01-r005.webm,,"]

    template = settings.recordings_dir / "labels-template.csv"
    assert template.exists()
    with pytest.raises(LabelError):          # blank labels: not ready for the harness yet
        read_labels(template)
    filled = template.with_name("labels.csv")  # the qari fills the label column
    # rows end with ",," (blank label, blank error_type); fill them in
    filled.write_text("\n".join([lines[0], lines[1][:-1] + "correct,", lines[2][:-1] + "wrong,madd"]) + "\n", encoding="utf-8")
    rows = read_labels(filled)
    assert [r.label for r in rows] == ["correct", "wrong"] and rows[1].error_type == "madd"
    assert Path(rows[0].audio_path).exists()
