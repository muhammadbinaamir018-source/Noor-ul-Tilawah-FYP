"""Unit tests for the evaluation harness (eval/): metrics, speaker split and
label loading. Pure logic, no ASR; scores are given directly."""

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.labels import LabelError, read_labels  # noqa: E402
from eval.metrics import Scored, compute_metrics, sweep, verdict  # noqa: E402
from eval.split import SpeakerOverlap, assert_no_overlap, split_by_speaker  # noqa: E402


def scored(label, score, item="p01-r004", speaker="s1", threshold=75.0, transcription="x", harakat_score=None):
    return Scored(speaker_id=speaker, item_id=item, audio_path="a.wav", label=label,
                  error_type="", transcription=transcription, score=score,
                  mode="letter-name", threshold=threshold, harakat_score=harakat_score)


# ---- verdict ---------------------------------------------------------------

def test_verdict_uses_threshold_and_unclear_for_empty_transcription():
    assert verdict(scored("correct", 80.0), None) == "correct"
    assert verdict(scored("correct", 70.0), None) == "incorrect"
    assert verdict(scored("correct", 70.0), 60.0) == "correct"  # override
    assert verdict(scored("correct", 100.0, transcription=""), None) == "unclear"


def test_harakat_threshold_only_rejects_when_the_signal_exists_and_is_low():
    row_low = scored("wrong", 90.0, harakat_score=0.0)
    row_high = scored("correct", 90.0, harakat_score=100.0)
    row_none = scored("correct", 90.0, harakat_score=None)
    assert verdict(row_low, None) == "correct"                          # variants only
    assert verdict(row_low, None, harakat_threshold=50) == "incorrect"  # + harakat signal
    assert verdict(row_high, None, harakat_threshold=50) == "correct"
    assert verdict(row_none, None, harakat_threshold=50) == "correct"   # no signal: no change
    m = compute_metrics([row_low, row_high, row_none], harakat_threshold=50)
    assert m["far"] == 0.0 and m["frr"] == 0.0


# ---- metrics ---------------------------------------------------------------

ROWS = [
    scored("correct", 90), scored("correct", 85), scored("correct", 60),   # 1 of 3 rejected
    scored("wrong", 30), scored("wrong", 80),                             # 1 of 2 accepted
    scored("wrong", 100, item="p01-r005", transcription=""),              # unclear = reject
]


def test_far_frr_accuracy():
    m = compute_metrics(ROWS)
    assert m["n"] == 6
    assert m["frr"] == pytest.approx(1 / 3)
    assert m["far"] == pytest.approx(1 / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)


def test_per_item_accuracy():
    m = compute_metrics(ROWS)
    assert m["per_item"]["p01-r004"] == {"n": 5, "correct": 3, "accuracy": pytest.approx(3 / 5)}
    assert m["per_item"]["p01-r005"] == {"n": 1, "correct": 1, "accuracy": 1.0}


def test_confusion_matrix_counts():
    m = compute_metrics(ROWS)
    c = m["confusion"]
    assert c[("correct", "correct")] == 2
    assert c[("correct", "incorrect")] == 1
    assert c[("wrong", "correct")] == 1
    assert c[("wrong", "incorrect")] == 1
    assert c[("wrong", "unclear")] == 1


def test_metrics_with_only_correct_rows_has_no_far():
    m = compute_metrics([scored("correct", 90)])
    assert m["far"] is None and m["frr"] == 0.0


def test_sweep_is_monotonic():
    points = sweep(ROWS, [0, 25, 50, 75, 100])
    fars = [p["far"] for p in points]
    frrs = [p["frr"] for p in points]
    assert fars == sorted(fars, reverse=True)
    assert frrs == sorted(frrs)
    assert points[0]["threshold"] == 0 and points[-1]["far"] == 0.0


# ---- speaker split ---------------------------------------------------------

def rows_for(speakers):
    return [scored("correct", 90, speaker=s, item=f"p01-r{i:03d}") for s in speakers for i in range(3)]


def test_split_keeps_each_speaker_on_one_side():
    rows = rows_for(["a", "b", "c", "d", "e"])
    train, test = split_by_speaker(rows, test_fraction=0.4, seed=1)
    assert {r.speaker_id for r in train} & {r.speaker_id for r in test} == set()
    assert len(train) + len(test) == len(rows)
    assert len({r.speaker_id for r in test}) == 2


def test_split_is_deterministic_for_a_seed():
    rows = rows_for(["a", "b", "c", "d"])
    assert split_by_speaker(rows, 0.5, seed=7) == split_by_speaker(rows, 0.5, seed=7)


def test_assert_no_overlap_raises():
    train = rows_for(["a", "b"])
    test = rows_for(["b"])
    with pytest.raises(SpeakerOverlap):
        assert_no_overlap(train, test)
    assert_no_overlap(rows_for(["a"]), rows_for(["c"]))  # no raise


# ---- end to end with a fake transcriber (no ASR) ---------------------------

def test_run_writes_outputs_with_fake_transcriber(tmp_path):
    from eval.run_eval import TranscriptCache, score_rows, synthetic_labels, write_outputs

    rows = synthetic_labels(n_items=4, seed=1)
    assert {r.label for r in rows} == {"correct", "wrong"}
    regions = __import__("eval.labels", fromlist=["scope_regions"]).scope_regions()
    # "hear" exactly the expected text of the item the clip really belongs to
    truth = {r.audio_path: regions[r.item_id].expected.text for r in rows if r.label == "correct"}
    calls = []

    def fake_transcribe(audio_bytes):
        calls.append(1)
        return next(t for p, t in truth.items() if Path(p).read_bytes() == audio_bytes)

    scored = score_rows(rows, fake_transcribe, TranscriptCache(tmp_path / "cache.json"), log=lambda s: None)
    assert len(calls) == 4  # the cache serves the "wrong" rows that reuse a clip
    m = write_outputs(tmp_path / "out", scored, None, True, note="test")
    assert m["frr"] == 0.0
    for name in ("predictions.csv", "metrics.csv", "per_item.csv", "confusion.csv", "confusion.png", "sweep.csv", "sweep.png"):
        assert (tmp_path / "out" / name).exists(), name


# ---- labels CSV ------------------------------------------------------------

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["speaker_id", "item_id", "audio_path", "label", "error_type"])
        w.writeheader()
        w.writerows(rows)


def test_read_labels_resolves_items_and_relative_paths(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    csv_path = tmp_path / "labels.csv"
    write_csv(csv_path, [{"speaker_id": "s1", "item_id": "p01-r004", "audio_path": "a.wav", "label": "correct", "error_type": ""}])
    rows = read_labels(csv_path)
    assert rows[0].audio_path == str(tmp_path / "a.wav")
    assert rows[0].label == "correct"


@pytest.mark.parametrize("bad", [
    {"item_id": "p13-r001", "label": "correct"},   # outside p01-p12
    {"item_id": "p01-r999", "label": "correct"},   # no such region
    {"item_id": "p01-r001", "label": "correct"},   # region has no expected text
    {"item_id": "p01-r004", "label": "maybe"},     # bad label
])
def test_read_labels_rejects_bad_rows(tmp_path, bad):
    (tmp_path / "a.wav").write_bytes(b"x")
    csv_path = tmp_path / "labels.csv"
    write_csv(csv_path, [{"speaker_id": "s1", "audio_path": "a.wav", "error_type": "", **bad}])
    with pytest.raises(LabelError):
        read_labels(csv_path)
