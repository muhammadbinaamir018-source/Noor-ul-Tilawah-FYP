"""Smoke test: the makharij pipeline runs end to end on publisher clips.

Reference-audio sanity check only: it asserts that files are produced and
that no accuracy is reported, never what the numbers are."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.makharij.data import LETTERS, publisher_dataset  # noqa: E402
from ml.makharij.evaluate import evaluate  # noqa: E402
from ml.makharij.features import mfcc_features  # noqa: E402
from ml.makharij.train import train  # noqa: E402


def test_mfcc_features_shape_and_determinism():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000) * 0.1).astype(np.float32)
    f = mfcc_features(x)
    assert f.shape == (27,) and np.isfinite(f).all()
    assert np.allclose(f, mfcc_features(x))


def test_publisher_dataset_covers_every_pair_with_pseudo_speakers():
    clips = publisher_dataset()
    assert {c.letter for c in clips} == LETTERS
    assert len({c.speaker_id for c in clips}) >= 3
    assert all(Path(c.audio_path).exists() for c in clips)


def test_pipeline_runs_end_to_end_without_reporting_accuracy(tmp_path):
    clips = publisher_dataset()
    run = tmp_path / "smoke"
    summary = train(clips, run, features="mfcc", model="svm", smoke=True, log=lambda s: None)
    assert summary["pairs"] and (run / "test_rows.json").exists()
    results = evaluate(run, log=lambda s: None)
    assert results["smoke"] is True and results["pairs"]
    for pair in results["pairs"]:
        name = pair.replace("/", "-")
        assert (run / f"confusion-{name}.csv").exists() and (run / f"confusion-{name}.png").exists()
    assert all("accuracy" not in entry for entry in results["pairs"].values())
    assert "reference-audio sanity check" in results["note"]
    assert "accuracy" not in json.dumps(results["pairs"])
