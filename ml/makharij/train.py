"""Train a makharij classifier on one set of speakers.

How it works: clips are split by speaker (eval/split.py, never the same voice
on both sides), features are extracted for the train side, one model per
confusable pair is fitted and saved with joblib, and the held-out rows are
written to the run folder so evaluate.py scores exactly those.

    python ml/makharij/train.py data/makharij-labels.csv --features mfcc --model svm --name run1
    python ml/makharij/train.py --smoke        # publisher clips, pipeline check only

Run folders live in ml/makharij/runs/<name>/ (git-ignored).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from eval.split import split_by_speaker  # noqa: E402
from ml.makharij.data import Clip, publisher_dataset, read_labels  # noqa: E402
from ml.makharij.features import extract  # noqa: E402
from ml.makharij.models import make_model  # noqa: E402

RUNS = HERE / "runs"
SMOKE_NOTE = "reference-audio sanity check: publisher clips with page ids as pseudo speakers; no accuracy is reported"


def train(clips: list[Clip], out_dir: Path, features: str = "mfcc", model: str = "svm",
          test_fraction: float = 0.4, seed: int = 0, smoke: bool = False, log=print) -> dict:
    train_rows, test_rows = split_by_speaker(clips, test_fraction, seed)
    log(f"{len(clips)} clips, {len({c.speaker_id for c in clips})} speakers -> "
        f"train {len(train_rows)} / test {len(test_rows)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_pair: dict[str, list[Clip]] = defaultdict(list)
    for c in train_rows:
        by_pair[c.pair].append(c)
    summary = {"features": features, "model": model, "seed": seed, "smoke": smoke, "pairs": {}}
    for pair, rows in sorted(by_pair.items()):
        letters = {c.letter for c in rows}
        if len(letters) < 2:
            log(f"  {pair}: only {letters} on the train side, skipped")
            continue
        X = extract([c.audio_path for c in rows], features)
        y = [c.letter for c in rows]
        clf = make_model(model, seed).fit(X, y)
        name = pair.replace("/", "-")
        joblib.dump(clf, out_dir / f"{name}.joblib")
        summary["pairs"][pair] = {"train_clips": len(rows), "model_file": f"{name}.joblib"}
        log(f"  {pair}: trained on {len(rows)} clips")
    (out_dir / "test_rows.json").write_text(
        json.dumps([c.__dict__ for c in test_rows], ensure_ascii=False, indent=1), encoding="utf-8")
    if smoke:
        summary["note"] = SMOKE_NOTE
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="?", type=Path, help="CSV: speaker_id, audio_path, letter")
    ap.add_argument("--smoke", action="store_true", help="publisher clips instead of a labels CSV")
    ap.add_argument("--features", choices=["mfcc", "wav2vec2"], default="mfcc")
    ap.add_argument("--model", choices=["svm", "mlp"], default="svm")
    ap.add_argument("--test-fraction", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", help="run folder name (default: labels stem or 'smoke')")
    args = ap.parse_args(argv)
    if not args.smoke and args.labels is None:
        ap.error("give a labels CSV or --smoke")
    clips = publisher_dataset() if args.smoke else read_labels(args.labels)
    if args.smoke:
        print(SMOKE_NOTE)
    out_dir = RUNS / (args.name or ("smoke" if args.smoke else args.labels.stem))
    train(clips, out_dir, args.features, args.model, args.test_fraction, args.seed, args.smoke)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
