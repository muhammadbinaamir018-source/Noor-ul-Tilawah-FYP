"""Run the Qaida tutor over labelled recordings and report how well it judges.

How it works: for each row of the labels CSV the recording is transcribed with
the backend's own `asr.transcribe` and scored with `matching.evaluate_match`
(the exact code path of POST /api/evaluate), so the numbers describe the real
app. Transcripts are cached in results/transcripts.json keyed by the audio
file's hash, so `--sweep` and re-runs do not pay for ASR again.

Usage (from the repo root, with the backend venv active):

    python eval/run_eval.py data/labels.csv                 # metrics at the app's thresholds
    python eval/run_eval.py data/labels.csv --threshold 70  # one threshold for every mode
    python eval/run_eval.py data/labels.csv --sweep         # FAR/FRR vs threshold plot
    python eval/run_eval.py data/labels.csv --harakat-threshold 60 --name with-harakat
                                                            # variants + vowel-mark signal, to compare
    python eval/run_eval.py --dry-run                       # reference-audio sanity check

Outputs in eval/results/<run-name>/: predictions.csv (every row with what was
heard, the score and the verdict), metrics.csv, per_item.csv, confusion.csv,
confusion.png, and with --sweep sweep.csv + sweep.png.

--dry-run needs no learner data: it labels the publisher's reference clips of
pages p01–p12 as correct for their own item and wrong for another item on the
same page. Those clips are a professional reciter plus Urdu explanations, so
its output is a reference-audio sanity check of the harness, not an
evaluation result, and every file it writes says so.
"""

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.config import settings  # noqa: E402
from app.services import asr  # noqa: E402
from app.services.content_store import store  # noqa: E402
from app.services.harakat import score_against  # noqa: E402
from app.services.matching import evaluate_match  # noqa: E402
from eval.labels import Row, read_labels, scope_regions  # noqa: E402
from eval.metrics import LABELS, VERDICTS, Scored, compute_metrics, sweep, verdict  # noqa: E402

RESULTS = ROOT / "eval" / "results"
SANITY_NOTE = "reference-audio sanity check: publisher clips, not learner recordings"


# ---- scoring -----------------------------------------------------------------

class TranscriptCache:
    """audio file hash -> transcription, persisted as JSON."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def get(self, audio_path: str, transcribe) -> str:
        audio = Path(audio_path).read_bytes()
        key = hashlib.md5(audio).hexdigest()
        if key not in self.data:
            self.data[key] = transcribe(audio)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=0), encoding="utf-8")
        return self.data[key]


def score_rows(rows: list[Row], transcribe, cache: TranscriptCache | None = None, log=print) -> list[Scored]:
    regions = scope_regions()
    out = []
    t0 = time.perf_counter()
    for i, row in enumerate(rows, start=1):
        expected = regions[row.item_id].expected
        text = cache.get(row.audio_path, transcribe) if cache else transcribe(Path(row.audio_path).read_bytes())
        res = evaluate_match(text, expected)
        out.append(Scored(row.speaker_id, row.item_id, row.audio_path, row.label, row.error_type,
                          text, res["score"], expected.mode, res["threshold"],
                          harakat_score=score_against(text, expected)))
        if i % 10 == 0 or i == len(rows):
            log(f"  scored {i}/{len(rows)} ({time.perf_counter() - t0:.0f}s)")
    return out


# ---- synthetic labels for --dry-run --------------------------------------------

def synthetic_labels(n_items: int, seed: int = 0) -> list[Row]:
    """Publisher clips as pseudo-learner recordings: each sampled item's own clip
    labelled correct, and the same clip labelled wrong for another item on the
    same page with different expected text."""
    regions = scope_regions()
    by_page: dict[str, list] = {}
    for r in regions.values():
        by_page.setdefault(r.id.split("-")[0], []).append(r)
    rng = random.Random(seed)
    sample = rng.sample(sorted(regions.values(), key=lambda r: r.id), min(n_items, len(regions)))
    rows = []
    for r in sample:
        clip = str(settings.audio_dir / r.audio)
        rows.append(Row("publisher", r.id, clip, "correct"))
        others = [o for o in by_page[r.id.split("-")[0]] if o.expected.text != r.expected.text]
        if others:
            rows.append(Row("publisher", rng.choice(others).id, clip, "wrong", "substitution"))
    return rows


# ---- output --------------------------------------------------------------------

def write_outputs(out_dir: Path, scored: list[Scored], threshold: float | None, do_sweep: bool, note: str = "",
                  harakat_threshold: float | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    m = compute_metrics(scored, threshold, harakat_threshold)

    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "item_id", "audio_path", "label", "error_type", "transcription", "score",
                    "harakat_score", "mode", "threshold", "verdict"])
        for r in scored:
            w.writerow([r.speaker_id, r.item_id, r.audio_path, r.label, r.error_type, r.transcription, r.score,
                        "" if r.harakat_score is None else r.harakat_score, r.mode,
                        r.threshold if threshold is None else threshold, verdict(r, threshold, harakat_threshold)])

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in ("n", "n_correct_labels", "n_wrong_labels", "accuracy", "far", "frr"):
            w.writerow([k, "" if m[k] is None else m[k]])
        w.writerow(["threshold", "app default per mode" if threshold is None else threshold])
        w.writerow(["harakat_threshold", "off (variants only)" if harakat_threshold is None else harakat_threshold])
        if note:
            w.writerow(["note", note])

    with (out_dir / "per_item.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "n", "correct", "accuracy"])
        for item, v in sorted(m["per_item"].items()):
            w.writerow([item, v["n"], v["correct"], v["accuracy"]])

    with (out_dir / "confusion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label \\ verdict", *VERDICTS])
        for label in LABELS:
            w.writerow([label, *[m["confusion"].get((label, v), 0) for v in VERDICTS]])
    plot_confusion(out_dir / "confusion.png", m, note)

    if do_sweep:
        points = sweep(scored, list(range(0, 101, 5)), harakat_threshold)
        with (out_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["threshold", "far", "frr", "accuracy"])
            w.writeheader()
            w.writerows(points)
        plot_sweep(out_dir / "sweep.png", points, note)
    return m


def plot_confusion(path: Path, m: dict, note: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = [[m["confusion"].get((label, v), 0) for v in VERDICTS] for label in LABELS]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.imshow(grid, cmap="Blues")
    ax.set_xticks(range(len(VERDICTS)), VERDICTS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("tutor verdict")
    ax.set_ylabel("human label")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, str(v), ha="center", va="center", color="black" if v < max(map(max, grid)) / 2 else "white")
    title = f"n={m['n']}  acc={m['accuracy']:.2f}  FAR={_fmt(m['far'])}  FRR={_fmt(m['frr'])}"
    ax.set_title(title + ("\n" + note if note else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(path: Path, points: list[dict], note: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [p["threshold"] for p in points]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for key, label in (("far", "False Accept Rate"), ("frr", "False Reject Rate"), ("accuracy", "accuracy")):
        ys = [p[key] if p[key] is not None else float("nan") for p in points]
        ax.plot(xs, ys, marker="o", ms=3, label=label)
    ax.set_xlabel("score threshold (rapidfuzz ratio)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("threshold ablation" + ("\n" + note if note else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


# ---- CLI --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="?", type=Path, help="labels CSV (speaker_id,item_id,audio_path,label,error_type)")
    ap.add_argument("--threshold", type=float, help="use one score threshold for every mode instead of the app's")
    ap.add_argument("--sweep", action="store_true", help="also write FAR/FRR vs threshold (0..100 step 5)")
    ap.add_argument("--harakat-threshold", type=float,
                    help="also require the vowel-mark score to reach this (variants + harakat signal); omit for variants only")
    ap.add_argument("--dry-run", action="store_true", help="publisher clips as synthetic labels (sanity check only)")
    ap.add_argument("--n", type=int, default=15, help="--dry-run: how many items to sample")
    ap.add_argument("--name", help="results sub-folder (default: labels file stem or 'dry-run')")
    args = ap.parse_args(argv)

    if not args.dry_run and args.labels is None:
        ap.error("give a labels CSV or --dry-run")
    store.load()
    note = ""
    if args.dry_run:
        rows = synthetic_labels(args.n)
        note = SANITY_NOTE
        print(f"{SANITY_NOTE}; {len(rows)} synthetic rows from {args.n} items")
    else:
        rows = read_labels(args.labels)
        print(f"{len(rows)} labelled rows, {len({r.speaker_id for r in rows})} speakers")

    name = args.name or ("dry-run" if args.dry_run else args.labels.stem)
    out_dir = RESULTS / name
    cache = TranscriptCache(RESULTS / "transcripts.json")
    print(f"transcribing with {'remote GPU' if asr.is_remote() else settings.resolved_asr_model} ...")
    scored = score_rows(rows, asr.transcribe, cache)
    m = write_outputs(out_dir, scored, args.threshold, args.sweep, note, args.harakat_threshold)
    print(f"n={m['n']}  accuracy={_fmt(m['accuracy'])}  FAR={_fmt(m['far'])}  FRR={_fmt(m['frr'])}")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
