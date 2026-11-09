"""Score a trained makharij run on its held-out speakers.

How it works: the run folder written by train.py holds one model per pair
and the rows of the speakers that were kept out of training. Each held-out
clip is featurised the same way, predicted with its pair's model, and the
predictions are tallied into a confusion matrix per pair (CSV + PNG). For real
labelled data the per-pair accuracy is printed; for a --smoke run (publisher
clips) only the matrices are written and no accuracy is reported anywhere.

    python ml/makharij/evaluate.py ml/makharij/runs/run1
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from ml.makharij.data import PAIRS, Clip  # noqa: E402
from ml.makharij.features import extract  # noqa: E402


def confusion_png(path: Path, letters: list[str], counts: Counter, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = [[counts[(t, p)] for p in letters] for t in letters]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.imshow(grid, cmap="Blues")
    ax.set_xticks(range(len(letters)), letters, fontsize=14)
    ax.set_yticks(range(len(letters)), letters, fontsize=14)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    top = max(max(r) for r in grid) or 1
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, str(v), ha="center", va="center", color="black" if v < top / 2 else "white")
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(run_dir: Path, smoke: bool | None = None, log=print) -> dict:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    smoke = summary.get("smoke", False) if smoke is None else smoke
    rows = [Clip(**r) for r in json.loads((run_dir / "test_rows.json").read_text(encoding="utf-8"))]
    note = summary.get("note", "")
    by_pair: dict[str, list[Clip]] = defaultdict(list)
    for c in rows:
        by_pair[c.pair].append(c)
    results = {"smoke": smoke, "note": note, "test_speakers": sorted({c.speaker_id for c in rows}), "pairs": {}}
    for pair, clips in sorted(by_pair.items()):
        model_file = summary["pairs"].get(pair, {}).get("model_file")
        if not model_file:
            log(f"  {pair}: no model in this run, skipped")
            continue
        clf = joblib.load(run_dir / model_file)
        X = extract([c.audio_path for c in clips], summary["features"])
        pred = clf.predict(X)
        counts = Counter(zip([c.letter for c in clips], pred))
        letters = sorted({k for k, v in PAIRS.items() if v == pair})
        name = pair.replace("/", "-")
        with (run_dir / f"confusion-{name}.csv").open("w", encoding="utf-8") as f:
            f.write("true \\ predicted," + ",".join(letters) + "\n")
            for t in letters:
                f.write(t + "," + ",".join(str(counts[(t, p)]) for p in letters) + "\n")
        title = f"{pair}  n={len(clips)}" + ("\n" + note if note else "")
        confusion_png(run_dir / f"confusion-{name}.png", letters, counts, title)
        entry = {"n": len(clips), "confusion": {f"{t}>{p}": n for (t, p), n in counts.items()}}
        if not smoke:
            entry["accuracy"] = sum(n for (t, p), n in counts.items() if t == p) / len(clips)
            log(f"  {pair}: n={len(clips)} accuracy={entry['accuracy']:.2f}")
        else:
            log(f"  {pair}: n={len(clips)} confusion written ({note})")
        results["pairs"][pair] = entry
    (run_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args(argv)
    res = evaluate(args.run_dir)
    if res["smoke"]:
        print(res["note"])
    print(f"wrote {args.run_dir}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
