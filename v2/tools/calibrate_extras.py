"""Choose EXTRA_MARGIN from evaluation reports, without re-running the models.

tools/evaluate.py reports every extra word with its audio support (kept ones under
"extras", dropped ones under "dropped_extras"). This re-scores mistake detection as if
the margin had been each value in --margins: an extra is predicted if its support is at
least the margin (extras never checked, support None, always count).

Usage: python tools/calibrate_extras.py testset/manifest.jsonl:testset/report_new_clean.json \
                                        testset/noisy/manifest.jsonl:testset/noisy/report_new.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import quran  # noqa: E402
from evaluate import match  # noqa: E402


def load(pair: str) -> list[tuple[dict, dict]]:
    manifest, report = pair.split(":", 1) if pair.count(":") == 1 else pair.rsplit(":", 1)
    items = {json.loads(line)["audio"]: json.loads(line)
             for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()}
    rows = json.loads(Path(report).read_text(encoding="utf-8"))
    return [(items[r["audio"]], r) for r in rows]


def score(pairs: list[tuple[dict, dict]], margin: float) -> dict:
    tp = n_lab = n_pred = false_extras = 0
    for item, r in pairs:
        preds = [p for p in r["predicted"] if p["type"] != "extra"]
        for e in r["extras"] + r["dropped_extras"]:
            if e.get("support") is None or e["support"] >= margin:
                preds.append({"ref_index": e["after"], "type": "extra", "heard": e["heard"]})
        ref_norm = [w.norm for w in quran.reference(item["surah"], item.get("from", 1), item.get("to"))]
        labels = item.get("errors", [])
        hits = match(labels, preds, ref_norm)
        tp, n_lab, n_pred = tp + hits, n_lab + len(labels), n_pred + len(preds)
        extra_hits = match([lab for lab in labels if lab["type"] == "extra"],
                           [p for p in preds if p["type"] == "extra"], ref_norm)
        false_extras += sum(p["type"] == "extra" for p in preds) - extra_hits
    return {"precision": tp / n_pred if n_pred else 1.0, "recall": tp / n_lab if n_lab else 1.0,
            "false_extras": false_extras}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="manifest.jsonl:report.json")
    ap.add_argument("--margins", type=float, nargs="+", default=[-100, 0, 2, 4, 6, 8, 10, 15, 20])
    args = ap.parse_args()
    sets = {pair: load(pair) for pair in args.pairs}
    for pair, pairs in sets.items():
        sup = [e["support"] for _, r in pairs for e in r["extras"] + r["dropped_extras"] if e.get("support") is not None]
        print(f"\n{pair}\n  {len(sup)} checked extras; support min/median/max:",
              f"{min(sup):.1f} / {sorted(sup)[len(sup) // 2]:.1f} / {max(sup):.1f}" if sup else "-")
        print(f"  {'margin':>7}  {'precision':>9}  {'recall':>7}  {'false extras':>12}")
        for m in args.margins:
            s = score(pairs, m)
            print(f"  {m:>7.1f}  {s['precision']:>9.1%}  {s['recall']:>7.1%}  {s['false_extras']:>12d}")


if __name__ == "__main__":
    main()
