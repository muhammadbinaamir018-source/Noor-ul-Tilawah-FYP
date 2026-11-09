"""Evaluate the pipeline on a labelled manifest (see tools/make_testset.py).

Each file is streamed through the same RecitationSession the server uses, in
100 ms chunks. Metrics:
  * WER          - final hypothesis vs what was actually recited ("recited")
  * detection    - precision / recall / F1 of labelled mistakes, matched one-to-one
                   by type and word index (extras may carry an index "window")
  * letters      - precision / recall of flagged letters, if items carry "letters"
  * latency      - audio-received -> feedback-emitted (p50/p90) and real-time factor

Usage: python tools/evaluate.py testset/manifest.jsonl [--realtime] [--no-pronunciation]
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import arabic, audio, quran  # noqa: E402
from app import models  # noqa: E402
from app.session import RecitationSession  # noqa: E402

CHUNK = 1600
WORD_ERRORS = {"wrong", "close", "missing"}


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1], len(ref)


def predicted_errors(final: dict) -> list[dict]:
    out = [{"ref_index": w["index"], "type": w["status"]} for w in final["words"] if w["status"] in WORD_ERRORS]
    out += [{"ref_index": e["after"], "type": "extra", "heard": e["heard"]} for e in final["extras"]]
    return out


def same_type(label: str, pred: str) -> bool:
    # "wrong" and "close" are one class: a word was recited but not correctly.
    group = lambda t: "sub" if t in ("wrong", "close") else t  # noqa: E731
    return group(label) == group(pred)


EQUIV_SPAN = 8   # identical words this close are acoustically interchangeable


def match(labels: list[dict], preds: list[dict], ref_norm: list[str]) -> int:
    """One-to-one matches between labelled and predicted mistakes.

    A prediction matches a label of the same type at the labelled index (or inside
    its "window"), or nearby when the words are identical: when a surah repeats a
    phrase, the audio cannot say which copy was skipped or repeated.
    """
    def hit(lab, p):
        if not same_type(lab["type"], p["type"]):
            return False
        lo, hi = lab.get("window", [lab["ref_index"], lab["ref_index"]])
        if lo <= p["ref_index"] <= hi:
            return True
        i, j = lab["ref_index"], p["ref_index"]
        if lab["type"] == "extra":
            # An extra word is identified by what was said, near where it was said.
            return lab.get("heard") == p.get("heard") and lo - EQUIV_SPAN <= j <= hi + EQUIV_SPAN
        return 0 <= i < len(ref_norm) and 0 <= j < len(ref_norm) and abs(i - j) <= EQUIV_SPAN             and ref_norm[i] == ref_norm[j]

    used = set()
    tp = 0
    for lab in labels:
        for k, p in enumerate(preds):
            if k not in used and hit(lab, p):
                used.add(k)
                tp += 1
                break
    return tp


async def run_item(item: dict, root: Path, asr, scorer, realtime: bool) -> dict:
    ref = quran.reference(item["surah"], item.get("from", 1), item.get("to"))
    x = audio.load(root / item["audio"])

    async def emit(_):
        pass

    t0 = time.perf_counter()
    s = RecitationSession(ref, asr, emit, scorer, level=item.get("level", "beginner"))
    for k in range(0, len(x), CHUNK):
        await s.feed(x[k:k + CHUNK])
        # Pace against the wall clock: per-chunk sleeps overshoot and drift.
        await asyncio.sleep(max(0.0, t0 + (k + CHUNK) / 16000 - time.perf_counter()) if realtime else 0)
    stopped = time.perf_counter()
    final = await s.finish()
    done = time.perf_counter()
    return {"final": final, "elapsed": done - t0, "duration": len(x) / 16000, "latencies": s.latencies,
            "decode_times": s.decode_times,
            "words_after_stop": s.words_done_at - stopped, "final_after_stop": done - stopped}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--realtime", action="store_true", help="feed audio at 1x speed (true streaming latency)")
    ap.add_argument("--no-pronunciation", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--report", default=None, help="write per-item JSON report here")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    items = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = items[:args.limit] if args.limit else items

    # Local models, or app.model_service when MODEL_SERVICE_URL is set.
    asr = models.load_asr_model()
    scorer = None if args.no_pronunciation else models.load_scorer()

    errs = words = tp = n_lab = n_pred = 0
    l_tp = l_lab = l_pred = 0
    latencies, rtf, words_wait, final_wait, decode = [], [], [], [], []
    report = []
    for item in items:
        r = asyncio.run(run_item(item, manifest.parent, asr, scorer, args.realtime))
        final = r["final"]
        hyp = arabic.words(final["transcript"]["committed"])
        e, n = wer(arabic.words(item["recited"]), hyp)
        errs, words = errs + e, words + n
        preds = predicted_errors(final)
        ref_norm = [w.norm for w in quran.reference(item["surah"], item.get("from", 1), item.get("to"))]
        hits = match(item.get("errors", []), preds, ref_norm)
        tp, n_lab, n_pred = tp + hits, n_lab + len(item.get("errors", [])), n_pred + len(preds)
        if "letters" in item:
            flagged = {(w["index"], l["cluster"]) for w in final["words"] for l in w["letters"] if l["flagged"]}
            labelled = {(l["ref_index"], l["cluster"]) for l in item["letters"]}
            l_tp, l_lab, l_pred = l_tp + len(flagged & labelled), l_lab + len(labelled), l_pred + len(flagged)
        latencies += r["latencies"]
        decode += r["decode_times"]
        rtf.append(r["elapsed"] / r["duration"])
        words_wait.append(r["words_after_stop"])
        final_wait.append(r["final_after_stop"])
        print(f"{item['audio']:<55} WER {e / max(n, 1):6.1%}  labels {len(item.get('errors', [])):3d}"
              f"  predicted {len(preds):3d}  matched {hits:3d}")
        report.append({"audio": item["audio"], "wer": e / max(n, 1), "hyp": " ".join(hyp),
                       "predicted": preds, "labels": item.get("errors", []),
                       "extras": final["extras"], "dropped_extras": final.get("dropped_extras", [])})

    def prf(tp_, lab, pred):
        p = tp_ / pred if pred else 1.0
        rc = tp_ / lab if lab else 1.0
        f = 2 * p * rc / (p + rc) if p + rc else 0.0
        return p, rc, f

    p, rc, f = prf(tp, n_lab, n_pred)
    print("\n=== summary ===")
    print(f"items                 {len(items)}")
    print(f"WER                   {errs / max(words, 1):.1%}  ({errs}/{words} words)")
    print(f"mistake detection     P {p:.1%}  R {rc:.1%}  F1 {f:.1%}  (labels {n_lab}, predicted {n_pred})")
    if l_lab or l_pred:
        lp, lr, lf = prf(l_tp, l_lab, l_pred)
        print(f"letter flags          P {lp:.1%}  R {lr:.1%}  F1 {lf:.1%}")
    if latencies:
        print(f"feedback latency      p50 {np.quantile(latencies, .5) * 1000:.0f} ms"
              f"  p90 {np.quantile(latencies, .9) * 1000:.0f} ms"
              f"{'' if args.realtime else '  (not --realtime: includes queueing, not true streaming)'}")
    if decode:
        print(f"ASR decode            p50 {np.quantile(decode, .5) * 1000:.0f} ms  p90 {np.quantile(decode, .9) * 1000:.0f} ms"
              f"  ({len(decode)} calls)")
    if args.realtime:
        print(f"after stop: words     mean {np.mean(words_wait):.2f} s  max {np.max(words_wait):.2f} s")
        print(f"after stop: letters   mean {np.mean(final_wait):.2f} s  max {np.max(final_wait):.2f} s")
    else:
        print(f"real-time factor      {np.mean(rtf):.2f} (processing time / audio duration)")
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
