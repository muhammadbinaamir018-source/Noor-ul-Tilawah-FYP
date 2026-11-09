"""Evaluate a manifest through a running app server's WebSocket, exactly like the browser.

Unlike tools/evaluate.py (which loads models itself), this measures whatever the running
server uses: local models or a remote model service (e.g. Kaggle), network included.
Scoring uses the same functions as tools/evaluate.py; live latency comes from the
server's per-session stats; "after stop" times are measured here.

Usage: python tools/ws_eval.py testset/manifest.jsonl [--url ws://localhost:8765/ws] [--out report.json]
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from app import arabic, audio, quran  # noqa: E402
from evaluate import match, predicted_errors, wer  # noqa: E402

CHUNK = 1600


async def run_item(url: str, item: dict, root: Path) -> dict:
    x = audio.load(root / item["audio"])
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "surah": item["surah"], "from": item.get("from", 1),
                                  "to": item.get("to"), "level": item.get("level", "beginner")}))
        stopped = None
        marks = {}

        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                if m["type"] == "error":
                    raise RuntimeError(m["message"])
                if m["type"] not in ("state", "final") or stopped is None:
                    continue
                statuses = {w["status"] for w in m["words"]}
                if "words" not in marks and "pending" not in statuses and not m["transcript"]["partial"]:
                    marks["words"] = time.perf_counter() - stopped
                if m["type"] == "final":
                    marks["final"] = time.perf_counter() - stopped
                    return m

        task = asyncio.create_task(reader())
        start = time.perf_counter()
        for k in range(0, len(pcm), CHUNK):
            await ws.send(pcm[k:k + CHUNK].tobytes())
            await asyncio.sleep(max(0.0, start + (k + CHUNK) / 16000 - time.perf_counter()))
        stopped = time.perf_counter()
        await ws.send(json.dumps({"type": "stop"}))
        final = await asyncio.wait_for(task, timeout=300)
    return {"final": final, "words_after_stop": marks.get("words", marks["final"]),
            "final_after_stop": marks["final"]}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--url", default="ws://localhost:8765/ws")
    ap.add_argument("--out")
    args = ap.parse_args()
    manifest = Path(args.manifest)
    items = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

    errs = words = tp = n_lab = n_pred = 0
    p50s, p90s, decode, w_wait, f_wait, report = [], [], [], [], [], []
    for item in items:
        r = await run_item(args.url, item, manifest.parent)
        final = r["final"]
        hyp = arabic.words(final["transcript"]["committed"])
        e, n = wer(arabic.words(item["recited"]), hyp)
        preds = predicted_errors(final)
        ref_norm = [w.norm for w in quran.reference(item["surah"], item.get("from", 1), item.get("to"))]
        hits = match(item.get("errors", []), preds, ref_norm)
        errs, words, tp = errs + e, words + n, tp + hits
        n_lab, n_pred = n_lab + len(item.get("errors", [])), n_pred + len(preds)
        st = final["stats"]
        p50s.append(st["latency_ms_p50"]); p90s.append(st["latency_ms_p90"]); decode.append(st["decode_ms_p50"])
        w_wait.append(r["words_after_stop"]); f_wait.append(r["final_after_stop"])
        verified = sum(w.get("verified", False) for w in final["words"])
        print(f"{Path(item['audio']).name:<42} WER {e / max(n, 1):6.1%}  labels {len(item.get('errors', [])):2d}"
              f"  pred {len(preds):2d}  hit {hits:2d}  verified {verified}  p50 {st['latency_ms_p50']}ms"
              f"  words+{r['words_after_stop']:.1f}s letters+{r['final_after_stop']:.1f}s", flush=True)
        report.append({"audio": item["audio"], "hyp": " ".join(hyp), "predicted": preds, "stats": st,
                       "words_after_stop": r["words_after_stop"], "final_after_stop": r["final_after_stop"]})

    p = tp / n_pred if n_pred else 1.0
    rc = tp / n_lab if n_lab else 1.0
    print("\n=== summary ===")
    print(f"items {len(items)} | WER {errs / max(words, 1):.1%} ({errs}/{words}) | detection P {p:.1%} R {rc:.1%}"
          f" (labels {n_lab}, predicted {n_pred})")
    print(f"live latency per session: median p50 {np.median(p50s):.0f} ms, median p90 {np.median(p90s):.0f} ms,"
          f" worst p90 {max(p90s)} ms | ASR call (incl. network) median {np.median(decode):.0f} ms")
    print(f"after stop: words mean {np.mean(w_wait):.2f}s max {max(w_wait):.2f}s | "
          f"letters mean {np.mean(f_wait):.2f}s max {max(f_wait):.2f}s")
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


asyncio.run(main())
