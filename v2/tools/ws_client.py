"""Stream an audio file to a running server over the WebSocket, like the browser does.

Usage: python tools/ws_client.py <audio> --surah 112 [--from 1 --to 4] [--url ws://localhost:8765/ws] [--speed 1]
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import audio  # noqa: E402

CHUNK = 1600  # 100 ms


async def run(args) -> None:
    x = audio.load(args.audio)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "surah": args.surah, "from": args.from_, "to": args.to}))
        ref = None
        t0 = time.perf_counter()
        updates = 0

        async def reader():
            nonlocal ref, updates
            async for raw in ws:
                m = json.loads(raw)
                if m["type"] == "ready":
                    ref = m["reference"]
                elif m["type"] in ("state", "final"):
                    updates += 1
                    marks = {"correct": "+", "close": "~", "wrong": "x", "missing": "-", "pending": ".",
                             "not_reached": "."}
                    line = "".join(marks.get(w["status"], "?") for w in m["words"])
                    flagged = sum(l["flagged"] for w in m["words"] for l in w["letters"])
                    print(f"[{time.perf_counter() - t0:5.1f}s] {m['type']:<5} {line}  extras={len(m['extras'])}"
                          f"  weak_letters={flagged}  latency_p50={m['stats']['latency_ms_p50']}ms")
                    if m["type"] == "final":
                        print("\nfeedback:")
                        for f in m["feedback"]:
                            print("  -", f["text"])
                        print("stats:", m["stats"])
                        return
                elif m["type"] == "error":
                    print("error:", m["message"])
                    return

        task = asyncio.create_task(reader())
        start = time.perf_counter()
        for k in range(0, len(pcm), CHUNK):
            await ws.send(pcm[k:k + CHUNK].tobytes())
            await asyncio.sleep(max(0.0, start + (k + CHUNK) / 16000 / args.speed - time.perf_counter()))
        await ws.send(json.dumps({"type": "stop"}))
        await task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--surah", type=int, required=True)
    ap.add_argument("--from", dest="from_", type=int, default=1)
    ap.add_argument("--to", type=int)
    ap.add_argument("--url", default="ws://localhost:8765/ws")
    ap.add_argument("--speed", type=float, default=1.0)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
