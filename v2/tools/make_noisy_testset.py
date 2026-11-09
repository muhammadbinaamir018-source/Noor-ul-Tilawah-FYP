"""Make a noisy copy of a labelled test set, to measure false alarms caused by room noise.

Every item gets synthetic household noise like the conditions reported by a real user
(ceiling fan, a street vendor's bell, knocks from moving things):
  * a continuous fan-like hum under the recitation at --snr dB,
  * --lead seconds of noise before it and --tail seconds after it, with bells and knocks.
The noise adds no words, so the labels are unchanged: any extra word the system
reports is a false alarm caused by noise.

Usage: python tools/make_noisy_testset.py testset/manifest.jsonl --out testset/noisy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import audio  # noqa: E402
from app.vad import SAMPLE_RATE  # noqa: E402


def fan(n: int, rng: np.random.Generator) -> np.ndarray:
    """Brown-ish broadband rumble with a slow blade wobble and a 50 Hz motor hum."""
    x = lfilter([1.0], [1.0, -0.98], rng.standard_normal(n))
    x -= lfilter([1.0], [1.0, -0.999], x) * 0.02            # drift removal
    t = np.arange(n) / SAMPLE_RATE
    x *= 1 + 0.15 * np.sin(2 * np.pi * 4.2 * t)             # blade wobble
    x = x / (np.std(x) + 1e-9) + 0.3 * np.sin(2 * np.pi * 50 * t)
    return x / np.std(x)


def bell(rng: np.random.Generator) -> np.ndarray:
    """One 'ding' of a handbell: inharmonic partials with exponential decay."""
    dur = rng.uniform(0.6, 1.0)
    t = np.arange(int(dur * SAMPLE_RATE)) / SAMPLE_RATE
    f0 = rng.uniform(900, 1400)
    tone = sum(a * np.sin(2 * np.pi * f0 * r * t) for a, r in [(1, 1), (0.6, 2.76), (0.4, 5.4), (0.25, 8.9)])
    return tone * np.exp(-t * rng.uniform(4, 7))


def knock(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(0.12 * SAMPLE_RATE)) / SAMPLE_RATE
    return rng.standard_normal(len(t)) * np.exp(-t * 60) + np.sin(2 * np.pi * 120 * t) * np.exp(-t * 30)


def events(n: int, rng: np.random.Generator, level: float) -> np.ndarray:
    """Bells (in bursts of 2-4 rings, like a cart bell) and knocks, over n samples."""
    out = np.zeros(n)
    pos = int(rng.uniform(0.2, 0.8) * SAMPLE_RATE)
    while pos < n:
        if rng.random() < 0.6:
            for _ in range(rng.integers(2, 5)):
                if pos >= n:
                    break
                b = bell(rng) * level
                end = min(n, pos + len(b))
                out[pos:end] += b[:end - pos]
                pos += int(rng.uniform(0.25, 0.4) * SAMPLE_RATE)
        else:
            k = knock(rng) * level * 1.5
            end = min(n, pos + len(k))
            out[pos:end] += k[:end - pos]
        pos += int(rng.uniform(1.0, 2.5) * SAMPLE_RATE)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default="testset/noisy")
    ap.add_argument("--snr", type=float, default=10.0, help="speech-to-fan ratio in dB")
    ap.add_argument("--lead", type=float, default=3.0)
    ap.add_argument("--tail", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.manifest)
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        speech = audio.load(src.parent / item["audio"])
        voiced = speech[np.abs(speech) > 0.02]
        speech_rms = float(np.sqrt(np.mean(voiced ** 2))) if len(voiced) else 0.1
        lead, tail = int(args.lead * SAMPLE_RATE), int(args.tail * SAMPLE_RATE)
        n = lead + len(speech) + tail
        noise = fan(n, rng) * speech_rms / 10 ** (args.snr / 20)
        noise[:lead] += events(lead, rng, speech_rms * 1.5)
        noise[lead + len(speech):] += events(tail, rng, speech_rms * 1.5)
        x = noise
        x[lead:lead + len(speech)] += speech
        x = (x / max(1.0, np.abs(x).max() / 0.95)).astype(np.float32)
        name = Path(item["audio"]).stem + "_noisy.wav"
        audio.save(out / "audio" / name, x)
        rows.append({**item, "audio": f"audio/{name}", "noise": {"snr_db": args.snr, "lead_s": args.lead,
                                                               "tail_s": args.tail}})
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                       encoding="utf-8")
    print(f"wrote {len(rows)} noisy items to {out / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
