"""Madd: is the long vowel held for the right number of counts?

How it works
1. The recording (webm/opus from the browser, or any file PyAV can decode) is
   decoded to mono 16 kHz samples.
2. A loudness envelope is taken every 10 ms (RMS). Frames louder than −30 dB
   relative to the loudest frame are "voiced"; runs of voiced frames are
   segments, gaps shorter than 60 ms inside a run are bridged (a flutter in
   the voice is not a pause), and runs shorter than 40 ms are dropped (clicks).
3. The longest segment is the vowel being judged. Speaking speed differs
   between learners, so the length is expressed in *counts* (harakat): the
   shortest other segment of at least 60 ms is taken as one harakah. If the
   clip has no such reference, a default harakah of 0.2 s is used and the
   confidence is capped at 0.5.
4. counts = vowel length / harakah → too-short (< 1.5), 2-count (1.5–3),
   4-count (3–5.5), too-long (≥ 5.5). The confidence says how far the
   measurement sits from the nearest class boundary.

Limitations: the envelope cannot separate two vowels inside one word with no
consonant dip between them, and the default harakah is a guess at speaking
speed. The thresholds were set on synthetic vowels and the publisher's clips;
they must be re-tuned on labelled learner recordings (see docs/TODO-testing.md).
"""

import io
from dataclasses import dataclass

import av
import numpy as np

SAMPLE_RATE = 16000
FRAME_S = 0.010          # envelope resolution
VOICED_DB = -30.0        # relative to the loudest frame
SILENCE_RMS = 1e-3       # absolute floor (about −60 dBFS): below this nothing is voiced
MIN_SEGMENT_S = 0.04
BRIDGE_GAP_S = 0.06
MIN_REFERENCE_S = 0.06
REFERENCE_RANGE_S = (0.12, 0.35)   # a plausible harakah; outside this the segment is not one
DEFAULT_HARAKAH_S = 0.20
MIN_AUDIO_S = 0.15
CLIP_LEVEL, CLIP_FRACTION = 0.99, 0.01

# count bands: (upper bound, verdict). The last band has no upper bound.
BANDS = [(1.5, "too-short"), (3.0, "2-count"), (5.5, "4-count"), (float("inf"), "too-long")]


@dataclass
class Measurement:
    segments: list[tuple[float, float]]   # (start_s, length_s) of each voiced run
    longest_s: float
    reference_s: float
    reference_is_default: bool


@dataclass
class Classification:
    counts: float
    verdict: str
    ok: bool
    confidence: float
    hint: str


@dataclass
class MaddResult:
    measured_s: float
    reference_s: float
    reference_is_default: bool
    counts: float
    expected_counts: int
    verdict: str          # too-short | 2-count | 4-count | too-long | unclear
    ok: bool
    confidence: float
    hint: str
    clipped: bool


# ---- audio ------------------------------------------------------------------------

def decode(audio_bytes: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Any container PyAV understands -> mono float32 samples at `sr`."""
    chunks = []
    with av.open(io.BytesIO(audio_bytes)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray()[0])
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def measure(samples: np.ndarray, sr: int = SAMPLE_RATE) -> Measurement:
    hop = int(sr * FRAME_S)
    n = len(samples) // hop
    if n == 0:
        return Measurement([], 0.0, DEFAULT_HARAKAH_S, True)
    frames = samples[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    peak = rms.max()
    if peak < SILENCE_RMS:
        return Measurement([], 0.0, DEFAULT_HARAKAH_S, True)
    voiced = (20 * np.log10(rms / peak + 1e-12) > VOICED_DB) & (rms > SILENCE_RMS)

    # bridge short gaps, then drop short runs
    bridge = int(BRIDGE_GAP_S / FRAME_S)
    runs = _runs(voiced)
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= bridge:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    segments = [(a * FRAME_S, (b - a) * FRAME_S) for a, b in merged if (b - a) * FRAME_S >= MIN_SEGMENT_S]
    if not segments:
        return Measurement([], 0.0, DEFAULT_HARAKAH_S, True)

    longest = max(segments, key=lambda s: s[1])
    others = [s[1] for s in segments if s is not longest and s[1] >= MIN_REFERENCE_S]
    if others:
        lo, hi = REFERENCE_RANGE_S
        reference = float(min(max(min(others), lo), hi))
        return Measurement(segments, round(longest[1], 3), round(reference, 3), False)
    return Measurement(segments, round(longest[1], 3), DEFAULT_HARAKAH_S, True)


def is_clipped(samples: np.ndarray) -> bool:
    return len(samples) > 0 and float((np.abs(samples) >= CLIP_LEVEL).mean()) > CLIP_FRACTION


# ---- classification -----------------------------------------------------------------

def classify(longest_s: float, reference_s: float, expected_counts: int) -> Classification:
    counts = longest_s / reference_s
    lower = 0.0
    for upper, verdict in BANDS:
        if counts < upper:
            break
        lower = upper
    ok = verdict == f"{expected_counts}-count"
    # confidence: how deep inside the band the measurement sits (0 at an edge)
    if verdict == "too-short":
        confidence = (upper - counts) / 0.75
    elif verdict == "too-long":
        confidence = (counts - lower) / 2.0
    else:
        half = (upper - lower) / 2
        confidence = 1 - abs(counts - (lower + half)) / half
    confidence = float(min(1.0, max(0.0, confidence)))

    held = f"you held it about {counts:.1f}"
    if ok:
        hint = f"Good madd: about {expected_counts} counts."
    elif verdict in ("too-short", "2-count") and expected_counts > counts:
        hint = f"Hold the vowel longer: it should be {expected_counts} counts, {held}."
    else:
        hint = f"Make the vowel shorter: it should be {expected_counts} counts, {held}."
    return Classification(round(counts, 2), verdict, ok, round(confidence, 2), hint)


def analyse(audio_bytes: bytes, expected_counts: int) -> MaddResult:
    samples = decode(audio_bytes)

    def unclear(hint: str) -> MaddResult:
        return MaddResult(0.0, DEFAULT_HARAKAH_S, True, 0.0, expected_counts, "unclear", False, 0.0, hint, is_clipped(samples))

    if len(samples) < MIN_AUDIO_S * SAMPLE_RATE:
        return unclear("The recording is too short to measure; try again.")
    m = measure(samples)
    if not m.segments:
        return unclear("No voice was detected; speak closer to the microphone.")
    c = classify(m.longest_s, m.reference_s, expected_counts)
    confidence = min(c.confidence, 0.5) if m.reference_is_default else c.confidence
    clipped = is_clipped(samples)
    hint = c.hint + (" (The recording is clipping: move a little away from the microphone.)" if clipped else "")
    return MaddResult(m.longest_s, m.reference_s, m.reference_is_default, c.counts, expected_counts,
                      c.verdict, c.ok, confidence, hint, clipped)


# ---- what the item asks for ---------------------------------------------------------

FATHA, DAMMA, KASRA, SUKUN, SHADDA = "َ", "ُ", "ِ", "ْ", "ّ"
DAGGER_ALIF, MADDA_SIGN, ALIF_MADDA = "ٰ", "ٓ", "آ"


def expected_counts_for(text: str) -> int | None:
    """How many counts the item's long vowel should be held, from its tashkeel:
    4 for a madd sign (ٓ / آ), 2 for a natural madd (fatha+alif, damma+waw
    sakin, kasra+ya sakin, dagger alif), None when the item has no madd or has
    no tashkeel to tell."""
    if MADDA_SIGN in text or ALIF_MADDA in text:
        return 4
    chars = [c for c in text if c != SHADDA]
    for i, c in enumerate(chars):
        prev = chars[i - 1] if i else ""
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if c == DAGGER_ALIF:
            return 2
        if c == "ا" and prev == FATHA:
            return 2
        if c == "و" and prev == DAMMA and nxt == SUKUN:
            return 2
        if c == "ي" and prev == KASRA and nxt == SUKUN:
            return 2
    return None
