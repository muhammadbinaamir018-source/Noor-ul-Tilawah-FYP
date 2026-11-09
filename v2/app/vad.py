"""Voice activity detection and utterance segmentation.

Audio arrives in arbitrary chunks and is cut into utterance segments in two stages:
  * Boundaries: EnergyDetector, an adaptive noise-floor loudness threshold. It follows
    slow, melodic recitation well (long madd and ghunnah stay inside one segment).
  * Gate: SileroDetector, a small neural VAD (Silero, ONNX via onnxruntime), counts the
    seconds of real speech in each segment. Segments with too little (a fan, a bell, a
    knock) are discarded before the ASR sees them. Silero is not used for boundaries:
    trained on conversational speech, it rates tartil poorly (a third of Husary's frames
    reach 0.5) and cut words apart when it drew them.
VAD=silero (default) enables the gate, VAD=energy disables it. Without onnxruntime the
gate is off and a warning is logged.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SILERO_MODEL = Path(__file__).resolve().parent / "assets" / "silero_vad.onnx"
# Gate frames count as speech at this probability. Measured on 132 studio recitation
# segments vs household-noise segments: noise never exceeded 0.05 on any frame, while
# very drawn-out tartil (Husary's "إله الناس", 4.6 s) had only 2 frames at >= 0.5.
GATE_PROB = 0.3


def _db(f: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(f * f)) + 1e-5))


class EnergyDetector:
    frame = SAMPLE_RATE * 20 // 1000            # 20 ms

    def __init__(self, margin_db: float = 10.0, min_db: float = -50.0):
        self.margin_db, self.min_db = margin_db, min_db
        self.noise_db = -60.0

    def is_speech(self, f: np.ndarray) -> bool:
        db = _db(f)
        speech = db > self.noise_db + self.margin_db and db > self.min_db
        if not speech:
            self.noise_db = min(db, 0.95 * self.noise_db + 0.05 * db)
        return speech


@lru_cache(maxsize=1)
def _silero_session():
    import onnxruntime

    opts = onnxruntime.SessionOptions()
    opts.inter_op_num_threads = opts.intra_op_num_threads = 1
    return onnxruntime.InferenceSession(str(SILERO_MODEL), sess_options=opts, providers=["CPUExecutionProvider"])


class SileroDetector:
    """Silero VAD v6 on 32 ms frames, with hysteresis: speech starts above `start`,
    and continues until the probability falls below `stop`."""

    frame = 512                                  # 32 ms at 16 kHz, the model's frame size
    context = 64                                 # samples of the previous frame the model expects

    def __init__(self, start: float = 0.5, stop: float = 0.35):
        self.session = _silero_session()
        self.start, self.stop = start, stop
        self.reset()

    def reset(self) -> None:
        """Forget the recurrent state. Over a long stream of melodic recitation the state
        drifts and probabilities collapse (peak 0.52 vs 0.99 fresh on the same Husary
        audio), so the gate resets it for every segment, as Silero recommends."""
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros((1, self.context), np.float32)
        self.active = False

    def probability(self, f: np.ndarray) -> float:
        x = np.concatenate([self.ctx, f.reshape(1, -1).astype(np.float32)], axis=1)
        out, self.state = self.session.run(None, {"input": x, "state": self.state,
                                                  "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
        self.ctx = x[:, -self.context:]
        return float(out.reshape(-1)[0])

    def is_speech(self, f: np.ndarray) -> bool:
        p = self.probability(f)
        self.active = p >= (self.stop if self.active else self.start)
        return self.active


def make_gate(kind: str | None = None) -> "SileroDetector | None":
    if (kind or os.getenv("VAD", "silero")).lower() != "silero":
        return None
    try:
        return SileroDetector()
    except Exception as e:                       # onnxruntime missing or model unreadable
        log.warning("Silero speech gate unavailable (%s); noise will reach the ASR", e)
        return None


class Segmenter:
    """Energy-based segmentation, optionally gated by a neural speech detector.

    With a gate, a segment is released (and current() reports it) only once the gate has
    found at least `min_gate_speech_s` seconds of frames with speech probability >= GATE_PROB.
    Energy frames close a segment after `end_silence_ms` of quiet following enough loud
    frames, or cut it at the quietest point before it outgrows the ASR window.
    """

    def __init__(self, detector=None, end_silence_ms: int = 600, min_speech_ms: int = 250,
                 max_segment_s: float = 20.0, preroll_ms: int = 300, gate="default",
                 min_gate_speech_s: float = 0.15):
        self.detector = detector or EnergyDetector()
        self.gate = make_gate() if gate == "default" else gate
        self.min_gate_frames = max(1, round(min_gate_speech_s * SAMPLE_RATE / SileroDetector.frame))
        self.gate_frames = 0                     # gate speech frames seen in the open segment
        self.rejected = 0                        # segments discarded by the gate
        self._gate_rest = np.zeros(0, np.float32)
        self.frame = self.detector.frame
        frame_ms = 1000 * self.frame / SAMPLE_RATE
        self.end_silence = round(end_silence_ms / frame_ms)
        self.min_speech = max(1, round(min_speech_ms / frame_ms))
        self.max_frames = int(max_segment_s * 1000 / frame_ms)
        self.preroll = round(preroll_ms / frame_ms)
        self._reset_segment()
        self._rest = np.zeros(0, np.float32)

    def _reset_segment(self, keep: np.ndarray | None = None, keep_db: list[float] | None = None,
                       keep_speech: list[bool] | None = None):
        self.buf = keep if keep is not None else np.zeros(0, np.float32)
        self.db = keep_db or []
        self.speech = keep_speech or []
        self.speech_frames = sum(self.speech)
        self.silence_run = 0

    @property
    def has_speech(self) -> bool:
        return self.speech_frames >= self.min_speech

    @property
    def gate_ok(self) -> bool:
        return self.gate is None or self.gate_frames >= self.min_gate_frames

    def current(self) -> np.ndarray | None:
        """Audio of the open segment, if it contains speech."""
        return self.buf if self.has_speech and self.gate_ok else None

    def _gated(self, seg: np.ndarray | None) -> np.ndarray | None:
        """Release a closed segment only if the gate heard speech in it; reset the count."""
        ok = self.gate_ok
        self._reset_gate()
        if seg is not None and not ok:
            self.rejected += 1
            log.info("discarded %.1fs segment without speech (noise)", len(seg) / SAMPLE_RATE)
            return None
        return seg

    def _reset_gate(self) -> None:
        self.gate_frames = 0
        if self.gate is not None:
            self.gate.reset()

    def _feed_gate(self, x: np.ndarray) -> None:
        g = np.concatenate([self._gate_rest, x])
        n = len(g) // self.gate.frame
        for k in range(n):
            if self.gate.probability(g[k * self.gate.frame:(k + 1) * self.gate.frame]) >= GATE_PROB:
                self.gate_frames += 1
        self._gate_rest = g[n * self.gate.frame:]

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        samples = samples.astype(np.float32)
        if self.gate is not None:
            self._feed_gate(samples)
        x = np.concatenate([self._rest, samples])
        n = len(x) // self.frame
        self._rest = x[n * self.frame:]
        done: list[np.ndarray] = []
        for k in range(n):
            seg = self._frame(x[k * self.frame:(k + 1) * self.frame])
            if seg is not None and (seg := self._gated(seg)) is not None:
                done.append(seg)
        return done

    def flush(self) -> np.ndarray | None:
        seg = self.buf if self.has_speech else None
        self._reset_segment()
        return self._gated(seg)

    def _frame(self, f: np.ndarray) -> np.ndarray | None:
        speech = self.detector.is_speech(f)
        self.buf = np.concatenate([self.buf, f])
        self.db.append(_db(f))
        self.speech.append(speech)

        if speech:
            self.speech_frames += 1
            self.silence_run = 0
        else:
            self.silence_run += 1

        if not self.has_speech:
            if self.silence_run > self.preroll:
                # No utterance yet: keep only a short pre-roll of audio (and forget any
                # gate speech heard before it).
                self.buf = self.buf[-self.preroll * self.frame:]
                self.db = self.db[-self.preroll:]
                self.speech = self.speech[-self.preroll:]
                self.speech_frames = 0
                self._reset_gate()
            return None

        if self.silence_run >= self.end_silence:
            keep = self.silence_run - self.preroll // 2
            seg = self.buf[:len(self.buf) - keep * self.frame]
            self._reset_segment()
            return seg

        if len(self.db) >= self.max_frames:
            # Too long for one ASR window: cut at the quietest frame of the last quarter.
            tail = self.db[-self.max_frames // 4:]
            cut = len(self.db) - len(tail) + int(np.argmin(tail))
            seg = self.buf[:cut * self.frame]
            self._reset_segment(self.buf[cut * self.frame:], self.db[cut:], self.speech[cut:])
            return seg
        return None
