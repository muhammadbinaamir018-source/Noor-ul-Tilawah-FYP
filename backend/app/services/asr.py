"""faster-whisper singleton for Quran-tuned transcription."""

import io
import logging
import os
import threading

from ..config import settings

logger = logging.getLogger(__name__)

_model = None
_load_lock = threading.Lock()
_transcribe_lock = threading.Lock()
_load_error: str | None = None


def is_remote() -> bool:
    return settings.asr_backend == "remote" and bool(settings.asr_remote_url)


def load_model() -> None:
    """Load (downloading on first run) the CTranslate2 Whisper model. Idempotent."""
    global _model, _load_error
    if is_remote():
        return
    with _load_lock:
        if _model is not None:
            return
        try:
            from faster_whisper import WhisperModel

            logger.info("Loading ASR model %s ...", settings.resolved_asr_model)
            _model = WhisperModel(
                settings.resolved_asr_model,
                device=settings.asr_device,
                compute_type=settings.asr_compute_type,
                cpu_threads=settings.asr_cpu_threads or (os.cpu_count() or 4),
            )
            logger.info("ASR model loaded")
        except Exception as exc:  # pragma: no cover
            _load_error = str(exc)
            logger.exception("Failed to load ASR model")
            raise


def preload_async() -> None:
    threading.Thread(target=_safe_load, daemon=True).start()


def _safe_load() -> None:
    if is_remote():
        return
    try:
        load_model()
        # Warm up: the first transcribe pays one-off allocation costs (~20s),
        # so spend them here instead of on the learner's first recitation.
        import numpy as np

        with _transcribe_lock:
            segments, _ = _model.transcribe(np.zeros(8000, dtype=np.float32), language="ar", beam_size=1)
            list(segments)
        logger.info("ASR warm-up complete")
    except Exception:
        pass  # already logged; surfaced via is_loaded()/_load_error


def is_loaded() -> bool:
    return is_remote() or _model is not None


def load_error() -> str | None:
    return _load_error


def _transcribe_remote(audio_bytes: bytes) -> str:
    import requests

    url = settings.asr_remote_url.rstrip("/") + "/transcribe"
    resp = requests.post(
        url,
        files={"audio": ("recitation.webm", audio_bytes, "audio/webm")},
        headers={"X-Tilawah-Token": settings.asr_remote_token},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["transcription"]


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe a short clip (any container PyAV can decode: webm/opus, wav, mp3)."""
    if is_remote():
        return _transcribe_remote(audio_bytes)
    load_model()
    # vad_filter stays off: VAD tends to swallow sub-second single-letter utterances.
    with _transcribe_lock:
        segments, _info = _model.transcribe(
            io.BytesIO(audio_bytes),
            language="ar",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            # Short letter clips make the decoder loop ("د د د د…" for 30s+);
            # blocking repeated trigrams ends decoding after the first few tokens.
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
