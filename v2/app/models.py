"""Model loading from the environment, shared by the app server, the model service and the tools.

MODEL_SERVICE_URL set   -> models run in app.model_service at that URL (this process loads no torch)
MODEL_SERVICE_URL unset -> models run in this process
"""

import logging
import os

from .asr import ASR, DEFAULT_ASR_MODELS, load_asr
from .pronunciation import CtcScorer

log = logging.getLogger(__name__)

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "").strip() or None
MODEL_SERVICE_TOKEN = os.getenv("MODEL_SERVICE_TOKEN") or None
ASR_BACKEND = os.getenv("ASR_BACKEND", "faster-whisper")
ASR_MODEL = os.getenv("ASR_MODEL", DEFAULT_ASR_MODELS.get(ASR_BACKEND, ""))
PRON_MODEL = os.getenv("PRON_MODEL", "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final")
PRON_ENABLED = os.getenv("PRONUNCIATION", "1") != "0"
PRON_THRESHOLD = float(os.getenv("PRON_THRESHOLD", "0.35"))
PRON_INT8 = os.getenv("PRON_INT8", "0") == "1"
DEVICE = os.getenv("DEVICE", "cpu")


def load_local_asr() -> ASR:
    return load_asr(ASR_BACKEND, ASR_MODEL, DEVICE)


def load_local_scorer() -> CtcScorer | None:
    if not PRON_ENABLED:
        return None
    from .pronunciation import PronunciationScorer

    return PronunciationScorer(PRON_MODEL, DEVICE, PRON_THRESHOLD, PRON_INT8)


def load_asr_model() -> ASR:
    if MODEL_SERVICE_URL:
        from .remote import ModelServiceClient, RemoteASR

        client = ModelServiceClient(MODEL_SERVICE_URL, MODEL_SERVICE_TOKEN)
        info = client.info()
        if not info.get("asr"):
            raise RuntimeError(f"model service at {MODEL_SERVICE_URL} has no ASR model loaded")
        log.info("using remote ASR %s on %s at %s", info["asr"], info["device"], MODEL_SERVICE_URL)
        return RemoteASR(client)
    return load_local_asr()


def load_scorer() -> CtcScorer | None:
    if not PRON_ENABLED:
        return None
    if MODEL_SERVICE_URL:
        from .remote import ModelServiceClient, RemotePronunciation

        client = ModelServiceClient(MODEL_SERVICE_URL, MODEL_SERVICE_TOKEN)
        info = client.info()
        if not info.get("pronunciation"):
            log.warning("model service at %s has no pronunciation model; letter scoring off", MODEL_SERVICE_URL)
            return None
        log.info("using remote pronunciation model %s on %s", info["pronunciation"], info["device"])
        return RemotePronunciation(client, info["vocab"], info["blank"], PRON_THRESHOLD)
    return load_local_scorer()


def describe() -> dict:
    """What /api/health reports about where models run."""
    if MODEL_SERVICE_URL:
        return {"where": "remote", "model_service": MODEL_SERVICE_URL}
    return {"where": "local", "asr_backend": ASR_BACKEND, "asr_model": ASR_MODEL,
            "pronunciation_model": PRON_MODEL if PRON_ENABLED else None, "device": DEVICE}
