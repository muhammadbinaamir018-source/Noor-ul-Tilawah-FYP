"""Tilawah remote transcription server.

Runs the Quran-tuned Whisper model on whatever hardware is available (NVIDIA
GPU on Colab/Kaggle, CPU elsewhere) and exposes a single /transcribe endpoint.
The Tilawah backend does all matching/verdict logic locally — this service
only turns audio into Arabic text.

Auth: every request must carry the X-Tilawah-Token header. The token is
generated at startup (or taken from TILAWAH_TOKEN) and printed to stdout.
"""

import io
import os
import secrets

import uvicorn
from fastapi import FastAPI, Header, HTTPException, UploadFile
from faster_whisper import WhisperModel

MODEL_ID = os.environ.get("TILAWAH_GPU_MODEL", "OdyAsh/faster-whisper-base-ar-quran")
TOKEN = os.environ.get("TILAWAH_TOKEN") or secrets.token_urlsafe(16)
PORT = int(os.environ.get("TILAWAH_GPU_PORT", "9000"))


def pick_device() -> tuple[str, str, int]:
    """(device, compute_type, beam_size) — best compute type the GPU supports.

    fp16 needs tensor-core era GPUs (T4 and newer, compute capability >= 7.0);
    older cards like Kaggle's P100 fall back to float32, which is still fast.
    """
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            supported = ctranslate2.get_supported_compute_types("cuda")
            for ct in ("float16", "int8_float16", "int8_float32", "float32"):
                if ct in supported:
                    return "cuda", ct, 5
    except Exception:
        pass
    return "cpu", "int8", 1


DEVICE, COMPUTE, BEAM = pick_device()
print(f"Loading {MODEL_ID} on {DEVICE} ({COMPUTE}, beam={BEAM}) ...")
model = WhisperModel(MODEL_ID, device=DEVICE, compute_type=COMPUTE)
# Warm up so the first real request is fast.
import numpy as np

list(model.transcribe(np.zeros(8000, dtype=np.float32), language="ar", beam_size=1)[0])
print("Model ready.")

app = FastAPI(title="Tilawah GPU server")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE, "beamSize": BEAM}


@app.post("/transcribe")
async def transcribe(audio: UploadFile, x_tilawah_token: str = Header(default="")) -> dict:
    if x_tilawah_token != TOKEN:
        raise HTTPException(401, "Bad or missing X-Tilawah-Token header")
    data = await audio.read()
    if not data:
        raise HTTPException(422, "Empty audio upload")
    segments, _ = model.transcribe(
        io.BytesIO(data),
        language="ar",
        beam_size=BEAM,
        vad_filter=False,
        condition_on_previous_text=False,
        no_repeat_ngram_size=3,
        repetition_penalty=1.05,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {"transcription": text}


if __name__ == "__main__":
    print(f"\n=== TILAWAH_TOKEN: {TOKEN} ===\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
