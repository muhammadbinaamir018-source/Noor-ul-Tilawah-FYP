"""Model service: runs the heavy models where the hardware is (GPU box, Colab, Kaggle, a Space).

The app server (app.server) streams audio, aligns, verifies and scores letters itself;
it only asks this service for model outputs, so it needs no torch.

  GET  /info       -> models, device, and the CTC vocabulary (for letter scoring)
  POST /asr        body: int16 little-endian 16 kHz mono PCM -> {"text": ...}
  POST /logprobs   same body -> float32 T x V CTC log-posteriors, shape in X-Shape: "T,V"

Protect it with MODEL_SERVICE_TOKEN when it is reachable from outside (tunnels, Spaces):
clients then send "Authorization: Bearer <token>".

Run:  python -m uvicorn app.model_service:app --host 0.0.0.0 --port 9000
Env:  ASR_BACKEND, ASR_MODEL, PRON_MODEL, PRON_INT8, PRONUNCIATION, DEVICE (see app.models)
"""

import hmac
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response

from . import models

log = logging.getLogger("model_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_AUDIO_SECONDS = 60


def create_app(asr=None, scorer=None, token: str | None = None, load: bool = True) -> FastAPI:
    """Build the service. Pass `asr`/`scorer` to serve given objects (tests); otherwise
    they are loaded from the environment at startup."""
    state = {"asr": asr, "scorer": scorer}
    token = os.getenv("MODEL_SERVICE_TOKEN") if token is None else token
    # One inference per model at a time: bounds GPU/CPU memory under concurrent sessions.
    asr_lock, pron_lock = threading.Lock(), threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if load and state["asr"] is None:
            state["asr"] = models.load_local_asr()
            state["scorer"] = models.load_local_scorer()
        yield

    app = FastAPI(title="Recitation model service", lifespan=lifespan)

    def auth(authorization: str | None = Header(default=None)) -> None:
        if token and not hmac.compare_digest(authorization or "", f"Bearer {token}"):
            raise HTTPException(401, "missing or wrong bearer token")

    async def pcm(request: Request) -> np.ndarray:
        # int16 on the wire: the browser captures int16 anyway, and it halves the upload
        # (partial decodes re-send the growing segment about once a second).
        body = await request.body()
        if not body or len(body) % 2:
            raise HTTPException(400, "body must be int16 PCM")
        if len(body) > MAX_AUDIO_SECONDS * 16000 * 2:
            raise HTTPException(413, f"audio longer than {MAX_AUDIO_SECONDS} s")
        return np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0

    @app.get("/info", dependencies=[Depends(auth)])
    def info() -> dict:
        sc = state["scorer"]
        return {
            "asr": models.ASR_MODEL if state["asr"] is not None else None,
            "pronunciation": models.PRON_MODEL if sc is not None else None,
            "device": models.DEVICE,
            "vocab": sc.vocab if sc is not None else None,
            "blank": sc.blank if sc is not None else None,
        }

    @app.post("/asr", dependencies=[Depends(auth)])
    async def asr_endpoint(audio: np.ndarray = Depends(pcm)) -> dict:
        if state["asr"] is None:
            raise HTTPException(503, "ASR model not loaded")
        t = time.perf_counter()
        text = await _run(asr_lock, state["asr"].transcribe, audio)
        log.info("asr %.1fs audio in %.2fs", len(audio) / 16000, time.perf_counter() - t)
        return {"text": text}

    @app.post("/logprobs", dependencies=[Depends(auth)])
    async def logprobs_endpoint(audio: np.ndarray = Depends(pcm)) -> Response:
        if state["scorer"] is None:
            raise HTTPException(503, "pronunciation model not loaded")
        t = time.perf_counter()
        lp = await _run(pron_lock, state["scorer"].log_probs, audio)
        log.info("logprobs %.1fs audio in %.2fs", len(audio) / 16000, time.perf_counter() - t)
        return Response(np.ascontiguousarray(lp, dtype="<f4").tobytes(), media_type="application/octet-stream",
                        headers={"X-Shape": f"{lp.shape[0]},{lp.shape[1]}"})

    return app


async def _run(lock: threading.Lock, fn, *args):
    """Run blocking inference in a worker thread, one call per model at a time."""
    import asyncio

    def locked():
        with lock:
            return fn(*args)

    return await asyncio.get_running_loop().run_in_executor(None, locked)


app = create_app()
