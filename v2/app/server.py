"""FastAPI server: reference text over REST, live assessment over a WebSocket.

WebSocket protocol (/ws):
  client -> {"type": "start", "surah": 1, "from": 1, "to": 7}
  server -> {"type": "ready", "reference": [...]}
  client -> binary frames: little-endian int16 PCM, 16 kHz mono
  server -> {"type": "state", ...}           (repeatedly, as feedback updates)
  server -> {"type": "pronunciation", ...}   (async letter scores)
  client -> {"type": "stop"}
  server -> {"type": "final", ...}
"""

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import models, quran
from .asr import ASR
from .pronunciation import DEFAULT_LEVEL, LEVELS, CtcScorer
from .recorder import SessionRecorder
from .session import RecitationSession

log = logging.getLogger("recitation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

WEB = Path(__file__).resolve().parent.parent / "web"


class Models:
    asr: ASR | None = None
    scorer: CtcScorer | None = None
    status = {"asr": "loading", "pronunciation": "loading" if models.PRON_ENABLED else "disabled"}

    @classmethod
    def load(cls) -> None:
        # Local models, or clients for app.model_service when MODEL_SERVICE_URL is set.
        try:
            cls.asr = models.load_asr_model()
            cls.status["asr"] = "ready"
        except Exception as e:
            log.exception("ASR model failed to load")
            cls.status["asr"] = f"error: {e}"
        if models.PRON_ENABLED:
            try:
                cls.scorer = models.load_scorer()
                cls.status["pronunciation"] = "ready" if cls.scorer is not None else "disabled"
            except Exception as e:
                log.exception("pronunciation model failed to load")
                cls.status["pronunciation"] = f"error: {e}"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load in the background so the UI is reachable while models download.
    threading.Thread(target=Models.load, daemon=True).start()
    yield


app = FastAPI(title="Quran Recitation Assessment", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {"models": Models.status, **models.describe()}


@app.get("/api/surahs")
def surahs() -> list[dict]:
    return quran.surahs()


@app.get("/api/reference")
def reference(surah: int, start: int = 1, end: int | None = None) -> list[dict]:
    try:
        return [w.to_dict() for w in quran.reference(surah, start, end)]
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    await socket.accept()
    session: RecitationSession | None = None
    recorder: SessionRecorder | None = None
    lock = asyncio.Lock()                      # serialises sends from concurrent tasks

    async def emit(msg: dict) -> None:
        async with lock:
            await socket.send_text(json.dumps(msg, ensure_ascii=False))

    try:
        while True:
            msg = await socket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if session is not None:
                    recorder.add(msg["bytes"])
                    pcm = np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768.0
                    await session.feed(pcm)
                continue
            data = json.loads(msg.get("text") or "{}")
            if data.get("type") == "start":
                if Models.asr is None:
                    await emit({"type": "error", "message": f"ASR model is {Models.status['asr']}"})
                    continue
                try:
                    ref = quran.reference(int(data["surah"]), int(data.get("from", 1)),
                                          data.get("to") and int(data["to"]))
                except (KeyError, ValueError) as e:
                    await emit({"type": "error", "message": str(e)})
                    continue
                level = data.get("level") if data.get("level") in LEVELS else DEFAULT_LEVEL
                if recorder is not None:           # a new start without stop: keep what we had
                    rec, recorder = recorder, None
                    await asyncio.to_thread(rec.save, session.state() if session else None, "restarted")
                session = RecitationSession(ref, Models.asr, emit, Models.scorer, level=level)
                recorder = SessionRecorder(ref[0].surah, ref[0].ayah, ref[-1].ayah, level, models.describe())
                await emit({"type": "ready", "reference": [w.to_dict() for w in ref]})
                await emit(session.state())
            elif data.get("type") == "stop" and session is not None:
                final = await session.finish()
                # The browser closes the socket as soon as it gets the final result, which
                # cancels this handler: detach the recorder first so it is saved exactly once.
                rec, session, recorder = recorder, None, None
                await asyncio.to_thread(rec.save, final, "stopped")
    except WebSocketDisconnect:
        pass
    finally:
        if recorder is not None:                   # browser closed mid-recitation
            await asyncio.to_thread(recorder.save, session.state() if session else None, "disconnected")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
