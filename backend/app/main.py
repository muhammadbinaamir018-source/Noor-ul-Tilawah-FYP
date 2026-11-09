import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import collect, content, editor, evaluate, logs, progress
from .services import asr
from .services.content_store import store

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    if settings.asr_preload:
        asr.preload_async()
    yield


app = FastAPI(title="Tilawah-AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(content.router)
app.include_router(evaluate.router)
app.include_router(editor.router)
app.include_router(logs.router)
app.include_router(collect.router)
app.include_router(progress.router)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "backend": settings.asr_backend,
        "model": "remote-gpu" if asr.is_remote() else settings.asr_model,
        "modelLoaded": asr.is_loaded(),
        "modelError": asr.load_error(),
        "lessons": len(store.manifest.lessons),
    }
