from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import settings
from ..services import eval_log

router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("")
def list_logs(limit: int = 100) -> list[dict]:
    return eval_log.entries(limit)


@router.get("/audio/{name}")
def log_audio(name: str) -> FileResponse:
    path = (settings.eval_log_dir / name).resolve()
    if path.parent != settings.eval_log_dir.resolve() or not path.exists():
        raise HTTPException(404, f"Unknown recording: {name}")
    return FileResponse(path, media_type="audio/webm")


@router.delete("")
def clear_logs() -> dict:
    return {"deleted": eval_log.clear()}
