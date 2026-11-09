from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import settings
from ..schemas import Manifest, PageContent
from ..services.content_store import store

router = APIRouter(prefix="/api", tags=["content"])


@router.get("/content/manifest", response_model=Manifest)
def get_manifest() -> Manifest:
    return store.manifest


@router.get("/content/pages/{page_id}", response_model=PageContent)
def get_page(page_id: str) -> PageContent:
    try:
        return store.load_page(page_id)
    except KeyError:
        raise HTTPException(404, f"Unknown page: {page_id}")


@router.get("/content/image/{filename}")
def get_image(filename: str) -> FileResponse:
    path = (settings.images_dir / filename).resolve()
    if path.parent != settings.images_dir.resolve() or not path.exists():
        raise HTTPException(404, f"Unknown image: {filename}")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/audio/{clip}")
def get_audio(clip: str) -> FileResponse:
    path = (settings.audio_dir / clip).resolve()
    if path.parent != settings.audio_dir.resolve() or not path.exists():
        raise HTTPException(404, f"Unknown clip: {clip}")
    return FileResponse(path, media_type="audio/mpeg")
