from fastapi import APIRouter, HTTPException

from ..schemas import PageContent, PageStatus
from ..services.content_store import store

router = APIRouter(prefix="/api/editor", tags=["editor"])


@router.put("/pages/{page_id}", response_model=PageContent)
def save_page(page_id: str, page: PageContent) -> PageContent:
    if page.id != page_id:
        raise HTTPException(422, "Body id does not match URL page id")
    try:
        store.save_page(page)
    except KeyError:
        raise HTTPException(404, f"Unknown page: {page_id}")
    return page


@router.get("/status", response_model=list[PageStatus])
def status() -> list[PageStatus]:
    return store.status()


@router.get("/clips/{audio_page}", response_model=list[str])
def clips(audio_page: str) -> list[str]:
    return store.clips_for_page(audio_page)
