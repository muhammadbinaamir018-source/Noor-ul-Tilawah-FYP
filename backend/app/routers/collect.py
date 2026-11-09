"""/api/collect: the recording-collection flow behind the /collect page."""

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ..schemas import CollectItem, NewSpeaker, SkipRequest, Speaker
from ..services import collection

router = APIRouter(prefix="/api/collect", tags=["collect"])


@router.get("/items", response_model=list[CollectItem])
def items() -> list[CollectItem]:
    return collection.items()


@router.post("/speakers", response_model=Speaker)
def new_speaker(body: NewSpeaker) -> Speaker:
    if not body.consent:
        raise HTTPException(422, "Consent is required before recording")
    return collection.new_speaker(body.ageGroup, body.native)


def _check(speaker_id: str, item_id: str) -> None:
    if not collection.speaker_exists(speaker_id):
        raise HTTPException(404, f"Unknown speaker: {speaker_id}")
    if item_id not in collection.item_ids():
        raise HTTPException(404, f"Not a recitable item of pages p01-p12: {item_id}")


@router.post("/recordings")
async def save_recording(audio: UploadFile, speaker_id: str = Form(...), item_id: str = Form(...)) -> dict:
    _check(speaker_id, item_id)
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(422, "Empty audio upload")
    return collection.save_recording(speaker_id, item_id, audio_bytes)


@router.post("/skip")
def skip(body: SkipRequest) -> dict:
    _check(body.speakerId, body.itemId)
    collection.skip(body.speakerId, body.itemId)
    return {"itemId": body.itemId, "status": "skipped"}


@router.get("/progress/{speaker_id}")
def progress(speaker_id: str) -> dict:
    if not collection.speaker_exists(speaker_id):
        raise HTTPException(404, f"Unknown speaker: {speaker_id}")
    return {"speakerId": speaker_id, "status": collection.progress(speaker_id)}


@router.get("/export")
def export() -> Response:
    """labels-template.csv for the qari, in the eval harness format."""
    text = collection.labels_template()
    return Response(text, media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="labels-template.csv"'})
