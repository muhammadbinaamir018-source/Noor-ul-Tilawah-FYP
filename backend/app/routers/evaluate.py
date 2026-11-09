import time
from dataclasses import asdict

import anyio
from fastapi import APIRouter, Form, HTTPException, UploadFile
from tajweed import madd

from ..config import settings
from ..schemas import EvaluateResult, MaddInfo
from ..services import asr, eval_log, progress
from ..services.content_store import store
from ..services.feedback import build_feedback
from ..services.matching import evaluate_match

router = APIRouter(prefix="/api", tags=["evaluate"])


@router.post("/evaluate", response_model=EvaluateResult)
async def evaluate(audio: UploadFile, region_id: str = Form(...), learner_id: str | None = Form(None)) -> EvaluateResult:
    region = store.find_region(region_id)
    if region is None:
        raise HTTPException(404, f"Unknown region: {region_id}")
    if learner_id:
        try:
            progress.check_id(learner_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
    if region.expected is None:
        raise HTTPException(422, f"Region {region_id} has no expected text; recitation checking is not enabled for it")
    if not asr.is_loaded():
        detail = asr.load_error() or "ASR model is still loading, try again shortly"
        raise HTTPException(503, detail)

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(422, "Empty audio upload")

    start = time.perf_counter()
    transcription = await anyio.to_thread.run_sync(asr.transcribe, audio_bytes)
    latency_ms = int((time.perf_counter() - start) * 1000)

    result = evaluate_match(transcription, region.expected)
    response = EvaluateResult(
        regionId=region_id,
        transcription=transcription,
        expectedText=region.expected.text,
        latencyMs=latency_ms,
        **result,
    )
    # Madd is measured on the audio, independently of the text verdict.
    # Letter names (با for ب) are read long by convention, so they are skipped.
    if settings.madd_enabled and region.expected.mode != "letter-name":
        counts = madd.expected_counts_for(region.expected.text)
        if counts:
            info = await anyio.to_thread.run_sync(madd.analyse, audio_bytes, counts)
            response.madd = MaddInfo(**asdict(info))
    if response.verdict != "correct" or (response.madd is not None and not response.madd.ok):
        response.feedback = build_feedback(result, region.expected, response.madd)
    if learner_id:
        # the letter to blame: what the feedback pointed at, else the item's own letter
        letter = (response.feedback.expectedLetter if response.feedback and response.feedback.expectedLetter
                  else progress.item_letter(region.expected))
        progress.save(learner_id, progress.apply_attempt(progress.load(learner_id), region_id,
                                                         response.verdict, response.score, letter))
    eval_log.record(
        audio_bytes,
        {
            **response.model_dump(),
            "variants": region.expected.variants,
            "mode": region.expected.mode,
        },
    )
    return response
