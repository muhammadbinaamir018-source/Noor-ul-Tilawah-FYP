"""/api/progress: a learner's tallies for the home page (services/progress.py)."""

from fastapi import APIRouter, HTTPException

from ..services import progress

router = APIRouter(prefix="/api/progress", tags=["progress"])


def _id(learner_id: str) -> str:
    try:
        return progress.check_id(learner_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("/{learner_id}")
def summary(learner_id: str) -> dict:
    lid = _id(learner_id)
    return progress.summarise(lid, progress.load(lid), progress.items_index())


@router.delete("/{learner_id}")
def reset(learner_id: str) -> dict:
    progress.reset(_id(learner_id))
    return {"learnerId": learner_id, "reset": True}
