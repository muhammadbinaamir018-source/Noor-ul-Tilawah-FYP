"""Learner recording collection for the evaluation dataset.

How it works: a volunteer gives consent on /collect, gets an anonymous speaker
id (S01, S02, ...) and records every recitable item of pages p01–p12 in
reading order. Each recording is saved as data/recordings/<speaker>/<item>.webm
and listed in data/recordings/metadata.csv; speakers.csv holds the consent
flag and two demographic answers (age group, native Arabic speaker). Nothing
identifying is stored. labels_template() writes labels-template.csv in the
format eval/run_eval.py reads, with the label column blank for a qari to fill.
"""

import csv
import time
from pathlib import Path

from ..config import settings
from ..schemas import CollectItem, Speaker
from .content_store import store

SCOPE_PAGES = [f"p{n:02d}" for n in range(1, 13)]
SPEAKERS = "speakers.csv"
METADATA = "metadata.csv"
TEMPLATE = "labels-template.csv"
SPEAKER_FIELDS = ["speaker_id", "created", "consent", "age_group", "native"]
META_FIELDS = ["speaker_id", "item_id", "file", "status", "updated", "bytes"]


def _root() -> Path:
    root = Path(settings.recordings_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---- items ------------------------------------------------------------------------

def items() -> list[CollectItem]:
    """Recitable items of pages p01–p12, rows top to bottom, right to left."""
    out = []
    for page_id in SCOPE_PAGES:
        page = store.load_page(page_id)
        for r in sorted(page.regions, key=lambda r: (r.row, -r.rect.x)):
            if r.expected:
                out.append(CollectItem(id=r.id, pageId=page_id, image=page.image, rect=r.rect,
                                       text=r.expected.text, mode=r.expected.mode, audio=r.audio))
    return out


def item_ids() -> set[str]:
    return {i.id for i in items()}


# ---- speakers -----------------------------------------------------------------------

def speakers() -> list[Speaker]:
    return [Speaker(speakerId=r["speaker_id"], ageGroup=r["age_group"], native=r["native"] == "true",
                    created=r["created"]) for r in _read(_root() / SPEAKERS)]


def new_speaker(age_group: str, native: bool) -> Speaker:
    """Next anonymous id: S01, S02, ... (consent was checked by the caller)."""
    rows = _read(_root() / SPEAKERS)
    speaker_id = f"S{len(rows) + 1:02d}"
    created = time.strftime("%Y-%m-%d %H:%M:%S")
    rows.append({"speaker_id": speaker_id, "created": created, "consent": "true",
                 "age_group": age_group, "native": "true" if native else "false"})
    _write(_root() / SPEAKERS, SPEAKER_FIELDS, rows)
    return Speaker(speakerId=speaker_id, ageGroup=age_group, native=native, created=created)


def speaker_exists(speaker_id: str) -> bool:
    return any(s.speakerId == speaker_id for s in speakers())


# ---- recordings ---------------------------------------------------------------------

def _upsert(speaker_id: str, item_id: str, file: str, status: str, nbytes: int) -> None:
    path = _root() / METADATA
    rows = [r for r in _read(path) if not (r["speaker_id"] == speaker_id and r["item_id"] == item_id)]
    rows.append({"speaker_id": speaker_id, "item_id": item_id, "file": file, "status": status,
                 "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "bytes": nbytes})
    rows.sort(key=lambda r: (r["speaker_id"], r["item_id"]))
    _write(path, META_FIELDS, rows)


def save_recording(speaker_id: str, item_id: str, audio_bytes: bytes) -> dict:
    """Write the clip (a re-record overwrites the previous one) and log it."""
    folder = _root() / speaker_id
    folder.mkdir(parents=True, exist_ok=True)
    file = f"{speaker_id}/{item_id}.webm"
    (_root() / file).write_bytes(audio_bytes)
    _upsert(speaker_id, item_id, file, "recorded", len(audio_bytes))
    return {"itemId": item_id, "file": file, "bytes": len(audio_bytes)}


def skip(speaker_id: str, item_id: str) -> None:
    _upsert(speaker_id, item_id, "", "skipped", 0)


def progress(speaker_id: str) -> dict[str, str]:
    """item_id -> recorded | skipped for one speaker."""
    return {r["item_id"]: r["status"] for r in _read(_root() / METADATA) if r["speaker_id"] == speaker_id}


# ---- export ---------------------------------------------------------------------------

def labels_template() -> str:
    """CSV for eval/run_eval.py: every recording, label blank for the qari.
    audio_path is relative to data/recordings/, where the file is saved too."""
    rows = [r for r in _read(_root() / METADATA) if r["status"] == "recorded"]
    lines = ["speaker_id,item_id,audio_path,label,error_type"]
    lines += [f'{r["speaker_id"]},{r["item_id"]},{r["file"]},,' for r in rows]
    text = "\n".join(lines) + "\n"
    (_root() / TEMPLATE).write_text(text, encoding="utf-8")
    return text
