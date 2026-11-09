"""Labels CSV for the evaluation.

How it works: one row per recording with columns
    speaker_id, item_id, audio_path, label, error_type
`item_id` is a region id from content/qaida/pages/page-NN.json (e.g.
p01-r004), pages p01–p12 only, and the region must have expected text.
`label` is "correct" or "wrong"; `error_type` is free text (madd, makhraj,
harakat, ...) and optional. Relative audio paths are resolved against the CSV's
folder. Any bad row stops the run with a LabelError naming the line.
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.content_store import store  # noqa: E402

SCOPE_PAGES = {f"p{n:02d}" for n in range(1, 13)}
LABELS = {"correct", "wrong"}


class LabelError(ValueError):
    pass


@dataclass(frozen=True)
class Row:
    speaker_id: str
    item_id: str
    audio_path: str
    label: str
    error_type: str = ""


def scope_regions() -> dict:
    """Recitable regions of pages p01–p12, keyed by region id."""
    if not store.manifest.lessons:
        store.load()
    out = {}
    for page_id in sorted(SCOPE_PAGES):
        for r in store.load_page(page_id).regions:
            if r.expected:
                out[r.id] = r
    return out


def read_labels(path: Path) -> list[Row]:
    path = Path(path)
    regions = scope_regions()
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"speaker_id", "item_id", "audio_path", "label"} - set(reader.fieldnames or [])
        if missing:
            raise LabelError(f"{path}: missing columns {sorted(missing)}")
        for n, rec in enumerate(reader, start=2):
            item = rec["item_id"].strip()
            label = rec["label"].strip().lower()
            if item.split("-")[0] not in SCOPE_PAGES:
                raise LabelError(f"{path}:{n}: {item} is outside pages p01-p12")
            if item not in regions:
                raise LabelError(f"{path}:{n}: {item} is not a recitable region (no expected text, or no such id)")
            if label not in LABELS:
                raise LabelError(f"{path}:{n}: label must be correct|wrong, got {rec['label']!r}")
            audio = Path(rec["audio_path"].strip())
            if not audio.is_absolute():
                audio = path.parent / audio
            if not audio.exists():
                raise LabelError(f"{path}:{n}: audio not found: {audio}")
            rows.append(Row(rec["speaker_id"].strip(), item, str(audio), label, (rec.get("error_type") or "").strip()))
    if not rows:
        raise LabelError(f"{path}: no rows")
    return rows
