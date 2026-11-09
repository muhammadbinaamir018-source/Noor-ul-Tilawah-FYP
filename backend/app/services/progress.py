"""Per-learner progress, kept as one JSON file per learner (no database).

How it works: the browser sends a learner id (any short name, kept in
localStorage) with each recitation check. After the verdict the router calls
`apply_attempt`, which updates two tallies in the learner's state: per item
(attempts, fails, passed once a correct verdict was earned, last score /
verdict / time) and per letter (attempts, fails), where the letter is the one
the feedback blamed or, failing that, the item's own letter. `summarise`
turns the state into what the home page shows: counts, the weak letters
(most fails first) and a "revise these" list of items tried but never passed,
hardest first. State lives in data/progress/<learner>.json; the pure
functions never touch the disk, so they are tested without it.
"""

import json
import re
import time

from ..config import settings
from ..schemas import Expected

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_NOW = "%Y-%m-%d %H:%M:%S"


def check_id(learner_id: str | None) -> str:
    """Learner ids become file names: letters, digits, - and _ only."""
    if not learner_id or not _SAFE_ID.match(learner_id):
        raise ValueError("learner id may only contain letters, digits, - and _ (max 40)")
    return learner_id


def empty() -> dict:
    return {"version": 1, "items": {}, "letters": {}}


def apply_attempt(state: dict, region_id: str, verdict: str, score: float, letter: str | None,
                  when: str | None = None) -> dict:
    """A new state with one more attempt recorded; the input is left as it was."""
    new = {"version": 1,
           "items": {k: dict(v) for k, v in state["items"].items()},
           "letters": {k: dict(v) for k, v in state["letters"].items()}}
    ok = verdict == "correct"
    item = new["items"].setdefault(region_id, {"attempts": 0, "fails": 0, "passed": False,
                                               "lastVerdict": None, "lastScore": None, "lastTime": None})
    item["attempts"] += 1
    item["fails"] += 0 if ok else 1
    item["passed"] = item["passed"] or ok
    item["lastVerdict"], item["lastScore"] = verdict, score
    item["lastTime"] = when or time.strftime(_NOW)
    if letter:
        stats = new["letters"].setdefault(letter, {"attempts": 0, "fails": 0})
        stats["attempts"] += 1
        stats["fails"] += 0 if ok else 1
    return new


def summarise(learner_id: str, state: dict, items_index: dict, limit: int = 10) -> dict:
    """items_index: region id -> {pageId, text, letter} for the display."""
    items = state["items"]
    weak = [{"letter": letter, **stats} for letter, stats in state["letters"].items() if stats["fails"] > 0]
    weak.sort(key=lambda w: (-w["fails"], -w["attempts"], w["letter"]))
    revise = []
    for region_id, v in items.items():
        if v["passed"] or v["fails"] == 0:
            continue
        info = items_index.get(region_id, {})
        revise.append({"regionId": region_id, "pageId": info.get("pageId", region_id.split("-")[0]),
                       "text": info.get("text", ""), "fails": v["fails"], "lastScore": v["lastScore"]})
    revise.sort(key=lambda r: (-r["fails"], r["regionId"]))
    return {
        "learnerId": learner_id,
        "itemsAttempted": len(items),
        "itemsPassed": sum(1 for v in items.values() if v["passed"]),
        "attempts": sum(v["attempts"] for v in items.values()),
        "weakLetters": weak[:limit],
        "revise": revise[:limit],
    }


# ---- what an item's letter is ---------------------------------------------------------------

def item_letter(expected: Expected) -> str | None:
    """The letter an item is about: the bare letter of a letter-name item, else
    the first letter of its text."""
    from .matching import normalize

    if expected.mode == "letter-name":
        single = next((v for v in expected.variants if len(v) == 1), None)
        if single:
            return single
    text = normalize(expected.text).replace(" ", "")
    return text[0] if text else None


_index: dict | None = None


def items_index() -> dict:
    """region id -> {pageId, text, letter} for every recitable item, built once."""
    global _index
    if _index is None:
        from .content_store import store

        _index = {}
        for lesson in store.manifest.lessons:
            for r in store.load_page(lesson.id).regions:
                if r.expected:
                    _index[r.id] = {"pageId": lesson.id, "text": r.expected.text, "letter": item_letter(r.expected)}
    return _index


# ---- files -----------------------------------------------------------------------------------

def _path(learner_id: str):
    return settings.progress_dir / f"{check_id(learner_id)}.json"


def load(learner_id: str) -> dict:
    path = _path(learner_id)
    if not path.exists():
        return empty()
    return json.loads(path.read_text(encoding="utf-8"))


def save(learner_id: str, state: dict) -> None:
    path = _path(learner_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def reset(learner_id: str) -> None:
    path = _path(learner_id)
    if path.exists():
        path.unlink()
