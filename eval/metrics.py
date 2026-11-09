"""Evaluation metrics: pure functions over scored rows.

How it works: every recording ends up as a `Scored` row holding the label the
human gave (correct / wrong) and the pipeline's rapidfuzz score. The verdict is
"correct" when the score reaches the threshold, "unclear" when Whisper heard
nothing, otherwise "incorrect". Then

    FRR  (False Reject Rate) = correct recitations the tutor rejected / all correct ones
    FAR  (False Accept Rate) = wrong recitations the tutor accepted / all wrong ones
    accuracy                 = verdicts that agree with the label / all rows

"unclear" counts as a reject. A sweep re-applies the verdict rule at many
thresholds to show the FAR/FRR trade-off; the scores are computed once, so a
sweep never re-runs ASR.

The optional harakat signal (app.services.harakat) is a second score kept on
each row. With `harakat_threshold` set, a row is "correct" only if the main
score passes AND the harakat score is at least the threshold; rows with no
harakat score (nothing to compare) are left to the main score alone. That is
how "variants only" and "variants + harakat signal" are compared on the same
transcripts.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

VERDICTS = ("correct", "incorrect", "unclear")
LABELS = ("correct", "wrong")


@dataclass(frozen=True)
class Scored:
    speaker_id: str
    item_id: str
    audio_path: str
    label: str            # human label: correct | wrong
    error_type: str
    transcription: str    # what Whisper heard
    score: float          # best rapidfuzz score against expected text + variants
    mode: str             # expected.mode of the item
    threshold: float      # the backend's threshold for that mode
    harakat_score: float | None = None   # vowel-mark agreement, None when not comparable


def verdict(row: Scored, threshold: float | None, harakat_threshold: float | None = None) -> str:
    """Same rule as app.services.matching.evaluate_match, with the threshold
    overridable so one set of scores serves a whole sweep."""
    if not row.transcription.strip():
        return "unclear"
    t = row.threshold if threshold is None else threshold
    if row.score < t:
        return "incorrect"
    if harakat_threshold is not None and row.harakat_score is not None and row.harakat_score < harakat_threshold:
        return "incorrect"
    return "correct"


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def compute_metrics(rows: list[Scored], threshold: float | None = None, harakat_threshold: float | None = None) -> dict:
    confusion: Counter = Counter()
    per_item: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})
    agree = 0
    for r in rows:
        v = verdict(r, threshold, harakat_threshold)
        confusion[(r.label, v)] += 1
        ok = (r.label == "correct") == (v == "correct")
        agree += ok
        per_item[r.item_id]["n"] += 1
        per_item[r.item_id]["correct"] += ok
    n_correct = sum(confusion[("correct", v)] for v in VERDICTS)
    n_wrong = sum(confusion[("wrong", v)] for v in VERDICTS)
    for item in per_item.values():
        item["accuracy"] = item["correct"] / item["n"]
    return {
        "n": len(rows),
        "n_correct_labels": n_correct,
        "n_wrong_labels": n_wrong,
        "accuracy": _rate(agree, len(rows)),
        "frr": _rate(n_correct - confusion[("correct", "correct")], n_correct),
        "far": _rate(confusion[("wrong", "correct")], n_wrong),
        "confusion": dict(confusion),
        "per_item": dict(per_item),
    }


def sweep(rows: list[Scored], thresholds: list[float], harakat_threshold: float | None = None) -> list[dict]:
    """FAR / FRR / accuracy at each threshold, for the ablation plot."""
    out = []
    for t in thresholds:
        m = compute_metrics(rows, t, harakat_threshold)
        out.append({"threshold": t, "far": m["far"], "frr": m["frr"], "accuracy": m["accuracy"]})
    return out
