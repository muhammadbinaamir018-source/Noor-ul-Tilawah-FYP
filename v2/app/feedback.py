"""Turns alignment and pronunciation results into learner-facing messages."""

from .alignment import CLOSE, MISSING, WRONG, Alignment
from .pronunciation import MAKHRAJ_HINTS
from .quran import RefWord


def _where(w: RefWord) -> str:
    return f"ayah {w.ayah}, word «{w.display}»"


def messages(ref: list[RefWord], al: Alignment, pron: dict[int, list[dict]], final: bool) -> list[dict]:
    out: list[dict] = []
    for i, w in enumerate(al.words):
        r = ref[i]
        if w.status == WRONG:
            out.append({"ref_index": i, "kind": "word", "severity": "error",
                        "text": f"{_where(r)}: heard «{w.heard}» instead."})
        elif w.status == CLOSE:
            out.append({"ref_index": i, "kind": "word", "severity": "warning",
                        "text": f"{_where(r)}: close, but heard «{w.heard}». Check the letters and vowels."})
        elif w.status == MISSING and (i == 0 or al.words[i - 1].status != MISSING):
            # One message per run of skipped words.
            k = i
            while k + 1 < len(al.words) and al.words[k + 1].status == MISSING:
                k += 1
            if k == i:
                text = f"{_where(r)}: this word was skipped."
            else:
                span = f"ayah {r.ayah}" if ref[k].ayah == r.ayah else f"ayahs {r.ayah}-{ref[k].ayah}"
                words_ = " ".join(x.display for x in ref[i:k + 1])
                text = f"Skipped {k - i + 1} words in {span}: «{words_}»."
            out.append({"ref_index": i, "kind": "word", "severity": "error", "text": text})
        for letter in pron.get(i, []):
            if letter["flagged"]:
                hint = MAKHRAJ_HINTS.get(letter["letter"], "")
                out.append({"ref_index": i, "kind": "letter", "severity": "warning",
                            "cluster": letter["cluster"],
                            "text": f"{_where(r)}: «{letter['letter']}» sounded unclear "
                                    f"(score {letter['score']:.2f}). Pronounce it {hint}."})
    # Consecutive extra words at one spot (e.g. a repeated phrase) become one message.
    runs: list[list[dict]] = []
    for e in al.extras:
        if runs and runs[-1][-1]["after"] == e["after"] and runs[-1][-1]["hyp_index"] + 1 == e["hyp_index"]:
            runs[-1].append(e)
        else:
            runs.append([e])
    for run in runs:
        pos = run[0]["after"]
        where = f"after «{ref[pos].display}»" if pos >= 0 else "at the start"
        heard = " ".join(e["heard"] for e in run)
        noun = "word" if len(run) == 1 else "words"
        out.append({"ref_index": max(pos, 0), "kind": "extra", "severity": "info",
                    "text": f"Extra {noun} «{heard}» {where}."})
    if final and al.position < len(ref):
        out.append({"ref_index": al.position, "kind": "incomplete", "severity": "info",
                    "text": f"Recitation stopped before {_where(ref[al.position])}."})
    return sorted(out, key=lambda m: m["ref_index"])
