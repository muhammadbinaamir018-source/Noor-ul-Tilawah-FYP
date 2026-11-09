"""Reference text: loads data/quran.json and builds the word list for a selection."""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import arabic

DATA = Path(__file__).resolve().parent.parent / "data" / "quran.json"


@dataclass
class RefWord:
    index: int          # position in the selection
    surah: int
    ayah: int
    display: str        # with tashkeel
    norm: str           # normalised, for matching
    clusters: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "surah": self.surah,
            "ayah": self.ayah,
            "display": self.display,
            "clusters": self.clusters,
        }


@lru_cache(maxsize=1)
def _load() -> dict[int, dict]:
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    return {s["number"]: s for s in raw["surahs"]}


def surahs() -> list[dict]:
    return [
        {"number": s["number"], "name": s["name"], "english_name": s["english_name"],
         "ayahs": len(s["ayahs"])}
        for s in _load().values()
    ]


def reference(surah: int, ayah_from: int = 1, ayah_to: int | None = None) -> list[RefWord]:
    s = _load().get(surah)
    if s is None:
        raise KeyError(f"surah {surah} is not in {DATA.name}")
    ayah_to = ayah_to or len(s["ayahs"])
    if not 1 <= ayah_from <= ayah_to <= len(s["ayahs"]):
        raise ValueError(f"invalid ayah range {ayah_from}-{ayah_to} for surah {surah}")
    out: list[RefWord] = []
    for a in s["ayahs"][ayah_from - 1:ayah_to]:
        for disp, clean in zip(a["display"], a["clean"]):
            out.append(RefWord(len(out), surah, a["number"], disp, arabic.normalize(clean),
                               arabic.clusters(disp)))
    return out
