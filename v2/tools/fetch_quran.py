"""Download reference text for a set of surahs into data/quran.json.

Two Tanzil editions are fetched from api.alquran.cloud:
  * quran-simple        - with tashkeel, used for display
  * quran-simple-clean  - without tashkeel, used for matching

Usage:  python tools/fetch_quran.py [surah numbers...]
"""

import json
import sys
import urllib.request
from pathlib import Path

DEFAULT_SURAHS = [1, 97, 103, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
API = "https://api.alquran.cloud/v1/surah/{n}/editions/quran-simple,quran-simple-clean"
OUT = Path(__file__).resolve().parent.parent / "data" / "quran.json"
BISMILLAH_WORDS = 4


def fetch(n: int) -> dict:
    with urllib.request.urlopen(API.format(n=n), timeout=30) as resp:
        data = json.load(resp)["data"]
    display, clean = data[0], data[1]
    assert display["edition"]["identifier"] == "quran-simple"
    ayahs = []
    for d, c in zip(display["ayahs"], clean["ayahs"]):
        dw = d["text"].replace("﻿", "").split()
        cw = c["text"].replace("﻿", "").split()
        # The API prepends the basmala to ayah 1 of every surah except 1 and 9.
        if n not in (1, 9) and d["numberInSurah"] == 1:
            dw, cw = dw[BISMILLAH_WORDS:], cw[BISMILLAH_WORDS:]
        if len(dw) != len(cw):
            raise ValueError(f"word count mismatch in {n}:{d['numberInSurah']}")
        ayahs.append({"number": d["numberInSurah"], "display": dw, "clean": cw})
    return {
        "number": n,
        "name": display["name"],
        "english_name": display["englishName"],
        "ayahs": ayahs,
    }


def main() -> None:
    surahs = [int(a) for a in sys.argv[1:]] or DEFAULT_SURAHS
    result = [fetch(n) for n in surahs]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"surahs": result}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(result)} surahs to {OUT}")


if __name__ == "__main__":
    main()
