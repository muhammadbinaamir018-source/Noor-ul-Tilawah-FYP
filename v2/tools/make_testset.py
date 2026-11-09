"""Build a labelled test set with controlled mistakes from per-ayah recordings.

Downloads per-ayah MP3s from everyayah.com for the given reciters and surahs,
then writes, per (reciter, surah):
  * correct     - all ayahs in order                     (no errors)
  * skip        - one ayah left out                      (its words: missing)
  * repeat      - one ayah recited twice                 (second copy: extra words)
Labels go to <out>/manifest.jsonl in the format tools/evaluate.py reads.

Word substitutions and letter-level mistakes need human recordings; add them
to the manifest by hand with the same fields.

Usage: python tools/make_testset.py --surahs 1 112 113 --out testset
"""

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import audio, quran  # noqa: E402

URL = "https://everyayah.com/data/{reciter}/{surah:03d}{ayah:03d}.mp3"
DEFAULT_RECITERS = ["Alafasy_128kbps", "Husary_128kbps", "Minshawy_Murattal_128kbps"]
GAP = np.zeros(int(0.8 * 16000), np.float32)


def fetch_ayah(cache: Path, reciter: str, surah: int, ayah: int) -> np.ndarray:
    path = cache / reciter / f"{surah:03d}{ayah:03d}.mp3"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL.format(reciter=reciter, surah=surah, ayah=ayah), timeout=60) as r:
            path.write_bytes(r.read())
    return audio.load(path)


def join(clips: list[np.ndarray]) -> np.ndarray:
    parts = []
    for c in clips:
        parts += [c, GAP]
    return np.concatenate(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surahs", type=int, nargs="+", default=[1, 112, 113, 114])
    ap.add_argument("--reciters", nargs="+", default=DEFAULT_RECITERS)
    ap.add_argument("--out", default="testset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for reciter in args.reciters:
        for s in args.surahs:
            ref = quran.reference(s)
            n_ayahs = ref[-1].ayah
            clips = [fetch_ayah(out / "cache", reciter, s, a) for a in range(1, n_ayahs + 1)]
            words_of = {a: [w.index for w in ref if w.ayah == a] for a in range(1, n_ayahs + 1)}
            base = f"{reciter}_{s:03d}"

            def write(name, clips_, errors, recited):
                path = out / "audio" / f"{base}_{name}.wav"
                audio.save(path, join(clips_))
                rows.append({"audio": str(path.relative_to(out)), "surah": s, "from": 1, "to": n_ayahs,
                             "variant": name, "reciter": reciter, "recited": recited, "errors": errors})

            text = {a: " ".join(w.norm for w in ref if w.ayah == a) for a in range(1, n_ayahs + 1)}
            write("correct", clips, [], " ".join(text.values()))

            if n_ayahs >= 3:
                skip = rng.randint(2, n_ayahs - 1)       # interior ayah: skip is detectable
                write(f"skip{skip}", [c for a, c in enumerate(clips, 1) if a != skip],
                      [{"ref_index": i, "type": "missing"} for i in words_of[skip]],
                      " ".join(t for a, t in text.items() if a != skip))

            rep = rng.randint(1, n_ayahs - 1)
            # Either copy of the repeated ayah may be the one aligned as "extra",
            # so its words can attach anywhere from just before the ayah to its end.
            # Repeated words across neighbouring ayahs (e.g. "الناس") widen it to the previous ayah.
            window = [words_of[max(rep - 1, 1)][0] - 1, words_of[rep][-1]]
            write(f"repeat{rep}", clips[:rep] + [clips[rep - 1]] + clips[rep:],
                  [{"ref_index": window[1], "type": "extra", "window": window, "heard": ref[i].norm}
                   for i in words_of[rep]],
                  " ".join([*list(text.values())[:rep], text[rep], *list(text.values())[rep:]]))
            print(f"{reciter} surah {s}: done")

    (out / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                        encoding="utf-8")
    print(f"wrote {len(rows)} items to {out / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
