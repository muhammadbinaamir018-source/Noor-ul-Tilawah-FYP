"""Generate hotspot regions for Qaida pages from qaida_spec.py.

Run from inside backend/:

    python scripts/map_qaida.py                  # dry run: counts per lesson
    python scripts/map_qaida.py --write          # write content/qaida/pages/*.json
    python scripts/map_qaida.py --check clips.jsonl
        # score every reference clip's transcript against its region's expected
        # text with app.services.matching; a correct reciter should pass

Boxes come from the ruled grid on each scan (qaida_grid.py); clips are taken
in reading order from the page's takhti, continuing where the previous scan
stopped. Page 1 was mapped by hand and is left untouched; its clips count as
used. Nothing is written unless every lesson's clip count matches exactly.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qaida_grid import INK, bands, is_blank, load_gray, split_cell, split_row  # noqa: E402
from qaida_spec import PAGES  # noqa: E402

from app.config import settings  # noqa: E402
from app.schemas import Expected, PageContent, Rect, Region  # noqa: E402
from app.services.content_store import store  # noqa: E402
from app.services.matching import evaluate_match  # noqa: E402

# Letter names as the reference reciter says them (Urdu/Qaida convention).
NAMES = {
    "ا": "الف", "أ": "الف", "ب": "با", "ت": "تا", "ة": "تا", "ث": "ثا", "ج": "جيم",
    "ح": "حا", "خ": "خا", "د": "دال", "ذ": "ذال", "ر": "را", "ز": "زا",
    "س": "سين", "ش": "شين", "ص": "صاد", "ض": "ضاد", "ط": "طا", "ظ": "ظا",
    "ع": "عين", "غ": "غين", "ف": "فا", "ق": "قاف", "ك": "كاف", "ل": "لام",
    "م": "ميم", "ن": "نون", "و": "واو", "ؤ": "واو", "ه": "ها", "ء": "همزه",
    "ي": "يا", "ى": "يا", "ئ": "يا", "ے": "يا",
}
# Characters outside matching.py's Arabic range, swapped for their Arabic twin.
DISPLAY_TO_ARABIC = str.maketrans({"ے": "ي", "ک": "ك", "ہ": "ه", "ی": "ي"})

Box = tuple[tuple[int, int, int, int], str | None, int]  # (x0, y0, x1, y1), text, row offset


def page1_variants() -> dict[str, list[str]]:
    """Variants harvested by hand on page 1, keyed by the bare letter."""
    out = {}
    for r in store.load_page("p01").regions:
        if r.expected and r.expected.mode == "letter-name":
            letter = next((v for v in r.expected.variants if len(v) == 1), None)
            if letter:
                out[letter] = [v for v in r.expected.variants if v != letter]
    return out


def expected_for(text, mode: str | None, harvested: dict) -> Expected | None:
    if text is None or text == "-" or mode is None:
        return None
    if isinstance(text, tuple):  # (printed text, [variants])
        text, variants = text
        return Expected(text=text.translate(DISPLAY_TO_ARABIC),
                        variants=[v.translate(DISPLAY_TO_ARABIC) for v in variants], mode=mode)
    arabic = text.translate(DISPLAY_TO_ARABIC)
    if mode == "letter-name":
        name = NAMES[text]
        variants = [arabic] + [v for v in harvested.get(arabic, []) if v != name]
        return Expected(text=name, variants=variants, mode="letter-name")
    if mode == "names":
        # the reference reciter spells the letters, then reads the joined form;
        # either is accepted from the learner
        return Expected(text=" ".join(NAMES[c] for c in text), variants=[arabic], mode="word")
    return Expected(text=arabic, variants=[], mode=mode)


def _filled_cells(ink, band: dict) -> list[tuple[int, int]]:
    """Cells with a glyph in them, right to left."""
    y0, y1 = band["y0"], band["y1"]
    return [c for c in band["cells"] if not is_blank(ink, c[0], c[1], y0, y1)][::-1]


def boxes(ink, found: list[dict], bi: int, spec: tuple) -> tuple[list[Box], int]:
    """Boxes for one spec entry, in reading order, and how many bands it used."""
    kind = spec[0]
    band = found[bi]
    y0, y1 = band["y0"], band["y1"]
    full = (band["cells"][0][0], band["cells"][-1][1])
    if kind == "none":
        return [], 1
    if kind == "one":
        text = spec[1] if len(spec) > 1 else None
        return [((full[0], y0, full[1], y1), text, 0)], 1
    if kind == "tall":  # one region across the next k bands
        k, text = spec[1], spec[2]
        return [((full[0], y0, full[1], found[bi + k - 1]["y1"]), text, 0)], k
    if kind == "vsplit":  # stacked items: cut at the widest blank pixel rows
        rows = split_cell(ink.T, full[0], full[1], y0, y1, spec[1])
        return [((full[0], a, full[1], b), None, 0) for a, b in rows], 1
    if kind == "cols":
        return [((a, y0, b, y1), t, 0) for a, b, t in spec[1]], 1
    if kind == "boxes":
        return [((x0, by0, x1, by1), t, r) for x0, by0, x1, by1, t, r in spec[1]], 1
    items = spec[1]
    if kind == "cells":
        cells = _filled_cells(ink, band)
        if len(cells) != len(items):
            raise ValueError(f"band y={y0}: {len(cells)} filled cells, spec has {len(items)}")
        return [((a, y0, b, y1), t, 0) for (a, b), t in zip(cells, items) if t != "-"], 1
    if kind == "cellsplit":  # each filled cell holds its own row of items
        cells = _filled_cells(ink, band)
        if len(cells) != len(items):
            raise ValueError(f"band y={y0}: {len(cells)} filled cells, spec has {len(items)}")
        out: list[Box] = []
        for (a, b), sub in zip(cells, items):
            if sub == ["-"]:
                continue
            spans = split_row(ink, y0, y1, a, b, len(sub))
            out += [((p, y0, q, y1), t, 0) for (p, q), t in zip(spans[::-1], sub)]
        return out, 1
    if kind == "cellvsplit":  # each cell: items stacked top to bottom, `lines` text lines
        cells = _filled_cells(ink, band)
        if len(cells) != len(items):
            raise ValueError(f"band y={y0}: {len(cells)} filled cells, spec has {len(items)}")
        lines = spec[3] if len(spec) > 3 else len(items[0])
        out = []
        for (a, b), sub in zip(cells, items):
            rows = split_cell(ink.T, a, b, y0, y1, lines)[: len(sub)]
            # the stacked items share the cell's row: clips run cell by cell
            out += [((a, p, b, q), t, 0) for (p, q), t in zip(rows, sub)]
        return out, 1
    if kind == "colsplit":  # explicit columns (right to left), each split into its items
        out = []
        for a, b, sub in items:
            spans = split_row(ink, y0, y1, a, b, len(sub))
            out += [((p, y0, q, y1), t, 0) for (p, q), t in zip(spans[::-1], sub)]
        return out, 1
    if kind in ("split", "splitx"):
        # circled single letters are equal width; words are cut at thin ink
        spans = split_row(ink, y0, y1, full[0], full[1], len(items),
                          trim=kind == "splitx", equal=spec[2] == "letter-name")
        return [((a, y0, b, y1), t, 0) for (a, b), t in zip(spans[::-1], items)], 1
    raise ValueError(f"unknown band kind {kind}")


def build() -> dict[str, PageContent]:
    harvested = page1_variants()
    used = defaultdict(int)
    used["01"] = len(store.load_page("p01").regions)
    skipped: set[str] = set()
    pages = {}
    for page_id, spec in PAGES.items():
        lesson = store.lesson(page_id)
        ink = load_gray(settings.images_dir / lesson.image) < INK
        h, w = ink.shape
        found = bands(ink, rows=spec.get("rows"))
        skip = set(spec.get("skip_clips", []))
        skipped |= skip
        n_bands = sum(b[0] != "lesson" and (b[1] if b[0] == "tall" else 1) for b in spec["bands"])
        if len(found) != n_bands:
            raise ValueError(f"{page_id}: grid has {len(found)} bands, spec covers {n_bands}")
        audio_page = spec["lesson"]
        clips = [c for c in store.clips_for_page(audio_page) if c not in skip]
        regions = []
        bi = row = 0
        for bspec in spec["bands"]:
            if bspec[0] == "lesson":  # the rest of the page continues another takhti
                audio_page = bspec[1]
                clips = [c for c in store.clips_for_page(audio_page) if c not in skip]
                continue
            row += 1
            mode_at = 3 if bspec[0] == "tall" else 2
            mode = bspec[mode_at] if len(bspec) > mode_at else None
            got, k = boxes(ink, found, bi, bspec)
            bi += k
            for (x0, y0, x1, y1), text, rowoff in got:
                idx = used[audio_page]
                used[audio_page] += 1
                regions.append(Region(
                    id=f"{page_id}-r{len(regions) + 1:03d}",
                    rect=Rect(x=round(x0 / w, 4), y=round(y0 / h, 4),
                              w=round((x1 - x0) / w, 4), h=round((y1 - y0) / h, 4)),
                    audio=clips[idx] if idx < len(clips) else "MISSING",
                    row=row + rowoff,
                    expected=expected_for(text, mode, harvested),
                ))
            row += max((r for _, _, r in got), default=0)
        pages[page_id] = PageContent(id=page_id, image=lesson.image, audioPage=spec["lesson"], regions=regions)

    ok = True
    for lesson_id in sorted(used):
        total = len([c for c in store.clips_for_page(lesson_id) if c not in skipped])
        status = "ok" if used[lesson_id] == total else "MISMATCH"
        ok &= status == "ok"
        print(f"lesson {lesson_id}: {used[lesson_id]} regions for {total} clips  {status}")
    if skipped:
        print(f"skipped exact-duplicate clips with no box of their own: {sorted(skipped)}")
    if not ok:
        raise SystemExit("clip counts do not match; nothing written")
    return pages


def check(pages: dict[str, PageContent], transcripts: Path) -> None:
    heard = {}
    for line in transcripts.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        heard[r["clip"]] = r["text"]
    # duplicate files were transcribed once; share the text by content hash
    by_hash = {}
    for clip in list(heard):
        by_hash[hashlib.md5((settings.audio_dir / clip).read_bytes()).hexdigest()] = heard[clip]

    def text_of(clip):
        if clip in heard:
            return heard[clip]
        return by_hash.get(hashlib.md5((settings.audio_dir / clip).read_bytes()).hexdigest())

    for page_id, page in pages.items():
        scored = passed = 0
        fails = []
        for r in page.regions:
            if not r.expected:
                continue
            t = text_of(r.audio)
            if t is None:
                continue
            scored += 1
            res = evaluate_match(t, r.expected)
            if res["verdict"] == "correct":
                passed += 1
            else:
                fails.append(f"    {r.id} {r.audio} expected={r.expected.text!r} heard={t[:60]!r} score={res['score']}")
        print(f"{page_id}: {passed}/{scored} reference clips pass")
        for f in fails:
            print(f)


def write(pages: dict[str, PageContent]) -> None:
    for page in pages.values():
        store.save_page(page)
        print(f"wrote {page.id}: {len(page.regions)} regions, "
              f"{sum(1 for r in page.regions if r.expected)} recitable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", type=Path)
    args = ap.parse_args()
    store.load()
    pages = build()
    if args.check:
        check(pages, args.check)
    if args.write:
        write(pages)


if __name__ == "__main__":
    main()
