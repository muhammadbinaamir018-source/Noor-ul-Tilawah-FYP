"""Content validation: the page JSON in content/qaida/ is consistent with the
audio on disk and with itself, and the lesson-grid counters reflect it."""

import hashlib

import pytest

from app.config import settings
from app.services.content_store import store

store.load()
MAPPED = [l for l in store.manifest.lessons if (settings.qaida_dir / l.file).exists()]

# Byte-identical copies of a clip that already has a box; see qaida_spec.py.
SKIPPED_DUPLICATES = {"a20006.mp3"}


@pytest.fixture(scope="module")
def pages():
    return {l.id: store.load_page(l.id) for l in MAPPED}


def test_every_scan_is_mapped(pages):
    assert set(pages) == {f"p{n:02d}" for n in range(1, 40)}


def test_every_region_audio_exists(pages):
    missing = [
        r.audio for p in pages.values() for r in p.regions if not (settings.audio_dir / r.audio).exists()
    ]
    assert missing == []


def test_region_ids_unique_and_prefixed(pages):
    for pid, p in pages.items():
        ids = [r.id for r in p.regions]
        assert len(ids) == len(set(ids)), pid
        assert all(i.startswith(f"{pid}-r") for i in ids), pid


def test_rects_inside_image(pages):
    for p in pages.values():
        for r in p.regions:
            assert r.rect.x + r.rect.w <= 1.0001 and r.rect.y + r.rect.h <= 1.0001, r.id


def test_clip_used_on_one_page_only(pages):
    seen = {}
    for p in pages.values():
        for r in p.regions:
            assert r.audio not in seen, f"{r.audio} on {seen.get(r.audio)} and {p.id}"
            seen[r.audio] = p.id


def test_every_clip_has_a_box(pages):
    """All 1398 clips are placed, except declared duplicates, which must be
    byte-identical to a clip that is placed."""
    used = {r.audio for p in pages.values() for r in p.regions}
    all_clips = {f.name for f in settings.audio_dir.glob("a*.mp3")}
    assert all_clips - used == SKIPPED_DUPLICATES
    digests = {hashlib.md5((settings.audio_dir / c).read_bytes()).hexdigest() for c in used}
    for c in SKIPPED_DUPLICATES:
        assert hashlib.md5((settings.audio_dir / c).read_bytes()).hexdigest() in digests, c


def test_rows_are_reading_order(pages):
    """Regions run right to left within a row, rows top to bottom (the
    practice flow and the editor's numbering rely on this order)."""
    for p in pages.values():
        for a, b in zip(p.regions, p.regions[1:]):
            assert b.row >= a.row, b.id
            if a.row == b.row:
                assert b.rect.x < a.rect.x + a.rect.w, f"{b.id} is not left of {a.id}"


def test_expected_text_is_arabic_and_mode_valid(pages):
    for p in pages.values():
        for r in p.regions:
            if r.expected:
                assert r.expected.text.strip(), r.id
                assert all(ord(c) > 0x600 or c == " " for c in r.expected.text), r.id


def test_scope_pages_are_recitable(pages):
    """Every FYP-scope page (1-12) has recitable items on it."""
    for n in range(1, 13):
        assert any(r.expected for r in pages[f"p{n:02d}"].regions), n


def test_lesson_grid_counts_split_lesson_between_pages():
    """A takhti's clips are shared between its scans, so each card counts only
    the clips not already placed on another page, and a fully mapped page
    shows assigned == total."""
    status = {s.id: s for s in store.status()}
    takhti1 = [s for s in status.values() if s.audioPage == "01"]
    assert len(takhti1) >= 2
    assert sum(s.totalClips for s in takhti1) == len(store.clips_for_page("01"))
    for s in takhti1:
        assert s.assignedClips == s.totalClips == s.regions, s.id


def test_lesson_grid_counts_page_that_starts_a_second_takhti():
    """Scan 27 finishes takhti 15 and starts takhti 16: its card counts both."""
    s = next(s for s in store.status() if s.id == "p27")
    assert s.totalClips == len(store.clips_for_page("15")) + len(store.clips_for_page("16"))
    assert s.assignedClips == s.totalClips == s.regions
