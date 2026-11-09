"""Loads and persists Qaida content: manifest, page region JSON, audio clip inventory."""

import json
import logging
import re
from pathlib import Path

from ..config import settings
from ..schemas import Lesson, Manifest, PageContent, PageStatus, Region

logger = logging.getLogger(__name__)

_CLIP_RE = re.compile(r"^a(\d{2})(\d{3})(?:_(\d))?\.mp3$")


def _default_manifest() -> Manifest:
    """Identity mapping: image _N.jpg <-> audio page N for pages 1-27.

    Images 28-39 exist in the scanned set but have no obvious audio page;
    their audioPage stays null until assigned in the hotspot editor.
    """
    lessons = []
    images = sorted(
        settings.images_dir.glob("_*.jpg"),
        key=lambda p: int(p.stem.lstrip("_")),
    )
    audio_pages = {m.group(1) for f in settings.audio_dir.glob("a*.mp3") if (m := _CLIP_RE.match(f.name))}
    for img in images:
        n = int(img.stem.lstrip("_"))
        page = f"{n:02d}"
        lessons.append(
            Lesson(
                id=f"p{page}",
                number=n,
                title=f"Page {n}",
                image=img.name,
                audioPage=page if page in audio_pages else None,
                file=f"pages/page-{page}.json",
            )
        )
    return Manifest(lessons=lessons)


class ContentStore:
    def __init__(self) -> None:
        self.manifest: Manifest = Manifest(lessons=[])

    def load(self) -> None:
        manifest_path = settings.qaida_dir / "manifest.json"
        if manifest_path.exists():
            self.manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        else:
            self.manifest = _default_manifest()
            self.save_manifest()
            logger.info("Generated default manifest with %d lessons", len(self.manifest.lessons))
        missing = [
            r.audio
            for lesson in self.manifest.lessons
            for r in self.load_page(lesson.id).regions
            if not (settings.audio_dir / r.audio).exists()
        ]
        if missing:
            logger.warning("%d regions reference missing audio files: %s", len(missing), missing[:10])

    def save_manifest(self) -> None:
        path = settings.qaida_dir / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.manifest.model_dump_json(indent=2), encoding="utf-8")

    def lesson(self, page_id: str) -> Lesson | None:
        return next((l for l in self.manifest.lessons if l.id == page_id), None)

    def _page_path(self, lesson: Lesson) -> Path:
        return settings.qaida_dir / lesson.file

    def load_page(self, page_id: str) -> PageContent:
        lesson = self.lesson(page_id)
        if lesson is None:
            raise KeyError(page_id)
        path = self._page_path(lesson)
        if path.exists():
            return PageContent.model_validate_json(path.read_text(encoding="utf-8"))
        return PageContent(id=lesson.id, image=lesson.image, audioPage=lesson.audioPage)

    def save_page(self, page: PageContent) -> None:
        lesson = self.lesson(page.id)
        if lesson is None:
            raise KeyError(page.id)
        path = self._page_path(lesson)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
        if page.audioPage != lesson.audioPage:
            lesson.audioPage = page.audioPage
            self.save_manifest()

    def find_region(self, region_id: str) -> Region | None:
        page_id = region_id.split("-")[0]
        try:
            page = self.load_page(page_id)
        except KeyError:
            return None
        return next((r for r in page.regions if r.id == region_id), None)

    def clips_for_page(self, audio_page: str | None) -> list[str]:
        if not audio_page:
            return []
        return sorted(
            f.name
            for f in settings.audio_dir.glob(f"a{audio_page}*.mp3")
            if (m := _CLIP_RE.match(f.name)) and m.group(1) == audio_page
        )

    def status(self) -> list[PageStatus]:
        # One takhti's audio spans several scans, and a scan can finish one
        # takhti and start the next. A page owns its takhti's clips minus those
        # placed on other pages, plus any clip its own regions already use.
        pages = {lesson.id: self.load_page(lesson.id) for lesson in self.manifest.lessons}
        out = []
        for lesson in self.manifest.lessons:
            page = pages[lesson.id]
            elsewhere = {r.audio for other in pages.values() if other.id != page.id for r in other.regions}
            assigned = {r.audio for r in page.regions if (settings.audio_dir / r.audio).exists()}
            clips = [c for c in self.clips_for_page(page.audioPage) if c not in elsewhere]
            clips += sorted(assigned - set(clips))
            out.append(
                PageStatus(
                    id=lesson.id,
                    title=lesson.title,
                    audioPage=page.audioPage,
                    totalClips=len(clips),
                    assignedClips=len(assigned & set(clips)),
                    regions=len(page.regions),
                    regionsWithExpected=sum(1 for r in page.regions if r.expected),
                )
            )
        return out


store = ContentStore()
