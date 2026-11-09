from typing import Literal, Optional

from pydantic import BaseModel, Field

ExpectedMode = Literal["letter-name", "syllable", "word", "phrase"]


class Rect(BaseModel):
    """Fractional rectangle (0-1 of image dimensions)."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)


class Expected(BaseModel):
    text: str
    variants: list[str] = []
    mode: ExpectedMode = "word"


class Region(BaseModel):
    id: str
    rect: Rect
    audio: str
    row: int = 1
    expected: Optional[Expected] = None


class PageContent(BaseModel):
    id: str
    image: str
    audioPage: Optional[str] = None
    regions: list[Region] = []


class Lesson(BaseModel):
    id: str
    number: int
    title: str
    image: str
    audioPage: Optional[str] = None
    file: str


class Manifest(BaseModel):
    version: int = 1
    lessons: list[Lesson]


class PageStatus(BaseModel):
    id: str
    title: str
    audioPage: Optional[str]
    totalClips: int
    assignedClips: int
    regions: int
    regionsWithExpected: int


class CollectItem(BaseModel):
    """One thing to record on /collect: where it is on the scan and what it says."""

    id: str
    pageId: str
    image: str
    rect: Rect
    text: str
    mode: ExpectedMode
    audio: str  # reference clip


class NewSpeaker(BaseModel):
    consent: bool
    ageGroup: Literal["child", "teen", "adult"]
    native: bool


class Speaker(BaseModel):
    speakerId: str
    ageGroup: str
    native: bool
    created: str


class SkipRequest(BaseModel):
    speakerId: str
    itemId: str


class MaddInfo(BaseModel):
    """Vowel-length check from tajweed.madd, present when TILAWAH_MADD_ENABLED
    and the item has a long vowel."""

    measured_s: float
    reference_s: float
    reference_is_default: bool
    counts: float
    expected_counts: int
    verdict: Literal["too-short", "2-count", "4-count", "too-long", "unclear"]
    ok: bool
    confidence: float
    hint: str
    clipped: bool


class Feedback(BaseModel):
    """What to tell the learner after a check (services/feedback.py)."""

    message: str
    heard: str
    expected: str
    expectedLetter: Optional[str] = None
    heardLetter: Optional[str] = None
    lowConfidence: bool = False
    maddHint: Optional[str] = None


class EvaluateResult(BaseModel):
    regionId: str
    verdict: Literal["correct", "incorrect", "unclear"]
    transcription: str
    normalizedTranscription: str
    expectedText: Optional[str] = None
    bestVariant: Optional[str] = None
    score: float
    threshold: float
    latencyMs: int
    madd: Optional[MaddInfo] = None
    # vowel-mark agreement 0-100 when TILAWAH_HARAKAT_SIGNAL_ENABLED; None if not comparable
    harakatScore: Optional[float] = None
    # present when the verdict is not correct, or the madd check failed
    feedback: Optional[Feedback] = None
