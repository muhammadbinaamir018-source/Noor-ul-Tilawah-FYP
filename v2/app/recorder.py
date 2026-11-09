"""Saves each recitation (audio + final result) under sessions/ for debugging and test data.

sessions/<YYYYmmdd-HHMMSS>_<surah>-<from>-<to>/
    audio.wav      what the server received (16 kHz mono int16)
    session.json   settings, final result, and a manifest row for tools/evaluate.py

Saved only on this machine; sessions/ is git-ignored. Disable with SAVE_SESSIONS=0.
To turn a session into a test item, label its mistakes in "manifest_row" (see README)
and append that row to testset/real/manifest.jsonl.
"""

import json
import logging
import os
import time
import wave
from pathlib import Path

log = logging.getLogger(__name__)

ENABLED = os.getenv("SAVE_SESSIONS", "1") != "0"
ROOT = Path(os.getenv("SESSIONS_DIR", Path(__file__).resolve().parent.parent / "sessions"))


class SessionRecorder:
    def __init__(self, surah: int, ayah_from: int, ayah_to: int, level: str, where: dict):
        self.meta = {"surah": surah, "from": ayah_from, "to": ayah_to, "level": level,
                     "started": time.strftime("%Y-%m-%d %H:%M:%S"), "models": where}
        self.pcm = bytearray()
        self.name = f"{time.strftime('%Y%m%d-%H%M%S')}_{surah:03d}-{ayah_from}-{ayah_to}"

    def add(self, chunk: bytes) -> None:
        self.pcm += chunk

    def save(self, final: dict | None, reason: str) -> Path | None:
        if not ENABLED or not self.pcm:
            return None
        try:
            folder = ROOT / self.name
            folder.mkdir(parents=True, exist_ok=True)
            with wave.open(str(folder / "audio.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(bytes(self.pcm))
            info = {**self.meta, "ended": time.strftime("%Y-%m-%d %H:%M:%S"), "end_reason": reason,
                    "seconds": round(len(self.pcm) / 32000, 2), "final": final,
                    "manifest_row": {"audio": str((folder / "audio.wav").resolve()), "surah": self.meta["surah"],
                                     "from": self.meta["from"], "to": self.meta["to"],
                                     "level": self.meta["level"],
                                     "recited": "TODO: what was actually recited (normalised words)",
                                     "errors": "TODO: labelled mistakes, [] if the recitation was correct"}}
            (folder / "session.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
            log.info("saved session to %s", folder)
            return folder
        except OSError:
            log.exception("could not save session")
            return None
