"""One-time import of Noorani Qaida assets from the extracted Android app.

Copies word-level MP3s and scanned page JPGs from the extracted
com.creativeapps.nooraniqaidawithaudio package in ~/Downloads into content/.
Academic/private use only — see README licensing note.
"""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = (
    Path.home() / "Downloads" / "com.creativeapps.nooraniqaidawithaudio_1.15"
)


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    if not src.is_dir():
        sys.exit(f"Source not found: {src}")

    audio_dst = PROJECT_ROOT / "content" / "audio" / "qaida"
    pages_dst = PROJECT_ROOT / "content" / "pages"
    audio_dst.mkdir(parents=True, exist_ok=True)
    pages_dst.mkdir(parents=True, exist_ok=True)

    clips = sorted((src / "res" / "raw").glob("a*.mp3"))
    for f in clips:
        shutil.copy2(f, audio_dst / f.name)
    print(f"Copied {len(clips)} audio clips -> {audio_dst}")

    images = sorted((src / "res" / "drawable").glob("_*.jpg"))
    for f in images:
        shutil.copy2(f, pages_dst / f.name)
    print(f"Copied {len(images)} page images -> {pages_dst}")


if __name__ == "__main__":
    main()
