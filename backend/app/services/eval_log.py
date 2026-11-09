"""Development log of every recitation evaluation: the recorded audio plus the
full text comparison, so mismatches can be diagnosed (noise? wrong variant?)
and the variants tables enriched. Also the raw data for the FYP evaluation
chapter."""

import json
import time
from typing import Any

from ..config import settings

_LOG_FILE = "log.jsonl"


def record(audio_bytes: bytes, entry: dict[str, Any]) -> None:
    if not settings.eval_log_enabled:
        return
    log_dir = settings.eval_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    audio_name = f"{stamp}-{entry['regionId']}-{int(time.time() * 1000) % 1000:03d}.webm"
    (log_dir / audio_name).write_bytes(audio_bytes)
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "audio": audio_name, **entry}
    with (log_dir / _LOG_FILE).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def entries(limit: int = 100) -> list[dict[str, Any]]:
    path = settings.eval_log_dir / _LOG_FILE
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out = [json.loads(line) for line in lines if line.strip()]
    return list(reversed(out))[:limit]


def clear() -> int:
    log_dir = settings.eval_log_dir
    if not log_dir.exists():
        return 0
    n = 0
    for f in log_dir.iterdir():
        f.unlink()
        n += 1
    return n
