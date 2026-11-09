"""Audio file loading for offline tools (the live path receives 16 kHz PCM directly)."""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .vad import SAMPLE_RATE


def _load_av(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode containers libsndfile cannot read (m4a/aac from phone recorders) with PyAV."""
    import av

    with av.open(str(path)) as c:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        chunks = [g.to_ndarray().reshape(-1) for f in c.decode(audio=0) for g in resampler.resample(f)]
        chunks += [g.to_ndarray().reshape(-1) for g in resampler.resample(None)]
    return np.concatenate(chunks).astype(np.float32), SAMPLE_RATE


def load(path: str | Path) -> np.ndarray:
    """Any audio file (wav, flac, ogg, mp3 via libsndfile; m4a/aac via PyAV) -> 16 kHz mono float32."""
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except sf.LibsndfileError:
        x, sr = _load_av(path)
    if sr != SAMPLE_RATE:
        g = gcd(sr, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return x


def save(path: str | Path, audio: np.ndarray) -> None:
    sf.write(str(path), audio, SAMPLE_RATE)
