"""Features for single-letter clips.

How it works:
- MFCC (numpy only): pre-emphasis, 25 ms Hamming frames every 10 ms, power
  spectrum, 26 triangular mel filters, log, DCT → 13 coefficients per frame.
  A clip becomes the mean and standard deviation of those over its voiced
  frames (frames above −30 dB of the loudest), 26 numbers, plus the clip's
  voiced length: 27 features.
- wav2vec2 (optional): mean-pooled last hidden state of a pretrained model
  from Hugging Face; needs `torch` and `transformers`, which are not in the
  backend requirements. `extract(..., kind="wav2vec2")` raises a clear error
  when they are missing.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from tajweed.madd import decode  # noqa: E402  PyAV decoding to mono 16 kHz

SR = 16000
N_MFCC, N_MELS, N_FFT = 13, 26, 512
FRAME, HOP = int(0.025 * SR), int(0.010 * SR)
WAV2VEC2_MODEL = "facebook/wav2vec2-base"


def _hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def _mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS) -> np.ndarray:
    edges = _mel_to_hz(np.linspace(_hz_to_mel(0), _hz_to_mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, mid):
            fb[m - 1, k] = (k - lo) / max(mid - lo, 1)
        for k in range(mid, hi):
            fb[m - 1, k] = (hi - k) / max(hi - mid, 1)
    return fb


def _dct_matrix(n_out, n_in) -> np.ndarray:
    n = np.arange(n_in)
    return np.cos(np.pi * np.outer(np.arange(n_out), (2 * n + 1)) / (2 * n_in))


_FB, _DCT = mel_filterbank(), _dct_matrix(N_MFCC, N_MELS)


def mfcc_frames(samples: np.ndarray) -> np.ndarray:
    """(n_frames, 13) MFCCs of a mono 16 kHz signal."""
    x = np.append(samples[0], samples[1:] - 0.97 * samples[:-1]).astype(np.float64)
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)))
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * np.hamming(FRAME)
    power = np.abs(np.fft.rfft(frames, N_FFT)) ** 2 / N_FFT
    logmel = np.log(power @ _FB.T + 1e-10)
    return logmel @ _DCT.T


def mfcc_features(samples: np.ndarray) -> np.ndarray:
    """27 numbers per clip: mean and std of the MFCCs over voiced frames, and voiced length (s)."""
    frames = mfcc_frames(samples)
    n = len(samples) // HOP
    rms = np.sqrt((samples[: n * HOP].reshape(n, HOP) ** 2).mean(axis=1)) + 1e-9
    voiced = (20 * np.log10(rms / rms.max()) > -30)[: len(frames)]
    sel = frames[voiced] if voiced.any() else frames
    return np.concatenate([sel.mean(axis=0), sel.std(axis=0), [voiced.sum() * 0.01]])


_w2v = None


def wav2vec2_features(samples: np.ndarray) -> np.ndarray:
    """Mean-pooled wav2vec2 hidden state (768 numbers). Optional dependency."""
    global _w2v
    try:
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("wav2vec2 features need `pip install torch transformers` (see ml/makharij/README.md)") from exc
    if _w2v is None:
        _w2v = (Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_MODEL), Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL).eval())
    extractor, model = _w2v
    with torch.no_grad():
        inputs = extractor(samples, sampling_rate=SR, return_tensors="pt")
        hidden = model(**inputs).last_hidden_state[0]
    return hidden.mean(dim=0).numpy()


def extract(paths: list[str], kind: str = "mfcc") -> np.ndarray:
    fn = {"mfcc": mfcc_features, "wav2vec2": wav2vec2_features}[kind]
    return np.stack([fn(decode(Path(p).read_bytes())) for p in paths])
