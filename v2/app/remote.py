"""Clients for app.model_service. Standard library only: the app server needs no torch."""

import json
import logging
import time
import urllib.error
import urllib.request

import numpy as np

from .pronunciation import CtcScorer

log = logging.getLogger(__name__)


class ModelServiceClient:
    def __init__(self, url: str, token: str | None = None, timeout: float = 60.0, retries: int = 2):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.retries = retries

    def _request(self, path: str, body: bytes | None = None) -> tuple[bytes, dict]:
        headers = {"Content-Type": "application/octet-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.url + path, data=body, headers=headers,
                                     method="POST" if body is not None else "GET")
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read(), dict(resp.headers)
            except urllib.error.HTTPError as e:
                if e.code < 500 or attempt == self.retries:
                    raise RuntimeError(f"model service {path}: HTTP {e.code} {e.read()[:200]!r}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.retries:
                    raise RuntimeError(f"model service {path} unreachable at {self.url}: {e}") from e
            time.sleep(0.5 * (attempt + 1))
        raise AssertionError("unreachable")

    def info(self) -> dict:
        return json.loads(self._request("/info")[0])

    @staticmethod
    def _pcm16(audio: np.ndarray) -> bytes:
        return (np.clip(audio, -1.0, 32767 / 32768) * 32768).astype("<i2").tobytes()

    def transcribe(self, audio: np.ndarray) -> str:
        body, _ = self._request("/asr", self._pcm16(audio))
        return json.loads(body)["text"]

    def log_probs(self, audio: np.ndarray) -> np.ndarray:
        body, headers = self._request("/logprobs", self._pcm16(audio))
        shape = next(v for k, v in headers.items() if k.lower() == "x-shape")
        t, v = (int(x) for x in shape.split(","))
        return np.frombuffer(body, dtype="<f4").reshape(t, v)


class RemoteASR:
    def __init__(self, client: ModelServiceClient):
        self.client = client

    def transcribe(self, audio: np.ndarray) -> str:
        return self.client.transcribe(audio)


class RemotePronunciation(CtcScorer):
    """Letter scoring and verification run here; only the acoustic model runs remotely."""

    def __init__(self, client: ModelServiceClient, vocab: dict[str, int], blank: int, threshold: float = 0.35):
        super().__init__(vocab, blank, threshold)
        self.client = client

    def log_probs(self, audio: np.ndarray) -> np.ndarray:
        return self.client.log_probs(audio)
