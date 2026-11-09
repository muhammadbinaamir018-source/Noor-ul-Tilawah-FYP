import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import uvicorn

from app import quran
from app.model_service import create_app
from app.pronunciation import CtcScorer
from app.remote import ModelServiceClient, RemoteASR, RemotePronunciation

VOCAB = {"<pad>": 0, "|": 1, **{ch: 2 + k for k, ch in enumerate("قلهواحد")}}


class FakeASR:
    def transcribe(self, audio):
        return f"heard {len(audio)} samples"


class FakeAcoustic(CtcScorer):
    """Deterministic 'model': log-posteriors derived from the audio length."""

    def __init__(self):
        super().__init__(VOCAB, blank=0)

    def log_probs(self, audio):
        rng = np.random.default_rng(len(audio))
        return np.log(rng.dirichlet(np.ones(len(VOCAB)), size=len(audio) // 320)).astype(np.float32)


@pytest.fixture(scope="module")
def service():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(FakeASR(), FakeAcoustic(), token="s3cret", load=False)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(5)


def test_token_is_required(service):
    with pytest.raises(RuntimeError, match="401"):
        ModelServiceClient(service, token="wrong", retries=0).info()


def test_remote_models_match_local(service):
    client = ModelServiceClient(service, token="s3cret")
    info = client.info()
    assert info["vocab"] == VOCAB and info["blank"] == 0

    # int16 on the wire: compare against local results on the same quantised audio.
    audio = (np.random.default_rng(1).standard_normal(16000) * 0.1).astype(np.float32)
    audio = np.round(audio * 32768).astype(np.int16).astype(np.float32) / 32768
    assert RemoteASR(client).transcribe(audio) == "heard 16000 samples"

    local = FakeAcoustic()
    remote = RemotePronunciation(client, info["vocab"], info["blank"])
    assert np.array_equal(remote.log_probs(audio), local.log_probs(audio))

    # Letter scores and verification run client-side and agree with the local path.
    words = quran.reference(112, 1, 1)
    as_dict = lambda r: {k: [x.to_dict() for x in v] for k, v in r.items()}  # noqa: E731
    assert as_dict(remote.score(audio, words)) == as_dict(local.score(audio, words))
    logp = local.log_probs(audio)
    norms = [w.norm for w in words]
    assert remote.verify(logp, norms, 3, "احد") == pytest.approx(local.verify(logp, norms, 3, "احد"))


def test_bad_audio_is_rejected(service):
    client = ModelServiceClient(service, token="s3cret", retries=0)
    with pytest.raises(RuntimeError, match="400"):
        client._request("/asr", b"abc")   # odd byte count is not int16 PCM


def test_app_server_needs_no_torch():
    # With a model service, the app server side must import and score without torch.
    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['faster_whisper'] = None; "
        "sys.modules['transformers'] = None; "
        "import app.server, app.remote, app.models, app.session; "
        "from app.remote import RemotePronunciation, ModelServiceClient; "
        "RemotePronunciation(ModelServiceClient('http://x'), {'<pad>': 0, 'a': 1}, 0); print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_server_saves_sessions(tmp_path, monkeypatch):
    import json
    import wave

    from fastapi.testclient import TestClient

    from app import recorder, server
    from app.asr import ScriptedASR

    monkeypatch.setattr(recorder, "ROOT", tmp_path)
    monkeypatch.setattr(recorder, "ENABLED", True)
    monkeypatch.setattr(server.Models, "asr", ScriptedASR(["قل هو الله احد"]))
    monkeypatch.setattr(server.Models, "scorer", None)
    tone = (0.3 * np.sin(np.arange(24000) * 2 * np.pi * 220 / 16000) * 32767).astype("<i2")
    client = TestClient(server.app)                 # no lifespan: no real models are loaded
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "start", "surah": 112, "from": 1, "to": 1}))
        assert json.loads(ws.receive_text())["type"] == "ready"
        ws.send_bytes(tone.tobytes())
        ws.send_text(json.dumps({"type": "stop"}))
        while json.loads(ws.receive_text())["type"] != "final":
            pass
    [folder] = list(tmp_path.iterdir())
    with wave.open(str(folder / "audio.wav")) as w:
        assert w.getframerate() == 16000 and w.getnframes() == len(tone)
    info = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert info["surah"] == 112 and info["end_reason"] == "stopped" and info["final"]["type"] == "final"
