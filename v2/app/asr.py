"""Speech recognition backends. All take 16 kHz mono float32 audio."""

import logging
import os
import time
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

# ASR and pronunciation models run concurrently on CPU; each gets half the cores.
CPU_SHARE = max(1, (os.cpu_count() or 2) // 2)

# Quran fine-tune of Whisper-base; the CTranslate2 repo is a conversion of the same weights.
DEFAULT_ASR_MODELS = {
    "faster-whisper": "OdyAsh/faster-whisper-base-ar-quran",
    "transformers": "tarteel-ai/whisper-base-ar-quran",
}


class ASR(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class WhisperASR:
    """Whisper fine-tuned on Quran recitation (default: tarteel-ai/whisper-base-ar-quran)."""

    def __init__(self, model_id: str, device: str = "cpu", num_beams: int = 1):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        t = time.perf_counter()
        self.torch = torch
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        self.num_beams = num_beams
        # Older fine-tunes ship generation configs without language tables;
        # pin Arabic transcription through explicit decoder prompt ids instead.
        self.prompt_ids = self.processor.get_decoder_prompt_ids(language="arabic", task="transcribe")
        gc = self.model.generation_config
        gc.forced_decoder_ids = None
        if hasattr(gc, "suppress_tokens") and gc.suppress_tokens is None:
            gc.suppress_tokens = []
        log.info("loaded ASR %s in %.1fs", model_id, time.perf_counter() - t)

    def transcribe(self, audio: np.ndarray) -> str:
        feats = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        prompt = [self.model.generation_config.decoder_start_token_id] + [t for _, t in self.prompt_ids]
        with self.torch.inference_mode():
            out = self.model.generate(
                feats.to(self.device),
                decoder_input_ids=self.torch.tensor([prompt], device=self.device),
                max_new_tokens=200,
                num_beams=self.num_beams,
            )
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()


class FasterWhisperASR:
    """Same model family converted to CTranslate2 (int8 on CPU): several times faster."""

    def __init__(self, model_id: str, device: str = "cpu", compute_type: str | None = None):
        from faster_whisper import WhisperModel

        # int8 is fastest on CPU; float16 is fast and exact on a GPU.
        compute_type = compute_type or ("float16" if device.startswith("cuda") else "int8")
        t = time.perf_counter()
        self.model = WhisperModel(model_id, device=device, compute_type=compute_type, cpu_threads=CPU_SHARE)
        log.info("loaded ASR %s (ctranslate2 %s) in %.1fs", model_id, compute_type, time.perf_counter() - t)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio, language="ar", task="transcribe", beam_size=1,
            vad_filter=False, condition_on_previous_text=False, without_timestamps=True,
        )
        kept = []
        for s in segments:
            if is_hallucination(s.no_speech_prob, s.avg_logprob, s.compression_ratio):
                log.info("dropped likely non-speech ASR segment %r (no_speech %.2f, logprob %.2f, ratio %.2f)",
                         s.text.strip(), s.no_speech_prob, s.avg_logprob, s.compression_ratio)
                continue
            kept.append(s.text.strip())
        return " ".join(kept)


def is_hallucination(no_speech_prob: float, avg_logprob: float, compression_ratio: float) -> bool:
    """Whisper's own thresholds for text it produced from non-speech: likely no speech and
    low confidence, or abnormally repetitive text."""
    return (no_speech_prob > 0.6 and avg_logprob < -1.0) or compression_ratio > 2.4


def load_asr(backend: str, model_id: str, device: str = "cpu") -> ASR:
    if backend == "faster-whisper":
        return FasterWhisperASR(model_id, device)
    if backend == "transformers":
        return WhisperASR(model_id, device)
    raise ValueError(f"unknown ASR backend {backend!r}")


class ScriptedASR:
    """Test double: returns queued transcripts in order (last one repeats)."""

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls = 0

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls += 1
        return self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
