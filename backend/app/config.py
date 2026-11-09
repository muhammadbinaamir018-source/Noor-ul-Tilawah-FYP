from pathlib import Path

from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    content_dir: Path = PROJECT_ROOT / "content"
    # Learner recordings collected on /collect for the evaluation (git-ignored).
    recordings_dir: Path = PROJECT_ROOT / "data" / "recordings"
    # Per-learner progress JSON (git-ignored); no accounts, no database.
    progress_dir: Path = PROJECT_ROOT / "data" / "progress"
    # "base" | "tiny" | any HF repo id or local CTranslate2 model directory.
    # base: ~1.5s/check on this CPU, most accurate. tiny: ~3x faster, less accurate.
    # Switch with the TILAWAH_ASR_MODEL env var and restart the backend.
    asr_model: str = "base"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_cpu_threads: int = 0  # 0 = all logical processors
    asr_preload: bool = True
    # "local" runs Whisper on this machine; "remote" sends audio to a GPU
    # transcription server (cloud/tilawah_colab.ipynb) and only matches locally.
    asr_backend: str = "local"
    asr_remote_url: str = ""
    asr_remote_token: str = ""
    eval_log_enabled: bool = True
    # Madd (vowel length) check from the audio, attached to /api/evaluate
    # results for items with a long vowel. Off until tuned on learner data.
    madd_enabled: bool = False
    # Secondary vowel-mark score (services/harakat.py) in /api/evaluate results.
    # Never changes the verdict; off until compared on learner data.
    harakat_signal_enabled: bool = False
    # A failed check whose score is within this many points of the threshold
    # is reported as "try again" (low confidence) instead of a mistake.
    feedback_confidence_margin: float = 10.0

    # Verdict thresholds (rapidfuzz ratio 0-100) per expected.mode
    # Short letter names share characters easily (fuzz("الم","دال") = 66.7),
    # and correct answers reach ~100 via the per-item variants tables anyway,
    # so these stay strict.
    threshold_letter_name: float = 75.0
    threshold_syllable: float = 75.0
    threshold_word: float = 75.0
    threshold_phrase: float = 80.0

    model_config = {
        "env_prefix": "TILAWAH_",
        # Settings can also live in backend/.env — easier than PowerShell env
        # vars, which only apply to the window they were typed in.
        "env_file": str(PROJECT_ROOT / "backend" / ".env"),
        "env_file_encoding": "utf-8",
    }

    @property
    def qaida_dir(self) -> Path:
        return self.content_dir / "qaida"

    @property
    def pages_json_dir(self) -> Path:
        return self.content_dir / "qaida" / "pages"

    @property
    def images_dir(self) -> Path:
        return self.content_dir / "pages"

    @property
    def audio_dir(self) -> Path:
        return self.content_dir / "audio" / "qaida"

    @property
    def eval_log_dir(self) -> Path:
        return self.content_dir / "eval_log"

    @property
    def resolved_asr_model(self) -> str:
        aliases = {
            "base": "OdyAsh/faster-whisper-base-ar-quran",
            "tiny": str(PROJECT_ROOT / "backend" / "models" / "faster-whisper-tiny-ar-quran"),
        }
        return aliases.get(self.asr_model, self.asr_model)

    def threshold_for(self, mode: str) -> float:
        return {
            "letter-name": self.threshold_letter_name,
            "syllable": self.threshold_syllable,
            "word": self.threshold_word,
            "phrase": self.threshold_phrase,
        }.get(mode, self.threshold_word)


settings = Settings()
