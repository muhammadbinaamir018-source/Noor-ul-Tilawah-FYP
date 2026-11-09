"""Compare speech-recognition models on the same test sets.

Each model runs through tools/evaluate.py (same streaming session, VAD and alignment as the
app) with only the ASR changed. Transformers checkpoints are converted once to CTranslate2
for faster-whisper and cached under models/ (int8 on CPU, float16 on GPU).

Usage:
  python tools/compare_asr.py --sets testset/real/manifest.jsonl testset/noisy/manifest.jsonl \
      --models OdyAsh/faster-whisper-base-ar-quran deepdml/whisper-small-ar-quran-mix-norm
  add --pronunciation to include the letter model (verification of close/extra words);
  by default only the ASR runs, which isolates the ASR's own errors and is faster.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "models"


def ct2_path(model_id: str, device: str) -> str:
    """A faster-whisper-loadable local path for `model_id`, converting if needed.
    Works offline (HF_HUB_OFFLINE=1) for models already in the Hugging Face cache."""
    from huggingface_hub import snapshot_download

    local = Path(model_id) if Path(model_id).exists() else Path(snapshot_download(
        model_id, allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt"],
        ignore_patterns=["runs/*", "optimizer*", "training_args*"]))
    if (local / "model.bin").exists():
        return str(local)                                   # already CTranslate2
    quant = "float16" if device.startswith("cuda") else "int8"
    out = CACHE / f"{model_id.replace('/', '__')}-{quant}"
    if not (out / "model.bin").exists():
        from ctranslate2.converters import TransformersConverter

        print(f"converting {model_id} to CTranslate2 ({quant}) ...", flush=True)
        t = time.time()
        # Without tokenizer.json faster-whisper falls back to the standard Whisper tokenizer.
        extra = [f for f in ("tokenizer.json", "preprocessor_config.json") if (local / f).exists()]
        TransformersConverter(str(local), copy_files=extra or None, low_cpu_mem_usage=True
                              ).convert(str(out), quantization=quant, force=True)
        print(f"  done in {time.time() - t:.0f}s -> {out}", flush=True)
    return str(out)


SUMMARY = {
    "wer": r"(?m)^WER\s+([\d.]+)%",       # the summary line; per-item lines also contain "WER"
    "precision": r"mistake detection\s+P ([\d.]+)%",
    "recall": r"mistake detection\s+P [\d.]+%\s+R ([\d.]+)%",
    "decode_p50_ms": r"ASR decode\s+p50 (\d+) ms",
}


def run(model: str, manifest: str, device: str, pronunciation: bool) -> dict:
    env = dict(os.environ, ASR_BACKEND="faster-whisper", ASR_MODEL=model, DEVICE=device)
    env.pop("MODEL_SERVICE_URL", None)
    cmd = [sys.executable, str(ROOT / "tools" / "evaluate.py"), manifest]
    if not pronunciation:
        cmd.append("--no-pronunciation")
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    if out.returncode:
        return {"error": (out.stderr or out.stdout)[-600:]}
    res = {}
    for key, pat in SUMMARY.items():
        m = re.search(pat, out.stdout)
        res[key] = float(m.group(1)) if m else None
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--device", default=os.getenv("DEVICE", "cpu"))
    ap.add_argument("--pronunciation", action="store_true")
    ap.add_argument("--out", help="write results as JSON here")
    args = ap.parse_args()

    results = []
    for model in args.models:
        path = ct2_path(model, args.device)
        for manifest in args.sets:
            t = time.time()
            r = run(path, manifest, args.device, args.pronunciation)
            r.update(model=model, set=manifest, minutes=round((time.time() - t) / 60, 1))
            results.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)

    print(f"\n| model | test set | WER | precision | recall | decode p50 ({args.device}) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in results:
        if "error" in r:
            print(f"| {r['model']} | {r['set']} | error | | | |")
            continue
        fmt = lambda v, u="": "-" if v is None else f"{v:g}{u}"  # noqa: E731
        print(f"| {r['model']} | {r['set']} | {fmt(r['wer'], ' %')} | {fmt(r['precision'], ' %')} | "
              f"{fmt(r['recall'], ' %')} | {fmt(r['decode_p50_ms'], ' ms')} |")
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
