"""Generates notebooks/kaggle_model_service.ipynb. Edit this script, then run it from the project root."""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md(r'''
# Tilawah Coach: model service on a Kaggle GPU

This notebook runs the two heavy models (Quran Whisper ASR + wav2vec2 letter model) on Kaggle's GPU.
Your laptop runs only the light app server (~60 MB RAM) and connects to this notebook over a secure tunnel.

**Before you run**, open the notebook's settings panel (right side):

1. **Accelerator:** *GPU T4 x2* (or a single T4).
2. **Internet:** **On**. Needed to install packages, download the models and open the tunnel. Kaggle only
   allows this after you verify your phone number in your account settings.
3. **Code:** either attach your project zip as a dataset (**Add Input**), or put a GitHub URL in the config
   cell below. The cell explains both.
4. *(Recommended)* **Add-ons → Secrets:** add `MODEL_SERVICE_TOKEN` (any long random string), so the
   password stays the same across sessions. Without it, the notebook makes a new one each run.

Then **Run All**. Section 7 prints the commands to paste on your laptop, and section 8 keeps the service
alive until you stop it.

> The tunnel address and token printed below give access to your model service. Don't publish this notebook
> with its outputs; clear outputs before making any version public.
''')

md('## 1. Configuration')
code(r'''
# Where the project code comes from. Option A wins if a matching dataset is attached.
#   Option A: Kaggle dataset. On your laptop:  git archive --format=zip -o fyp-2-src.zip HEAD
#             then Datasets -> New Dataset -> upload fyp-2-src.zip, and attach it here with "Add Input".
#   Option B: GitHub. Public repo: set REPO_URL. Private repo: also add a Kaggle secret GITHUB_TOKEN.
REPO_URL = ""            # e.g. "https://github.com/<you>/fyp-2.git"
REPO_BRANCH = "main"

PORT = 9000
DEVICE = "cuda"          # "cpu" if you did not enable a GPU
PRON_MODEL = "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final"
ASR_MODEL = "OdyAsh/faster-whisper-base-ar-quran"
WORKDIR = "/kaggle/working/fyp-2"
''')

md('## 2. Get the code')
code(r'''
import glob, os, shutil, subprocess, zipfile

def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None

shutil.rmtree(WORKDIR, ignore_errors=True)
trees = glob.glob("/kaggle/input/**/app/model_service.py", recursive=True)
zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
if trees:                                   # dataset attached as a folder (Kaggle usually extracts zips)
    src = os.path.dirname(os.path.dirname(trees[0]))
    shutil.copytree(src, WORKDIR)
    print("code from dataset folder:", src)
elif zips:
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(WORKDIR)
    print("code from dataset zip:", zips[0])
elif REPO_URL:
    url = REPO_URL
    token = secret("GITHUB_TOKEN")
    if token and url.startswith("https://"):
        url = url.replace("https://", f"https://x-access-token:{token}@", 1)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", REPO_BRANCH, url, WORKDIR], check=True)
    print("code from", REPO_URL)
else:
    raise SystemExit("No code found: attach the project zip as a dataset, or set REPO_URL in the config cell.")

os.chdir(WORKDIR)
assert os.path.exists("app/model_service.py"), "unexpected layout: app/model_service.py missing"
print(sorted(os.listdir(WORKDIR)))
''')

md(r'''
## 3. Install packages

Kaggle already ships a CUDA build of PyTorch and `transformers`. Only the missing packages are installed,
**not** `requirements-models.txt`, because its CPU index could replace the GPU PyTorch.
''')
code(r'''
!pip install -q faster-whisper fastapi "uvicorn[standard]" soundfile onnxruntime
import torch, transformers
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
print("transformers", transformers.__version__)
if DEVICE == "cuda" and not torch.cuda.is_available():
    raise SystemExit("DEVICE='cuda' but no GPU: enable the GPU accelerator in the settings, or set DEVICE='cpu'.")
''')

md('## 4. Start the model service')
code(r'''
import json, secrets, sys, time, urllib.request

TOKEN = secret("MODEL_SERVICE_TOKEN") or secrets.token_urlsafe(24)
env = dict(os.environ, DEVICE=DEVICE, PRON_MODEL=PRON_MODEL, ASR_MODEL=ASR_MODEL,
           ASR_BACKEND="faster-whisper", MODEL_SERVICE_TOKEN=TOKEN)
env.pop("MODEL_SERVICE_URL", None)
log_file = open("/kaggle/working/model_service.log", "w")
service = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.model_service:app",
                            "--host", "127.0.0.1", "--port", str(PORT)],
                           cwd=WORKDIR, env=env, stdout=log_file, stderr=subprocess.STDOUT)

def info():
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/info", headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.load(r)

t0 = time.time()
while True:                                   # the first run downloads ~1.5 GB of models
    if service.poll() is not None:
        print(open("/kaggle/working/model_service.log").read()[-3000:])
        raise SystemExit("model service exited; log above")
    try:
        i = info()
        if i["asr"] and i["pronunciation"]:
            break
    except Exception:
        pass
    time.sleep(3)
print(f"model service ready in {time.time() - t0:.0f}s on {i['device']}: ASR={i['asr']}  letters={i['pronunciation']}")
''')

md(r'''
## 5. Speed check on this GPU

This times both models on real ayahs through the same HTTP API your laptop will use. The laptop numbers
for comparison (8 GB RAM, CPU only): ASR about 1.0 s per call; letter model about 2–4 s per ayah.
''')
code(r'''
sys.path.insert(0, WORKDIR)
import numpy as np
from app import audio
from app.remote import ModelServiceClient

client = ModelServiceClient(f"http://127.0.0.1:{PORT}", TOKEN)
os.makedirs("/kaggle/working/bench", exist_ok=True)
for name in ["001007", "112001"]:            # Al-Fatiha 7 (long), Al-Ikhlas 1 (short)
    path = f"/kaggle/working/bench/{name}.mp3"
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://everyayah.com/data/Alafasy_128kbps/{name}.mp3", path)
    x = audio.load(path)
    for label, fn in [("asr", client.transcribe), ("letters", client.log_probs)]:
        fn(x)                                 # warm-up
        times = []
        for _ in range(3):
            t = time.perf_counter(); out = fn(x); times.append(time.perf_counter() - t)
        extra = f"  -> {out}" if label == "asr" else f"  -> {out.shape[0]} frames"
        print(f"{name} ({len(x)/16000:4.1f}s audio) {label:<7} median {np.median(times):.2f}s{extra}")
''')

md(r'''
## 6. Open the tunnel

This uses a free Cloudflare *quick tunnel* (no account needed) to give the service a public HTTPS address.
The address changes every session.
''')
code(r'''
import re, stat
cf = "/kaggle/working/cloudflared"
if not os.path.exists(cf):
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", cf)
    os.chmod(cf, os.stat(cf).st_mode | stat.S_IEXEC)
tunnel_log = open("/kaggle/working/tunnel.log", "w")
tunnel = subprocess.Popen([cf, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                          stdout=tunnel_log, stderr=subprocess.STDOUT)
PUBLIC_URL = None
for _ in range(60):
    m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", open("/kaggle/working/tunnel.log").read())
    if m:
        PUBLIC_URL = m.group(0)
        break
    time.sleep(1)
if not PUBLIC_URL:
    print(open("/kaggle/working/tunnel.log").read()[-2000:])
    raise SystemExit("tunnel did not start; log above")

for _ in range(30):                           # the new hostname can take a few seconds to resolve
    try:
        remote_info = ModelServiceClient(PUBLIC_URL, TOKEN, retries=0).info()
        break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit(f"{PUBLIC_URL} is not reachable yet; re-run this cell")
print("tunnel OK:", PUBLIC_URL, "->", remote_info["device"])
''')

md('## 7. Connect your laptop')
code(r'''
print(f"""On your laptop, in the project folder (the light install is enough: pip install -r requirements-app.txt):

Windows cmd:
    set MODEL_SERVICE_URL={PUBLIC_URL}
    set MODEL_SERVICE_TOKEN={TOKEN}
    python -m uvicorn app.server:app --port 8765

PowerShell:
    $env:MODEL_SERVICE_URL = "{PUBLIC_URL}"
    $env:MODEL_SERVICE_TOKEN = "{TOKEN}"
    python -m uvicorn app.server:app --port 8765

Then open http://localhost:8765. /api/health should show "where": "remote".
The evaluation tools use the same two variables, e.g.:
    python tools/evaluate.py testset/real/manifest.jsonl --realtime
""")
''')

md(r'''
## 8. Keep the service running

This cell keeps the session busy and prints a status line every minute, with the number of model calls
served. **Interrupt** it (■ Stop) when you are done, then run the last cell to shut down. Kaggle ends GPU
sessions after its time limit (about 12 h), and GPU time counts against your weekly quota while the notebook
runs, even if the laptop is not using it. Stop the session when you finish.
''')
code(r'''
try:
    while True:
        calls = sum(1 for line in open("/kaggle/working/model_service.log") if "POST /" in line)
        up_s, up_t = service.poll() is None, tunnel.poll() is None
        print(time.strftime("%H:%M:%S"), "service up" if up_s else "SERVICE DOWN",
              "| tunnel up" if up_t else "| TUNNEL DOWN", f"| {calls} model calls served", flush=True)
        if not (up_s and up_t):
            print(open("/kaggle/working/model_service.log").read()[-1500:])
            break
        time.sleep(60)
except KeyboardInterrupt:
    print("stopped watching; the service keeps running until you run the next cell")
''')

md('## 9. Shut down')
code(r'''
for name in ("tunnel", "service"):          # either may not exist (e.g. research-only runs)
    proc = globals().get(name)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=30)
        print(name, "stopped")
''')

md(r'''
---
## Optional: research on the GPU (no laptop needed)

These cells run the evaluation tools **in this notebook** with the models on the GPU. They're for comparing
models the laptop cannot load. They need sections 1–3 only. If the model service from section 4 is running,
run section 9 first to free GPU memory.

To include your own recordings, upload them (with `testset/real/manifest.jsonl`) as a Kaggle dataset and set
`REAL_MANIFEST` below.
''')
code(r'''
os.chdir(WORKDIR)
run_env = dict(os.environ, DEVICE=DEVICE, PRON_MODEL=PRON_MODEL, ASR_BACKEND="faster-whisper")
run_env.pop("MODEL_SERVICE_URL", None)       # in-process models on this GPU
# Studio test set with controlled mistakes, plus a copy with household noise.
subprocess.run([sys.executable, "tools/make_testset.py", "--surahs", "1", "112", "113", "114",
                "--reciters", "Alafasy_128kbps", "Husary_128kbps"], env=run_env, check=True)
subprocess.run([sys.executable, "tools/make_noisy_testset.py", "testset/manifest.jsonl", "--out", "testset/noisy"],
               env=run_env, check=True)
REAL_MANIFEST = ""                           # e.g. "/kaggle/input/<your-dataset>/manifest.jsonl"
''')
code(r'''
# Speech-model comparison: same streaming pipeline, only the ASR changes.
MODELS = [
    "OdyAsh/faster-whisper-base-ar-quran",                            # current default (base, 74M)
    "deepdml/whisper-small-ar-quran-mix-norm",                        # small, 244M
    "deepdml/whisper-medium-ar-quran-mix-norm",                       # medium, 769M
    "naazimsnh02/whisper-large-v3-turbo-ar-quran",                    # large-v3-turbo, 809M
    "MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix-ct2",
]
sets = ["testset/noisy/manifest.jsonl", "testset/manifest.jsonl"] + ([REAL_MANIFEST] if REAL_MANIFEST else [])
subprocess.run([sys.executable, "tools/compare_asr.py", "--device", DEVICE, "--models", *MODELS,
                "--sets", *sets, "--out", "/kaggle/working/asr_comparison.json"], env=run_env, check=True)
''')
code(r'''
# Letter-substitution probe: recall / false alarms of the pronunciation layer, threshold sweep.
subprocess.run([sys.executable, "tools/letter_probe.py"], env=run_env, check=True)
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path("notebooks/kaggle_model_service.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(len(cells), "cells ->", out)
