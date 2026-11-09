# Tilawah Coach — AI-based Quran recitation & pronunciation assessment

The learner recites into the microphone. The browser streams 16 kHz audio to the server, which recognises the
speech, aligns it with the selected ayahs and highlights mistakes word by word as the learner goes. A second
model scores selected letters for pronunciation in the background.

```
browser mic ─► AudioWorklet (16 kHz int16, 100 ms chunks) ─► WebSocket /ws
                                                               │
      ┌────────────────────────────────────────────────────────┘
      ▼
  VAD segmenter (app/vad.py) ── open segment ──► partial ASR every 1 s ─┐   fast path
      │                                                                 │
      └── closed segment (pause) ──► final ASR ──► committed words ─────┤
                                         │                              ▼
                                         │                   word alignment vs reference
                                         │                   (app/alignment.py)
                                         ▼                              │
                             CTC letter scoring (GOP)                   ▼
                             (app/pronunciation.py)  ──────►  feedback engine ─► UI highlights
                                  async path                  (app/feedback.py)
```

## Run it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt       # everything, models in-process (Windows; .venv/bin/pip elsewhere)
.venv/Scripts/python -m uvicorn app.server:app --port 8765
```

Open http://localhost:8765 in Chrome or Edge (microphone access works on `localhost`). The first start
downloads the models (~150 MB ASR, ~1.3 GB pronunciation). After that, set `HF_HUB_OFFLINE=1` to skip the hub
checks. Set environment variables the way your shell does:

```bat
:: Windows cmd
set HF_HUB_OFFLINE=1
python -m uvicorn app.server:app --port 8765
```

```powershell
# PowerShell
$env:HF_HUB_OFFLINE = "1"
python -m uvicorn app.server:app --port 8765
```

With both models loaded the server uses roughly 2–3 GB of RAM. `PRONUNCIATION=0` skips the letter model if
memory is tight.
The model pills at the top turn green when each model is ready. You can recite live, or use **Use a recording**
to stream an audio file through the same pipeline at real-time speed.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `ASR_BACKEND` | `faster-whisper` | `faster-whisper` (CTranslate2 int8, fast on CPU) or `transformers` |
| `ASR_MODEL` | `OdyAsh/faster-whisper-base-ar-quran` / `tarteel-ai/whisper-base-ar-quran` | Quran-tuned Whisper |
| `PRONUNCIATION` | `1` | `0` disables the letter-scoring layer |
| `PRON_MODEL` | `rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final` | any Arabic character-level CTC model |
| `PRON_THRESHOLD` | `0.35` | letter score below which a focus letter is flagged (calibrated with `tools/letter_probe.py`) |
| `PRON_INT8` | `0` | `1` = int8 pronunciation model: ~1.75× faster on CPU, slightly lower recall |
| `DEVICE` | `cpu` | `cuda` if a GPU is available |
| `MODEL_SERVICE_URL` | unset | use a model service at this URL instead of loading models in-process (see below) |
| `MODEL_SERVICE_TOKEN` | unset | shared secret between the app server and the model service |

## Run the models somewhere else (model service)

The models can run as a separate service on any machine with more memory or a GPU: Kaggle, Colab, a rented GPU,
a Hugging Face Space, or a university server. Your laptop then runs only the app server, which takes about
60 MB of RAM and needs no PyTorch.

```
browser ──WebSocket──► app server (laptop, ~60 MB)          model service (GPU box, ~1.7 GB)
                       VAD, alignment, letter scoring, ──HTTP──► POST /asr       -> transcript
                       verification, feedback                   POST /logprobs  -> CTC log-posteriors
```

Only the neural networks run in the service. The app server keeps all the assessment logic, so results
are identical wherever the models run (`tests/test_model_service.py` checks this).

**1. Start the model service** on the big machine (`pip install -r requirements-models.txt` there):

```bash
export MODEL_SERVICE_TOKEN=choose-a-long-random-string
export DEVICE=cuda            # on a GPU machine; leave unset for CPU
python -m uvicorn app.model_service:app --host 0.0.0.0 --port 9000
```

If the machine has no public address (Colab, Kaggle), expose the port with a tunnel, for example
`cloudflared tunnel --url http://localhost:9000`. It prints an `https://….trycloudflare.com` URL.
Always set the token when the service is reachable from outside.

**2. Point the app server at it** on your laptop (`pip install -r requirements-app.txt` is enough):

```bat
:: Windows cmd
set MODEL_SERVICE_URL=https://your-tunnel.trycloudflare.com
set MODEL_SERVICE_TOKEN=choose-a-long-random-string
python -m uvicorn app.server:app --port 8765
```

**On Kaggle's free GPU:** `notebooks/kaggle_model_service.ipynb` does step 1 for you. It gets the code, starts
the service on the GPU, times both models, opens the Cloudflare tunnel, and prints the step 2 commands with the
URL and token filled in. Its last section runs the evaluation tools on the GPU for model comparisons. To
get the code onto Kaggle without GitHub, run `git archive --format=zip -o fyp-2-src.zip HEAD` and upload the zip
as a private Kaggle dataset. Commit first: the laptop and the service must run the same version.

`/api/health` then reports `"where": "remote"`. `tools/evaluate.py` and `tools/letter_probe.py` read the same
variables, so evaluations can use the remote models too. Without `MODEL_SERVICE_URL`, everything runs
in-process as before.

**Measured with the service on Kaggle (T4 GPU, Cloudflare tunnel) on 2026-09-24**, streamed through the app
server on the laptop with `tools/ws_eval.py`. Accuracy was identical to running everything on the laptop, and
every speed number improved:

| Measure | Laptop only (CPU) | Kaggle GPU via tunnel |
| --- | --- | --- |
| Studio set: WER / mistakes caught | 0 % / 55 of 55 | 0 % / 55 of 55 |
| Real Al-Kawthar recording: false flags | 0 | 0 |
| Live feedback latency p50 / p90 | 1036 / 1533 ms | 614 / 886 ms |
| Word results after the learner stops | 0.55 s | 0.30 s |
| Letter scores after the learner stops | 3.5 s | 1.0 s |
| App-server RAM on the laptop | ~1.4 GB+ | 67 MB |

About 300 ms of each ~580 ms ASR call is the tunnel round trip, so the network now dominates, not the GPU.
The laptop figures are pooled over all updates; the Kaggle figures are the median of per-session values.

Measured on this laptop (8 GB RAM, CPU only, both processes on the same machine), streaming the Al-Kawthar
recording: the app server used 63 MB and the model service 1.68 GB. Results were identical (0 flags). Latency
was worse than in-process (p50 1.95 s), because the laptop was paging under memory pressure. The
service is meant to run on a machine with spare RAM or a GPU.

## How it works

| Stage | File | Method |
| --- | --- | --- |
| Reference text | `app/quran.py`, `data/quran.json` | Tanzil *simple* (display) + *simple-clean* (matching); `tools/fetch_quran.py` adds surahs |
| Normalisation | `app/arabic.py` | strip tashkeel/Quranic marks, unify alef/ya/ta-marbuta/hamza forms |
| Segmentation | `app/vad.py` | adaptive-noise-floor energy VAD draws the boundaries (after a 600 ms pause, or at the quietest point before 20 s); a Silero neural VAD gate discards segments with no speech in them (fan, bell, knocks) before the ASR sees them |
| Noise filtering | `app/asr.py`, `app/session.py` | Whisper segments that are likely non-speech and low-confidence (or abnormally repetitive) are dropped; an extra word is kept only if including it raises the audio's CTC likelihood by `EXTRA_MARGIN` |
| ASR | `app/asr.py` | Whisper-base fine-tuned on Quran recitation |
| Alignment | `app/alignment.py` | semi-global, affine-gap (Gotoh) edit distance with fuzzy word similarity. Every heard word is explained. Words not reached yet cost less than a skip, so an early stop is not mistaken for skipping. Skipped or repeated stretches come out as one run, preferably a whole ayah |
| Word errors | `app/alignment.py` | `correct`, `close` (≥ 60 % similar), `wrong`, `missing`, `extra`; words from the live partial decode are marked tentative |
| Verification | `app/pronunciation.py`, `app/session.py` | a `close`/`wrong` word is checked against the audio, not just the transcript: it stays flagged only if the ASR's version has a CTC log-likelihood at least `VERIFY_MARGIN` (4.0) higher than the expected word's. Until checked, it shows as `checking` |
| Pronunciation | `app/pronunciation.py` | Quran-tuned wav2vec2 (XLS-R) CTC model; Viterbi forced alignment of the reference letters, then a GOP-style score per letter: P(expected) renormalised over letter tokens at its best frame. Silent letters (hamzat al-wasl, sun-letter lam) are skipped |
| Feedback | `app/feedback.py` | per-mistake messages with location, and articulation (makhraj) hints for flagged letters |
| Streaming | `app/session.py` | fast path (partial decodes) + commit worker + async pronunciation path; latency is measured per update |

**Learner level** (selector in the UI): *Beginner* accepts common approximations of hard letters (ض may sound
like ظ, د, ز or ذ; ظ like ز or ذ), *Intermediate* accepts ض as ظ, and *Advanced (tajweed)* accepts only the
exact sounds. The letters and their approximations are listed in `LEVELS` in `app/pronunciation.py`.

**Saved sessions.** Every recitation is saved to `sessions/<time>_<surah>-<from>-<to>/` as `audio.wav` plus
`session.json` (settings, final result, and a ready-to-label manifest row). The folder stays on this machine and
is git-ignored. `SAVE_SESSIONS=0` turns saving off. To add a session to the real test set, fill in `recited`
and `errors` in its `manifest_row` and append that row to `testset/real/manifest.jsonl`.

Pronunciation flags are limited to the letters learners most often substitute:
ع ح خ غ ق ض ظ ذ ث ص ط ه ء (`FOCUS_LETTERS`). Every letter is still scored and shown on hover.

## Recording conditions

The app handles common household noise (see below), but a clean signal still helps the speech models most:

- Sit close to the microphone. A headset or earphone mic 5–10 cm from the mouth beats a laptop's built-in mic
  at arm's length, because the recitation then dominates the fan and street noise.
- Point the fan away from the microphone or slow it down, and close a window facing the street.
- Keep the browser's noise suppression on (the app requests it). Avoid background speech: the app cannot tell
  another person's speech from yours.
- Pause a moment before and after reciting, and stop the recording when you finish.

## Evaluate

```bash
# Build a labelled set with controlled mistakes (skipped ayah, repeated ayah) from everyayah.com recordings
.venv/Scripts/python tools/make_testset.py --surahs 1 112 113 114 --reciters Alafasy_128kbps Husary_128kbps
# WER, mistake-detection precision/recall, letter-flag precision/recall (if labelled), latency
.venv/Scripts/python tools/evaluate.py testset/manifest.jsonl --report testset/report.json
.venv/Scripts/python tools/evaluate.py testset/manifest.jsonl --realtime   # true streaming latency
# Pronunciation layer: controlled letter substitutions + threshold sweep
.venv/Scripts/python tools/letter_probe.py
# Noisy copy of a test set (fan hum under the recitation, bells/knocks before and after)
.venv/Scripts/python tools/make_noisy_testset.py testset/manifest.jsonl --out testset/noisy
# Choose EXTRA_MARGIN from evaluation reports (extras' audio support), offline
.venv/Scripts/python tools/calibrate_extras.py testset/manifest.jsonl:testset/report.json
# Compare speech models on the same sets (converts to faster-whisper format as needed)
.venv/Scripts/python tools/compare_asr.py --sets testset/real/manifest.jsonl --models OdyAsh/faster-whisper-base-ar-quran deepdml/whisper-small-ar-quran-mix-norm
# Drive a running server over the WebSocket, exactly like the browser
.venv/Scripts/python tools/ws_client.py testset/audio/Alafasy_128kbps_001_skip5.wav --surah 1
# Evaluate a whole manifest through a running server (local or remote models, network included)
.venv/Scripts/python tools/ws_eval.py testset/manifest.jsonl
```

### Results so far (2026-09-23, 8-core CPU, no GPU)

Controlled-mistake set: surahs 1, 112, 113, 114 × Alafasy and Husary, 24 recordings, 521 words, streamed at
real-time speed with both models on (`--realtime`).

| Metric | Result |
| --- | --- |
| WER | 0.0 % (0 / 521 words) |
| Mistake detection (skipped / repeated ayahs) | precision 100 %, recall 100 % (55 labelled mistakes); no false alarms on the 8 correct recitations |
| Live feedback latency (audio received → highlight) | p50 1.04 s, p90 1.53 s |
| Word results after the learner stops | mean 0.55 s, max 1.9 s |
| Letter scores after the learner stops | mean 3.5 s, max 7.0 s |

Letter-substitution probe (`tools/letter_probe.py`, 240 probes over 11 confusion pairs, 114 correctly recited
focus letters). At `PRON_THRESHOLD=0.35`, recall is 100 % with 0 % false alarms. Any threshold from 0.3 to 0.7
gives the same result. With `PRON_INT8=1`, recall is 99.2 %.

**Real recording.** One correct learner recitation of Al-Kawthar (a real m4a recording)
(`testset/real/`). Before verification, 4 of 10 words were falsely flagged `close`: the ASR wrote الفوقر, فاصفل,
للربك, الابطر. With verification, 0 are flagged. The expected word fitted the audio better for three of them
(Δ −5.8, −10.8, −4.0). For the fourth, الأبتر, the model leaned slightly towards ط (Δ +1.4), still below the
4.0 margin. `VERIFY_MARGIN` was calibrated against 298 single-letter substitutions in studio audio: median Δ +12.5,
and 92.6 % stay flagged at any margin from 2 to 6.

**Room noise (2026-09-24).** A real test next to a ceiling fan with a street vendor's bell had noise
transcribed as extra words, including after the surah ended. The noisy set reproduces this: the 24 studio
recordings with fan hum under the recitation (10 dB below it) and 3 s / 6 s of noise with bells and knocks before
and after (`tools/make_noisy_testset.py`). Labels are unchanged, so any extra word is a false alarm.

| Noisy set (24 recordings) | Before | After (Silero gate + Whisper filter + extra-word check) |
| --- | --- | --- |
| False extra words from noise | 14 | **0** |
| Mistake detection precision / recall | 38.4 % / 50.9 % | 49.1 % / 50.9 % |
| WER | 17.7 % | 15.4 % |

The clean studio set is unchanged (0 % WER, 55/55, all 27 genuinely repeated words kept by the extra-word
check), and so is the real Al-Kawthar recording (0 flags). The remaining noisy-set errors are the ASR mishearing
real words under the fan, which hides genuine mistakes (recall 51 %). That calls for a more robust ASR model,
not for more filtering: see [docs/research/asr_comparison_2026-09-24.md](docs/research/asr_comparison_2026-09-24.md).
In short, `deepdml/whisper-small-ar-quran-mix-norm` got 9 of 10 words right on the real learner recording (base
6 of 10) at ~3× the CPU decode time, and large-v3-turbo is the most robust in noise but regressed on clean audio
and needs a GPU.

These are upper bounds. The recordings are professional reciters in studio conditions, and the mistakes are
whole-ayah skips and repeats, or complete letter swaps. Real learners, microphones and partial
mis-articulations will score lower. Measure that by adding labelled learner recordings to the manifest.

Manifest rows are JSON lines. To add your own recordings with real mistakes, append rows like:

```json
{"audio": "audio/student01.wav", "surah": 1, "from": 1, "to": 7,
 "recited": "بسم الله الرحمن الرحيم الحمد لله رب العلمين",
 "errors": [{"ref_index": 7, "type": "wrong"}],
 "letters": [{"ref_index": 5, "cluster": 0}]}
```

`ref_index` is the word's position in the selected range. `cluster` is the letter's position in that word, as in
`/api/reference`. Error types: `wrong`/`close` (scored as one class), `missing`, `extra`.

## Tests

```bash
.venv/Scripts/python -m pytest -q
```

The 21 tests cover normalisation, alignment (including repeated-phrase and skipped-ayah cases), CTC forced alignment,
VAD segmentation, a full streaming session, word verification (including the numpy CTC against PyTorch's),
and the model service end to end (auth, remote results equal local, app server runs without torch).
They use a scripted ASR, so they need no model download.

## Limitations

- The pronunciation model was trained on correct recitation, not on labelled learner mistakes. The
  letter probe uses clean, complete substitutions; real learners make partial mis-articulations, so real
  recall will be lower. Re-tune `PRON_THRESHOLD` once labelled learner recordings exist.
- A general Arabic CTC model (`jonatasgrosman/wav2vec2-large-xlsr-53-arabic`, the base of the default) was
  tried first and rejected: it drops ayah-final prolonged sounds and misheard slow tartil (خلق → أقال),
  which flagged expert reciters.
- Only one real learner recording so far, and it is correct. Verification clears ASR drift on it, but whether
  real word mistakes survive verification is measured only with simulated substitutions. Add learner
  recordings with deliberate mistakes to `testset/real/manifest.jsonl`.
- Suspect words wait for the pronunciation pass before their verdict. With no pause in the recitation,
  that pass runs after the learner stops (9 s for 17 s of audio on this CPU; `PRON_INT8=1` is ~1.75× faster).
- The controlled-mistake set covers skipped and repeated ayahs only. Word substitutions and letter errors need
  real learner recordings with labels.
- No Tajweed rules yet (madd length, ghunnah, qalqalah, idgham…). They belong in separate modules on top of the
  letter alignment, as planned.
- Whisper normalises what it hears towards valid Quran text. A slightly mispronounced word can still come out
  "correct", which is why the pronunciation layer is separate.
