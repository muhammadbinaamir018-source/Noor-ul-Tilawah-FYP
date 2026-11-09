# Speech-model comparison, 2026-09-24 (laptop CPU, int8)

Run with `tools/compare_asr.py --device cpu` (ASR only, no letter model, so close and extra words are **not**
verified in these runs). The pipeline is the same as in the app: energy segmentation, Silero speech gate,
Whisper non-speech filter, affine-gap alignment with word joins.

| Model | Params | CPU decode p50 |
| --- | --- | --- |
| `OdyAsh/faster-whisper-base-ar-quran` (current default) | 74 M | 0.9–1.3 s |
| `deepdml/whisper-small-ar-quran-mix-norm` | 244 M | 2.6–2.9 s |
| `MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix-ct2` | 809 M | ~12 s |

## Results that are valid

**Real recording** (the user's correct Al-Kawthar recitation, 10 words, laptop microphone). WER, i.e. how many
words the ASR itself got wrong:

| base | small | turbo |
| --- | --- | --- |
| 40 % (4 of 10) | **10 % (1 of 10)** | 40 % (4 of 10) |

**Mistake detection on whole sets** (precision / recall of the 55 labelled skip/repeat mistakes):

| Set | base | small | turbo |
| --- | --- | --- | --- |
| Noisy studio set (fan under the recitation, bells and knocks around it) | 43.8 % / 50.9 % | 43.8 % / 50.9 % | **59.7 % / 72.7 %** |
| Clean studio set | 100 % / 100 % | 100 % / 100 % | 81.5 % / 96.4 % |

**One noisy recording** (Al-Fatiha, Alafasy, noisy copy), WER: base 44.8 %, small 13.8 %, turbo 3.4 %.
For comparison, the base model's WER on the whole noisy set is 15.4 % (80/521 words).

## What is not valid in this run

The whole-set WER of the small and turbo models on the noisy and clean sets was **not measured**: the tool read
the first recording's WER line instead of the summary line. This is fixed in `tools/compare_asr.py` (summary-line
pattern); rerun to fill it in, preferably on a GPU (the Kaggle notebook's research section runs this comparison).

## Reading

- **small** is the best candidate for the laptop. It got 9 of 10 words right on the real learner recording (base:
  6 of 10), it's no worse on either studio set, and it decodes in ~2.7 s on this CPU (about 3× base, so live
  feedback would be slower). The evidence for the real-voice gain is **one recording**, so collect more (saved
  sessions) before making it the default.
- **turbo** is by far the most robust in noise (recall 51 % → 73 % on the noisy set), but it **regressed on clean
  studio audio** (18 % of its detections were false), and it gains nothing on the real recording. At ~12 s per
  decode it is unusable live on this CPU; judge it on a GPU, after looking at its clean-set false alarms.
- No model fixes everything. The per-model differences on the real recording show that learner speech needs
  its own test set: the professional studio set is too easy (every model scores 0 % WER on it).
- Not compared yet: `deepdml/whisper-medium-ar-quran-mix-norm` and `naazimsnh02/whisper-large-v3-turbo-ar-quran`
  (too large to convert on 8 GB RAM; included in the Kaggle notebook's research section).
