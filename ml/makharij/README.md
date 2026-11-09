# Makharij confusion classifier (scaffold)

Whisper "corrects" a mispronounced letter to the right word, so a learner who
says **س** like **ص** passes the text check. This folder classifies the *audio*
of a single letter into one of four confusable pairs:

| pair | letters |
| --- | --- |
| س/ص | seen / saad |
| د/ض | daal / daad |
| ك/ق | kaaf / qaaf |
| ه/ح | haa / haa (pharyngeal) |

**Status: pipeline only. No model has been trained on real data and no
accuracy exists.** Running it on the publisher's clips (`--smoke`) proves the
code works; those numbers are a reference-audio sanity check and are not
printed on purpose.

## Files

| file | what |
| --- | --- |
| `data.py` | labels CSV → `Clip(speaker_id, audio_path, letter)`; `publisher_dataset()` for smoke runs |
| `features.py` | MFCC statistics (numpy only, 27 numbers per clip) or wav2vec2 embeddings (optional) |
| `models.py` | baseline SVM (RBF) and a small MLP head, scikit-learn pipelines |
| `train.py` | speaker-wise split (`eval/split.py`), fit one model per pair, save to `runs/<name>/` |
| `evaluate.py` | predict the held-out speakers, confusion matrix per pair as CSV + PNG |

## Train on real recordings

1. Collect recordings with `/collect` (pages 1–12) and have a qari label the
   letters. For this classifier you need *single-letter* clips, cut if needed,
   with the letter the learner **actually produced** (so a learner who was
   asked for ص but said س is a سample of س).
2. Write `data/makharij-labels.csv`:

   ```csv
   speaker_id,audio_path,letter
   S01,S01/p01-r015.webm,س
   S01,S01/p01-r017.webm,ص
   S02,S02/p07-r013.webm,س
   ```

   Paths are relative to the CSV. Aim for several speakers and both letters
   of a pair from every speaker; the split is by speaker, so with fewer than
   three speakers there is nothing meaningful to hold out.
3. Train and evaluate (backend venv active, from the repo root):

   ```bash
   python ml/makharij/train.py data/makharij-labels.csv --features mfcc --model svm --name mfcc-svm
   python ml/makharij/evaluate.py ml/makharij/runs/mfcc-svm
   ```

   `runs/mfcc-svm/` then holds `confusion-<pair>.csv/.png`, `results.json` and
   the held-out rows. Repeat with `--model mlp`, and with
   `--features wav2vec2` after `pip install torch transformers` (downloads
   `facebook/wav2vec2-base`, ~360 MB; an Arabic-tuned model can be set in
   `features.py`).
4. Report the per-pair confusion matrices and accuracy on the held-out
   speakers only. Try a few `--seed` values: with a handful of speakers the
   split matters more than the model.

## Smoke run

```bash
python ml/makharij/train.py --smoke
python ml/makharij/evaluate.py ml/makharij/runs/smoke
```

Uses the publisher's single-letter clips with the *page* as a pseudo
speaker. It exists to catch broken code, nothing else.

## Where the FYP integration would go

Once a model is worth using, `backend/tajweed/` is the place for a
`makharij.py` that loads `runs/<name>/<pair>.joblib`, runs `features.extract`
on the learner's clip for items whose expected letter is in a pair, and adds a
hint to `/api/evaluate` behind a config flag, the same pattern as `madd.py`.
