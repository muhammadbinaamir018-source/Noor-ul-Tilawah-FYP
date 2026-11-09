# Progress log

One section per feature: the plan written before starting, then what was
actually done, the commit/tag, and known limitations. Checkpoints are git tags
`ckpt-NN-<name>`; `git checkout ckpt-01-eval` rolls the tree back to that point.

## 0. Page mapping (done before this queue, commits 4aa1910..3fdf401)

All 39 Qaida scans mapped to the publisher's 1398 clips by
`backend/scripts/map_qaida.py`; order verified against Whisper transcripts of
every reference clip. 55 tests green.

## 1. Evaluation harness — `eval/` (tag `ckpt-01-eval`)

### Plan

1. `eval/metrics.py` (pure logic, TDD): accuracy, False Accept Rate (wrong
   recitation passed), False Reject Rate (correct recitation failed), per-item
   accuracy, confusion matrix label × verdict, threshold sweep. "unclear"
   counts as a reject.
2. `eval/split.py` (pure logic, TDD): speaker-wise train/test split; a check
   that raises on any speaker overlap.
3. `eval/labels.py`: read the labels CSV, validate item ids against the page
   JSON, pages p01–p12 only.
4. `eval/run_eval.py`: CLI. Transcribes each clip with the backend's own
   `asr.transcribe` and scores with `matching.evaluate_match` (no duplicated
   pipeline), caches transcripts so sweeps do not re-run ASR, writes CSV + PNG
   to `eval/results/`. `--threshold`, `--sweep`, `--dry-run`.
5. `--dry-run` builds a synthetic label set from the publisher's reference
   clips (each clip labelled correct for its own item and wrong for a different
   item on the same page). Its numbers are a **reference-audio sanity check**,
   not an evaluation result.
6. matplotlib added to backend/requirements.txt; `eval/results/` git-ignored.

### Done (tag `ckpt-01-eval`)

- `eval/labels.py`, `eval/metrics.py`, `eval/split.py`, `eval/run_eval.py` as
  planned. 15 unit tests in `backend/tests/test_eval.py` (metrics, sweep
  monotonicity, speaker split + overlap guard, CSV validation, and an
  end-to-end run with a fake transcriber that checks every output file).
- `python eval/run_eval.py --dry-run --sweep` ran end to end on 12 publisher
  items (24 rows, 26 s of ASR on CPU). Output is labelled
  "reference-audio sanity check" in metrics.csv and on both plots.
- Test suite: 70 passed.

Known limitations

- No learner recordings exist yet, so no real metric has been produced. The
  dry-run's numbers (FRR 0.75 on reference clips) only reflect that the
  publisher's clips contain Urdu explanations after the Arabic; they say
  nothing about learners.
- `--threshold` applies one value to every mode; per-mode overrides are not
  exposed (edit backend/.env for that).
- Transcript cache is keyed by audio content only; if the ASR model changes,
  delete eval/results/transcripts.json.

## 2. Madd duration module — `backend/tajweed/madd.py` (tag `ckpt-02-madd`)

### Plan

1. Audio in: any container PyAV decodes (webm/opus from the browser, mp3
   publisher clips, wav fixtures) → mono 16 kHz float. numpy + PyAV only; no
   librosa/Parselmouth, to keep the install the same for the whole team.
2. Voiced-segment detection from a 10 ms RMS envelope: frames above
   −30 dB relative to the clip's loudest frame, segments ≥ 40 ms, gaps ≤ 60 ms
   bridged. The **longest** segment is the vowel being judged; the **shortest
   segment ≥ 60 ms** in the same clip is the speaking-speed reference (one
   harakah). counts = longest / reference. If there is only one segment the
   reference falls back to a default harakah length (configurable, 0.2 s), so
   lone letters still get a verdict but with lower confidence.
3. Classification by counts: too-short (< 1.5), 2-count [1.5, 3), 4-count
   [3, 5.5), too-long (≥ 5.5); confidence = distance of counts from the
   nearest class boundary, scaled to 0–1. Silent or < 0.15 s audio → verdict
   "unclear" with confidence 0 and hint.
4. `MaddResult`: measured_s, reference_s, counts, expected_counts, verdict,
   ok (verdict matches expectation), confidence, hint (plain language, e.g.
   "Stretch the alif to about two counts").
5. Unit tests first (synthetic vowels with known durations: 2-count, 4-count,
   too short, too long; silence; 50 ms clip; clipped/overdriven audio) plus
   publisher clips as fixtures (a short-vowel clip must measure shorter than a
   madd clip on the same letter). Measurements on publisher clips are a
   reference-audio sanity check, not an evaluation.
6. Wire into POST /api/evaluate behind `TILAWAH_MADD_ENABLED` (default off):
   when on and the item's expected text contains a madd letter, the response
   gains a `madd` object. One integration test with TestClient and a fake ASR.

### Done (tag `ckpt-02-madd`)

- `backend/tajweed/madd.py` as planned (numpy + PyAV only). `decode`,
  `measure`, `classify`, `analyse`, `expected_counts_for`; `MaddResult`
  carries measured and reference length, counts, expected counts, verdict,
  ok, confidence, hint, clipped.
- 29 unit tests in `backend/tests/test_madd.py`, written first: synthetic
  2-count / 4-count / too-short / too-long vowels, silence, 50 ms clip,
  clipped audio, missing reference segment, gap bridging, expected counts
  from tashkeel, and three publisher-clip pairs (madd clip's longest voiced
  stretch > short-vowel clip of the same letter: reference-audio sanity
  check).
- `/api/evaluate` returns `madd` when `TILAWAH_MADD_ENABLED=true` and the
  item has a long vowel (letter-name items skipped). `MaddInfo` schema,
  `MaddInfo` TS type. 3 integration tests in `test_api_evaluate.py` with a
  fake ASR (madd present and ok, too-short reported without changing the
  text verdict, absent when disabled).
- Test suite: 102 passed; frontend lint and build clean.

Known limitations

- Count boundaries (1.5 / 3 / 5.5) and the 0.2 s default harakah come from
  synthetic audio and the publisher's clips, not from learners. They must be
  re-tuned once labelled recordings exist; until then the flag stays off.
- A word whose long vowel is not separated from a neighbouring vowel by a
  consonant dip (e.g. يَا) is measured as one segment, so counts come out
  high. Items with several long vowels report only the longest.
- No UI shows the madd hint yet; the field is in the API response and the
  TS type only.

## 3. Collection page — `/collect` (tag `ckpt-03-collect`)

### Plan

1. Backend `app/services/collection.py`: items = recitable regions of
   p01–p12 in reading order; `new_speaker` hands out S01, S02, … from
   `data/recordings/speakers.csv` (consent, age group, native: nothing
   identifying); `save_recording` writes `data/recordings/<speaker>/<item>.webm`
   (re-record overwrites) and upserts `metadata.csv`; `skip` records a skipped
   item; `labels_template` writes `labels-template.csv` in the eval harness
   format with the label column blank for a qari.
2. Router `/api/collect`: GET items, POST speakers (consent must be true →
   422 otherwise), POST recordings (multipart), POST skip, GET
   progress/{speaker}, GET export (CSV download).
3. Frontend `/collect`: consent form → speaker id (kept in localStorage so a
   refresh resumes) → one item at a time: the Qaida cell cropped from the
   scan, the expected text, play-reference, record (auto-stops on silence),
   playback, Save & next / Re-record / Skip, progress counter, export link at
   the end.
4. Integration tests on the API with a temp recordings dir, including "export
   → fill labels → eval.labels.read_labels accepts it".
5. `data/recordings/` is already git-ignored.

### Done (tag `ckpt-03-collect`)

- Backend: `app/services/collection.py`, `app/routers/collect.py`, schemas
  `CollectItem`, `NewSpeaker`, `Speaker`, `SkipRequest`; setting
  `recordings_dir`. 6 integration tests in `tests/test_collect.py`, including
  export → fill the label column → `eval.labels.read_labels` accepts it (so
  the collection output is guaranteed to feed the harness).
- Frontend: `/collect` page (`CollectPage.tsx`), link on the home page, API
  client functions, styles. Uses the existing `useRecorder` hook (auto-stop on
  silence, "Done" to stop early), shows the item's cell cropped from the scan
  plus its expected text and a play-reference button. Speaker id in
  localStorage so a refresh resumes at the first unrecorded item.
- Test suite: 108 passed; frontend lint and build clean.

Known limitations

- 547 items per speaker is long (~30–40 min). There is Skip, and progress is
  per item, so a volunteer can stop and resume, but no "pages 1–3 only" mode.
- The page was not yet exercised in a real browser session in this run (no
  microphone here); see docs/TODO-testing.md.
- Consent is a checkbox, not a signed form; add the project's consent text if
  the department requires one.
- Export covers all speakers in one CSV; there is no per-speaker export.

## 4. Harakat signal — `backend/app/services/harakat.py` (tag `ckpt-04-harakat`)

### Plan

1. The main verdict strips tashkeel (Whisper's diacritics are unreliable), so
   vowel errors (بَ vs بِ) can only be caught through per-item variants. Add a
   *secondary* score that keeps the diacritics: align the bare letters of the
   transcription with the expected text (difflib), then, on aligned letters
   where both sides carry a vowel mark, count how many marks agree. Score =
   agreeing / compared × 100; None when nothing can be compared (expected has
   no tashkeel, or Whisper wrote none). Best over expected text + variants.
2. `matching.py`'s normalisation is untouched. `evaluate_match` gains an
   optional `harakatScore` only when `TILAWAH_HARAKAT_SIGNAL_ENABLED=true`
   (default off); the verdict never changes.
3. Eval harness: `Scored` gets `harakat_score` (always computed, it is cheap);
   `--harakat-threshold T` makes a row "correct" only if the main score passes
   AND (harakat score is None or ≥ T), so "variants only" vs "variants +
   harakat" can be compared on the same transcripts. predictions.csv gains
   the column.
4. Unit tests first for the scoring function and the combined verdict rule.

### Done (tag `ckpt-04-harakat`)

- `backend/app/services/harakat.py`: `harakat_score(heard, expected_text)`
  and `score_against(heard, Expected)` (best over text + variants). 15 unit
  tests written first (`tests/test_harakat.py`): fatha/kasra/damma swaps,
  partial marks, letter errors and extra words not counted, alef forms and
  shadda, None cases, variants.
- `evaluate_match` adds `harakatScore` only when
  `TILAWAH_HARAKAT_SIGNAL_ENABLED=true`; normalisation and verdict untouched.
  `EvaluateResult.harakatScore` (API + TS type).
- Harness: `Scored.harakat_score` always filled, `--harakat-threshold`,
  `predictions.csv` column, `metrics.csv` records which mode ran. Dry run
  with `--harakat-threshold 60` re-used the cached transcripts (0 s of ASR).
- Test suite: 123 passed; frontend lint and build clean.

Known limitations

- Most items on pages 1–6 have no tashkeel in their expected text, so the
  signal is None there unless a diacritised variant exists; it mainly applies
  to pages 7–12.
- Whether Whisper's diacritics carry real information about the learner's
  vowels is exactly what the harness must measure; until then the flag stays
  off and no claim is made.

## 5. Makharij classifier scaffold — `ml/makharij/` (tag `ckpt-05-makharij`)

### Plan

1. `data.py`: labelled clips from a CSV (speaker_id, audio_path, letter) for
   the confusable pairs س/ص, د/ض, ك/ق, ه/ح; plus `publisher_dataset()` that
   pulls the publisher's single-letter clips (letter names on p01–p02,
   harakat syllables on p07–p09) with the page id as a *pseudo* speaker, only
   to exercise the pipeline.
2. `features.py`: MFCC in numpy (25 ms frames, 26 mel bands, 13 coefficients,
   mean+std pooling) with no new dependency; wav2vec2 mean-pooled embeddings
   behind an optional torch/transformers import with a clear error if absent.
3. `models.py`: baseline SVM (RBF) and a small MLP head, both scikit-learn
   pipelines with standardisation.
4. `train.py` / `evaluate.py`: speaker-wise split via `eval.split`, fit, save
   with joblib, confusion matrix per pair as CSV + PNG. Accuracy is printed
   only for real labelled data; `--smoke` runs on publisher clips and prints
   no accuracy at all (reference-audio sanity check).
5. Smoke test in `backend/tests/test_makharij_smoke.py` (MFCC + SVM on ~40
   publisher clips, end to end). scikit-learn added to requirements;
   `ml/makharij/runs/` git-ignored. README explains how to train on real
   recordings.

### Done (tag `ckpt-05-makharij`)

- `ml/makharij/`: `data.py`, `features.py` (numpy MFCC; wav2vec2 optional),
  `models.py` (SVM baseline, MLP head), `train.py`, `evaluate.py`,
  `README.md`, `requirements-optional.txt` (torch, transformers).
- `python ml/makharij/train.py --smoke` + `evaluate.py` ran end to end on 43
  publisher clips (5 pseudo speakers → 22 train / 21 test): four models,
  four confusion matrices (CSV + PNG), `results.json` with no accuracy field.
  Reference-audio sanity check; the matrices are not reported.
- 3 tests in `tests/test_makharij_smoke.py` (feature shape/determinism,
  dataset coverage, end-to-end run without an accuracy field).
- Test suite: 126 passed.

Known limitations

- No real data, no model, no number. The scaffold exists so that training
  starts the day labelled single-letter recordings exist.
- The SVM has `probability=False` (Platt scaling needs ≥ 5 clips per class);
  a confidence for the API hint will need `decision_function` or more data.
- Publisher pseudo speakers are pages of the same reciter, so the smoke split
  is not a real speaker split; the code path is exercised, nothing more.
- wav2vec2 features were not run here (torch not installed); the import path
  and error message exist, the model call is untested in this environment.

## 6. Better learner feedback (tag `ckpt-06-feedback`)

### Plan

1. `app/services/feedback.py` (pure logic, TDD): from the match result,
   the item and the optional madd result, build a `Feedback` object: heard
   vs expected (normalised), the first differing letter (expected / heard,
   via difflib opcodes), `lowConfidence`, a message, and the madd hint.
2. Confidence rule: the verdict is low-confidence when Whisper heard nothing
   ("unclear") or the score is within `TILAWAH_FEEDBACK_CONFIDENCE_MARGIN`
   (default 10 points) of the threshold. Then the message is "Try again…",
   never "wrong".
3. `/api/evaluate` attaches `feedback` whenever the verdict is not correct,
   and also on a correct verdict when the madd check (if on) is not ok.
4. Frontend: `ReciteControls` shows the message, heard vs expected, the
   differing letter and the madd hint on incorrect / moveon (single item and
   review round). `useReciteMachine.ts` is not touched: it already keeps
   `result` in state for those phases.
5. Unit tests first for the builder; one API test for the field; lint + build.

### Done (tag `ckpt-06-feedback`)

- `app/services/feedback.py`: `differing_letter`, `build_feedback`;
  `Feedback` schema + TS type; `feedback` on `EvaluateResult` when the
  verdict is not correct or the madd check failed. Setting
  `TILAWAH_FEEDBACK_CONFIDENCE_MARGIN` (default 10 points).
- 9 unit tests written first (`tests/test_feedback.py`) + 2 API tests.
- `ReciteControls.tsx` shows the message instead of "Not quite — listen
  again", then heard vs expected, the differing letters, and the madd hint;
  a low-confidence miss gets a brass (not red) bar. The 3-attempt flow and
  `useReciteMachine.ts` are unchanged.
- Test suite: 137 passed; frontend lint and build clean.

Known limitations

- The differing letter is the first difference between normalised strings,
  so for a hallucinated extra word it may point at a letter of that word
  rather than at the learner's actual slip.
- "Try again" is decided from the score margin only; audio quality (clipping,
  silence length) is not yet used, except through the madd hint.
- Not yet seen in a browser with a real microphone in this run.

## 7. Progress tracking (tag `ckpt-07-progress`)

### Plan

1. No accounts, no database: a learner id lives in the browser's
   localStorage (default "learner", editable on the home page) and is sent
   with every `/api/evaluate` call as an optional form field; `api.evaluate`
   adds it itself so `useReciteMachine.ts` is not touched.
2. `app/services/progress.py`: pure functions (TDD) `apply_attempt` (per
   item: attempts, fails, passed, last score/verdict/time; per letter:
   attempts, fails, using the feedback's expected letter or the item's first
   letter) and `summarise` (counts, weak letters sorted by fails, a "revise
   these" list of attempted-but-not-passed items, hardest first); thin JSON
   load/save in `data/progress/<learner>.json` (git-ignored).
3. `/api/progress/{learner}` GET summary, DELETE reset. The evaluate router
   records the attempt after the verdict.
4. Home page: learner id field, "Revise these" list linking to the page, weak
   letters. Lint + build; API tests with a temp progress dir.

### Done (tag `ckpt-07-progress`)

- `app/services/progress.py` (`check_id`, `empty`, `apply_attempt`,
  `summarise`, `item_letter`, `items_index`, `load`/`save`/`reset`),
  `app/routers/progress.py` (GET/DELETE `/api/progress/{learner}`),
  `learner_id` form field on `/api/evaluate` (validated, optional; nothing is
  recorded without it, so the eval harness and old clients are unaffected).
- 10 unit tests written first (`tests/test_progress.py`) + 2 API tests.
- Frontend: `learner.ts` (localStorage id, sanitised), `api.evaluate` sends
  it, home page shows counts, weak letters, "Revise these" (links to the
  page) and a reset button. `useReciteMachine.ts` untouched.
- Test suite: 151 passed; frontend lint and build clean.

Known limitations

- A learner id is a name typed on the home page, shared by anyone on that
  browser; there is no login and no privacy beyond that.
- "Revise these" links to the page, not to the item; the item is highlighted
  only by its text. Deep-linking to a region would need a route change.
- Weak letters blame the feedback's first differing letter, which for a
  hallucinated extra word may be a letter of that word.

## Finish

README updated (routes, flags, eval/ and ml/ workflows), TODO-testing.md
regrouped in queue order, final test run: **151 passed**, frontend lint and
build clean. Branch `bilal/qaida-page-mapping`, all tags pushed.

| # | feature | tag | commit | status |
| --- | --- | --- | --- | --- |
| 1 | Evaluation harness (`eval/`) | `ckpt-01-eval` | `d7eecaa` | done |
| 2 | Madd module (`backend/tajweed/madd.py`) | `ckpt-02-madd` | `b613afc` | done |
| 3 | Collection page (`/collect`) | `ckpt-03-collect` | `777324b` | done |
| 4 | Harakat signal | `ckpt-04-harakat` | `d8b5495` | done |
| 5 | Makharij classifier scaffold (`ml/makharij/`) | `ckpt-05-makharij` | `ce2229a` | done (scaffold only, no model) |
| 6 | Better learner feedback | `ckpt-06-feedback` | `3aad7dd` | done |
| 7 | Progress tracking | `ckpt-07-progress` | `c42dd9c` | done |

Nothing was reverted. No evaluation number exists yet: every metric the code
can produce waits for labelled learner recordings (feature 3 → qari → feature 1).
