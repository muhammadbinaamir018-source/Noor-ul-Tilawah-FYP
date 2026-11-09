# Testing still to do

Running list of what needs manual or real-data testing before the final
evaluation. One line per item, grouped by feature in queue order. Heavy
testing is deferred to the end on purpose; this is the queue for it.

## 0. Page mapping

- Hover through pages 2–12 in `/editor` once and confirm each box plays the right sound (order was checked by transcript, box edges by overlay).
- Recite a few letters, syllables and words on pages 7–12 as a learner; harvest variants into the page JSON from `/logs` where correct recitations fail.

## 1. Evaluation harness

- Run `eval/run_eval.py` on real learner recordings with labels (none exist yet; `--dry-run` only uses publisher clips = reference-audio sanity check).
- Check FAR/FRR on at least two speakers that were held out by `eval/split.py`.
- Decide error_type categories with the advisors before labelling (madd / makhraj / harakat / other).

## 2. Madd module

- Record a few learners holding بَا / دِيْنِيْ / نُوْحُ deliberately too short, right, and too long; check `tajweed.madd.analyse` verdicts and re-tune the count bands and default harakah.
- Check the envelope on browser webm/opus recordings (tests use wav); confirm the −30 dB voiced threshold works with laptop-mic noise.
- Try a word with two long vowels (e.g. قَالُوْا) and decide whether "longest segment" is acceptable or per-vowel alignment is needed.
- Turn `TILAWAH_MADD_ENABLED=true` on in a dev session and watch `/logs` for false "too-short" hints on correct recitations.

## 3. Collection page

- Run through `/collect` in Chrome with a real microphone: consent → record → listen back → re-record → skip → finish; check `data/recordings/S01/` and `metadata.csv`.
- Refresh mid-way and confirm it resumes at the first unrecorded item.
- Check the cell preview crops correctly for a wide phrase (p06) and a single letter (p01).
- Open the export in Excel/LibreOffice and confirm Arabic ids and relative paths survive; fill two labels and run `eval/run_eval.py` on it.
- Collect at least 3 speakers before tuning anything (speaker-wise split needs ≥ 2 on each side).

## 4. Harakat signal

- On labelled recordings, run the harness twice (`--name variants-only` and `--harakat-threshold 60 --name with-harakat`) and compare FAR/FRR; report both.
- Inspect `predictions.csv` for rows where harakat_score is low but the label is correct: is Whisper's diacritic wrong, or the learner's vowel?
- Pick the harakat threshold from a sweep on the train speakers only, then report on the test speakers.

## 5. Makharij classifier

- Get single-letter learner clips labelled with the letter *produced* (qari), ≥ 3 speakers, both letters of each pair per speaker.
- Run `train.py` / `evaluate.py` with MFCC+SVM, MFCC+MLP, then `pip install torch transformers` and wav2vec2 features; compare per-pair confusion matrices on held-out speakers.
- Confirm the wav2vec2 path actually runs (not executed in this environment) and pick an Arabic-tuned checkpoint if the base model is weak.
- Decide the confidence source for the API hint (`decision_function` margin vs probability once data allows `probability=True`).

## 6. Learner feedback

- Recite wrong letters on purpose on p01 and p07 and check the message names the right pair of letters; check the bar turns brass (not red) when the score is near the threshold.
- With `TILAWAH_MADD_ENABLED=true`, hold a madd too short and confirm the hint appears on a correct text verdict.
- Check the feedback block does not overflow the bottom bar on a narrow window (phrases on p06).
- Tune `TILAWAH_FEEDBACK_CONFIDENCE_MARGIN` from the harness's sweep: it should cover the band where FAR and FRR cross.

## 7. Progress tracking

- Practise a page as two different learner names and confirm each sees only their own counts and revise list.
- Fail an item three times, pass it later, and confirm it leaves "Revise these" while the letter stays in weak letters.
- Confirm `data/progress/` is never committed and that a typo'd learner name (spaces, slashes) is sanitised, not rejected.

## Whole app

- Full browser session with a real microphone through every route (`/`, `/lesson/p07`, `/collect`, `/logs`) after the frontend build; none of the UI in features 3, 6 and 7 was exercised in a browser during the queue run.
- Run the separate adversarial-tester session (CLAUDE.md) over features 1–7 and fix what it finds.
