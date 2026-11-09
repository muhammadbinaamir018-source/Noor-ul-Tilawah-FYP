"""Evaluation harness for the Qaida tutor (FYP evaluation chapter).

How it works: a labels CSV says, for each learner recording, which Qaida item
it was and whether the recitation was correct. run_eval.py pushes every
recording through the backend's own pipeline (Whisper transcription, then
rapidfuzz matching against the item's expected text), compares the verdict
with the label, and reports accuracy, False Accept Rate, False Reject Rate,
per-item accuracy, a confusion matrix and a threshold sweep.
"""
