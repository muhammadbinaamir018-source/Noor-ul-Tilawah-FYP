"""Makharij (articulation point) confusion classifier: scaffold.

How it works: Whisper tends to "correct" a mispronounced letter to the right
word, so س said like ص is invisible in the transcription. This package
classifies the *audio* of a single letter into one of the confusable pairs
(س/ص, د/ض, ك/ق, ه/ح): clips are cut from labelled recordings, turned into
features (MFCC statistics, or wav2vec2 embeddings when torch is installed),
and a baseline SVM or a small MLP head is trained on one set of speakers and
evaluated on speakers it has never heard. Nothing here reports accuracy until
real labelled learner recordings exist; the publisher's clips only prove the
pipeline runs.
"""
