"""Tajweed checks that look at the audio itself, not at the transcription.

Whisper normalises what it hears towards valid text, so vowel length (madd)
and articulation points (makharij) cannot be judged from its output. These
modules measure the recording directly and return a result object the API
attaches next to the text verdict.
"""
