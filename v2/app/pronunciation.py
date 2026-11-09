"""Letter-level pronunciation scoring with a CTC acoustic model.

The reference letters of the words the learner said are force-aligned to the
CTC posteriors (Viterbi over the blank-augmented target sequence). Each letter
then gets a Goodness-of-Pronunciation style score: at the letter's best frame,
its posterior renormalised over letter tokens only, i.e. P(expected letter) /
sum P(any letter). Blank and tashkeel tokens are left out of the competition:
CTC output is peaky and mostly blank, and a vowel mark is not a wrong letter.
1.0 = the model is sure it heard the expected letter; near 0 = another letter
won. Letters that are silent in connected recitation are not scored.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np

from .arabic import strip_diacritics
from .quran import RefWord

log = logging.getLogger(__name__)

# Sounds the FYP targets first: letters Arabic learners commonly substitute.
FOCUS_LETTERS = set("عحخغقضظذثصطهء")
SHADDA = chr(0x0651)
# A "close"/"wrong" word is kept only if the ASR's alternative explains the audio at
# least this much better (log-likelihood, nats) than the expected word. Calibrated on
# 2026-09-23: single-letter mistakes in studio audio median +12.5 (92.6% above 2..6);
# ASR drift on a real, correct learner recitation (Al-Kawthar) at most +1.4.
VERIFY_MARGIN = 4.0
# An "extra" word is kept only if including it improves the audio's CTC log-likelihood by
# at least this much: words the ASR produced from noise are not supported by the audio.
# Calibrated on 2026-09-24 with tools/calibrate_extras.py: 27 really repeated words in the
# studio set had support 27.5..105.9; 14 words the ASR made from synthetic household noise
# had -67.0..8.2. 15 sits in the gap.
EXTRA_MARGIN = 15.0
FALLBACK = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"}

MAKHRAJ_HINTS = {
    "ع": "from the middle of the throat, voiced - not a hamza or alif",
    "ح": "from the middle of the throat, a breathy voiceless sound - not ه",
    "خ": "from the upper throat, near the mouth, with friction",
    "غ": "from the upper throat, voiced, like a soft gargle",
    "ق": "back of the tongue against the soft palate, further back than ك",
    "ض": "side of the tongue against the upper molars, heavy",
    "ظ": "tip of the tongue at the edge of the upper front teeth, heavy",
    "ذ": "tip of the tongue at the edge of the upper front teeth, light and voiced",
    "ث": "tip of the tongue at the edge of the upper front teeth, voiceless",
    "ص": "a heavy س with the back of the tongue raised",
    "ط": "tip of the tongue against the gum ridge, heavy and strong",
    "ه": "from the deepest part of the throat, a light breath",
    "ء": "a clean stop from the deepest part of the throat",
}


# Learner levels: approximations of hard letters accepted at each level. A beginner's
# ض is usually ظ/ز/د-like (e.g. Urdu speakers) and is acceptable at that stage; only
# advanced (tajweed) level asks for the exact sound.
LEVELS = {
    "beginner": {"ض": "ظدزذ", "ظ": "زذ"},
    "intermediate": {"ض": "ظ"},
    "advanced": {},
}
DEFAULT_LEVEL = "beginner"


@dataclass
class LetterScore:
    cluster: int      # index into RefWord.clusters
    letter: str
    score: float      # strict: P(expected letter) among letters, at its best frame
    flagged: bool
    level_score: float | None = None   # P(expected or an approximation accepted at this level)

    def to_dict(self) -> dict:
        d = {"cluster": self.cluster, "letter": self.letter,
             "score": round(self.score, 3), "flagged": self.flagged}
        if self.level_score is not None:
            d["level_score"] = round(self.level_score, 3)
        return d


def silent_clusters(clusters: list[str], first_in_segment: bool) -> set[int]:
    """Indices of letters not pronounced in connected recitation.

    * word-initial bare alef (hamzat al-wasl) unless the word starts the utterance
    * the lam of al- before a sun letter (written: bare lam, next letter has shadda)
    """
    silent: set[int] = set()
    if clusters and clusters[0] == "ا" and not first_in_segment:
        silent.add(0)
    if len(clusters) > 2 and clusters[0] == "ا" and clusters[1] == "ل" and SHADDA in clusters[2]:
        silent.add(1)
    return silent


def ctc_force_align(logp: np.ndarray, targets: list[int], blank: int) -> list[list[int]] | None:
    """Viterbi CTC alignment: the frames assigned to each target token, or None if infeasible."""
    T, L = logp.shape[0], len(targets)
    S = 2 * L + 1
    ext = np.full(S, blank)
    ext[1::2] = targets
    emit = logp[:, ext]                                  # T x S
    neg = -np.inf
    dp = np.full((T, S), neg)
    bp = np.zeros((T, S), dtype=np.int8)                 # 0 stay, 1 from s-1, 2 from s-2
    dp[0, 0] = emit[0, 0]
    if S > 1:
        dp[0, 1] = emit[0, 1]
    # s-2 skip allowed into a label that differs from the label two back.
    skip_ok = np.zeros(S, bool)
    skip_ok[3::2] = ext[3::2] != ext[1:-2:2]
    for t in range(1, T):
        prev = dp[t - 1]
        stay = prev
        step = np.concatenate([[neg], prev[:-1]])
        skip = np.where(skip_ok, np.concatenate([[neg, neg], prev[:-2]]), neg)
        cand = np.stack([stay, step, skip])
        bp[t] = np.argmax(cand, axis=0)
        dp[t] = cand[bp[t], np.arange(S)] + emit[t]
    s = S - 1 if S == 1 or dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    if not np.isfinite(dp[-1, s]):
        return None
    frames: list[list[int]] = [[] for _ in range(L)]
    for t in range(T - 1, -1, -1):
        if s % 2 == 1:
            frames[s // 2].append(t)
        s -= int(bp[t, s]) if t > 0 else 0
    return [f[::-1] for f in frames]


def ctc_log_likelihood(logp: np.ndarray, targets: list[int], blank: int) -> float:
    """log P(targets | logp) summed over all CTC alignments (forward algorithm, log space)."""
    T, L = logp.shape[0], len(targets)
    if L == 0:
        return float(np.sum(logp[:, blank]))    # no words: every frame is blank
    S = 2 * L + 1
    ext = np.full(S, blank)
    ext[1::2] = targets
    emit = logp[:, ext]
    neg = -np.inf
    skip_ok = np.zeros(S, bool)
    skip_ok[3::2] = ext[3::2] != ext[1:-2:2]
    alpha = np.full(S, neg)
    alpha[0] = emit[0, 0]
    alpha[1] = emit[0, 1]
    for t in range(1, T):
        step = np.concatenate([[neg], alpha[:-1]])
        skip = np.where(skip_ok, np.concatenate([[neg, neg], alpha[:-2]]), neg)
        alpha = np.logaddexp(np.logaddexp(alpha, step), skip) + emit[t]
    return float(np.logaddexp(alpha[-1], alpha[-2]))


def letter_token_ids(vocab: dict[str, int]) -> np.ndarray:
    """Vocabulary ids of Arabic base letters (hamza..ya, alef wasla), excluding tatweel."""
    return np.array(sorted(i for t, i in vocab.items()
                           if len(t) == 1 and (0x0621 <= ord(t) <= 0x064A or ord(t) == 0x0671)
                           and t != chr(0x0640)))


class CtcScorer:
    """Letter scores and word verification from CTC log-posteriors.

    Pure numpy: subclasses only supply `log_probs(audio)`, either by running the
    acoustic model in-process (PronunciationScorer) or by asking a model service
    (app.remote.RemotePronunciation), so the app server needs no torch.
    """

    def __init__(self, vocab: dict[str, int], blank: int, threshold: float = 0.35):
        self.vocab = vocab
        self.blank = blank
        self.threshold = threshold
        self.sep = vocab.get("|")
        self.letter_ids = letter_token_ids(vocab)

    def log_probs(self, audio: np.ndarray) -> np.ndarray:
        """T x vocab CTC log-posteriors for 16 kHz audio."""
        raise NotImplementedError

    def _token(self, cluster: str) -> tuple[str, int | None]:
        base = strip_diacritics(cluster)[:1] or cluster[:1]
        if base in self.vocab:
            return base, self.vocab[base]
        alt = FALLBACK.get(base)
        return base, self.vocab.get(alt) if alt else None

    def score(self, audio: np.ndarray, words: list[RefWord], level: str = DEFAULT_LEVEL) -> dict[int, list[LetterScore]]:
        return self.score_logp(self.log_probs(audio), words, level) if words else {}

    def score_logp(self, logp: np.ndarray, words: list[RefWord],
                   level: str = DEFAULT_LEVEL) -> dict[int, list[LetterScore]]:
        """Score reference letters against precomputed CTC log-posteriors (T x vocab).
        A letter is flagged only if neither it nor an approximation accepted at `level`
        (see LEVELS) was heard."""
        accepted = LEVELS.get(level, LEVELS[DEFAULT_LEVEL])
        targets: list[int] = []
        slots: list[tuple[int, int, str] | None] = []      # (ref index, cluster, letter) per target
        for k, w in enumerate(words):
            if k and self.sep is not None:
                targets.append(self.sep)
                slots.append(None)
            silent = silent_clusters(w.clusters, first_in_segment=k == 0)
            for ci, cl in enumerate(w.clusters):
                if ci in silent:
                    continue
                letter, tok = self._token(cl)
                if tok is not None:
                    targets.append(tok)
                    slots.append((w.index, ci, letter))
        frames = ctc_force_align(logp, targets, self.blank) if targets else None
        if frames is None:
            log.warning("could not align %d letters to %d frames", len(targets), logp.shape[0])
            return {}
        lp = logp[:, self.letter_ids]
        norm = lp.max(axis=1) + np.log(np.exp(lp - lp.max(axis=1, keepdims=True)).sum(axis=1))
        out: dict[int, list[LetterScore]] = {w.index: [] for w in words}
        for slot, tok, fr in zip(slots, targets, frames):
            if slot is None or not fr:
                continue
            ref_i, ci, letter = slot
            sc = float(np.exp(np.max(logp[fr, tok] - norm[fr])))
            alts = [self.vocab[a] for a in accepted.get(letter, "") if a in self.vocab]
            if alts:
                both = np.logaddexp.reduce(logp[np.ix_(fr, [tok, *alts])], axis=1)
                lvl = float(np.exp(np.max(both - norm[fr])))
            else:
                lvl = None
            judged = sc if lvl is None else lvl
            out[ref_i].append(LetterScore(ci, letter, sc, letter in FOCUS_LETTERS and judged < self.threshold, lvl))
        return out

    def log_likelihood(self, logp: np.ndarray, words: list[str]) -> float:
        """log P(audio | these normalised words) under the CTC model, summed over all alignments."""
        targets: list[int] = []
        for k, w in enumerate(words):
            if k and self.sep is not None:
                targets.append(self.sep)
            targets += [self.vocab[c] for c in w if c in self.vocab]
        return ctc_log_likelihood(logp, targets, self.blank)

    def verify(self, logp: np.ndarray, words: list[str], i: int, heard: str) -> float:
        """How much better the ASR's word explains the audio than the expected one.

        `words` are the expected (normalised) words spoken in this audio; word `i` is
        the suspect, `heard` what the ASR wrote for it. Returns
        LL(audio | heard in place of word i) - LL(audio | expected words).
        Near or below 0: the expected word fits the audio as well, so the mismatch
        was the ASR's, not the learner's.
        """
        alt = words[:i] + [heard] + words[i + 1:]
        return self.log_likelihood(logp, alt) - self.log_likelihood(logp, words)

    def extra_support(self, logp: np.ndarray, words: list[str], i: int) -> float:
        """How much the audio supports word `i` of the heard sequence `words`:
        LL(audio | words) - LL(audio | words without word i). A word the ASR produced
        from noise has little or negative support; a really spoken word has a lot."""
        return self.log_likelihood(logp, words) - self.log_likelihood(logp, words[:i] + words[i + 1:])


class PronunciationScorer(CtcScorer):
    """CtcScorer backed by a wav2vec2 CTC model running in this process."""

    def __init__(self, model_id: str, device: str = "cpu", threshold: float = 0.35, int8: bool = False):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        from .asr import CPU_SHARE

        t = time.perf_counter()
        if device == "cpu":
            torch.set_num_threads(CPU_SHARE)
        self.torch = torch
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
        if int8 and device == "cpu":
            # Dynamic int8 on the Linear layers: ~1.75x faster on CPU, 98.6% of frames decode
            # identically; letter-probe recall 99.2% vs 100% fp32 (misses some أ-for-ع).
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        super().__init__(self.processor.tokenizer.get_vocab(), self.processor.tokenizer.pad_token_id, threshold)
        log.info("loaded pronunciation model %s on %s in %.1fs", model_id, device, time.perf_counter() - t)

    def log_probs(self, audio: np.ndarray) -> np.ndarray:
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        with self.torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
        return self.torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
