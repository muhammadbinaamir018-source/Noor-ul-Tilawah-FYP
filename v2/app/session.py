"""One streaming recitation: audio in, incremental assessment state out.

Fast path  - the open (still being spoken) segment is re-decoded every
             `partial_interval` seconds so highlights move while reciting.
Commit     - when the VAD closes a segment it is decoded once more and its
             words are appended to the committed hypothesis.
Async path - committed segments are scored letter-by-letter by the
             pronunciation model without blocking the fast path. The same pass
             verifies words the ASR heard differently: a "close"/"wrong" word is
             kept only if the ASR's version explains the audio clearly better
             than the expected word; until then it shows as "checking".
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import arabic, feedback
from .alignment import CLOSE, CORRECT, PENDING, WRONG, Alignment, WordResult, align
from .asr import ASR
from .pronunciation import DEFAULT_LEVEL, EXTRA_MARGIN, VERIFY_MARGIN, PronunciationScorer
from .quran import RefWord
from .vad import SAMPLE_RATE, Segmenter

log = logging.getLogger(__name__)

Emit = Callable[[dict], Awaitable[None]]

# One CPU inference at a time for ASR; pronunciation gets its own worker.
ASR_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
PRON_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pron")


class RecitationSession:
    def __init__(self, ref: list[RefWord], asr: ASR, emit: Emit,
                 scorer: PronunciationScorer | None = None,
                 partial_interval: float = 1.0, segmenter: Segmenter | None = None,
                 level: str = DEFAULT_LEVEL):
        self.ref = ref
        self.ref_norm = [w.norm for w in ref]
        self.ayah_starts = frozenset(w.index for k, w in enumerate(ref) if k == 0 or ref[k - 1].ayah != w.ayah)
        self.asr, self.scorer, self.emit = asr, scorer, emit
        self.partial_interval = partial_interval
        self.level = level
        self.seg = segmenter or Segmenter()

        self.committed: list[str] = []
        self.partial: list[str] = []
        self.pron: dict[int, list[dict]] = {}
        # Verification of suspect words, keyed by (ref index, heard word).
        self._checking: set[tuple[int, str]] = set()
        self.verdicts: dict[tuple[int, str], float] = {}
        # Audio support of every committed heard word, by hypothesis index. Kept for all
        # words, not just current extras: a later segment can change which copy of a
        # repeated phrase the alignment calls the extra one.
        self._checking_extras: set[int] = set()
        self.extra_support: dict[int, float] = {}
        self.alignment: Alignment = align(self.ref_norm, [], self.ayah_starts)

        self._queue: asyncio.Queue[tuple[np.ndarray, float] | None] = asyncio.Queue()
        self._asr_lock = asyncio.Lock()
        self._epoch = 0                      # bumps on every commit; stale partials are dropped
        self._last_partial = time.perf_counter()
        self._partial_task: asyncio.Task | None = None
        self._pron_tasks: set[asyncio.Task] = set()
        self._finishing = False
        self._worker = asyncio.create_task(self._commit_worker())
        self.audio_seconds = 0.0
        self.latencies: list[float] = []     # audio-received -> feedback-emitted, seconds
        self.decode_times: list[float] = []

    # ---- input ---------------------------------------------------------------

    async def feed(self, samples: np.ndarray) -> None:
        now = time.perf_counter()
        self.audio_seconds += len(samples) / SAMPLE_RATE
        for segment in self.seg.feed(samples):
            await self._queue.put((segment, now))
        if (self.seg.current() is not None and self._queue.empty()
                and (self._partial_task is None or self._partial_task.done())
                and now - self._last_partial >= self.partial_interval):
            self._last_partial = now
            self._partial_task = asyncio.create_task(self._decode_partial(self.seg.current().copy(), now))

    async def finish(self) -> dict:
        last = self.seg.flush()
        if last is not None:
            await self._queue.put((last, time.perf_counter()))
        await self._queue.put(None)
        await self._worker
        if self._partial_task:
            await asyncio.gather(self._partial_task, return_exceptions=True)
        self.partial = []
        # Word results are complete now; send them before waiting on letter scores.
        self._finishing = True
        self.words_done_at = time.perf_counter()
        await self.emit({**self.state(final=True), "type": "state"})
        if self._pron_tasks:
            await asyncio.gather(*self._pron_tasks, return_exceptions=True)
        state = self.state(final=True)
        await self.emit(state)
        return state

    # ---- decoding --------------------------------------------------------------

    async def _transcribe(self, audio: np.ndarray) -> list[str]:
        async with self._asr_lock:
            t = time.perf_counter()
            text = await asyncio.get_running_loop().run_in_executor(ASR_POOL, self.asr.transcribe, audio)
            self.decode_times.append(time.perf_counter() - t)
        return arabic.words(text)

    async def _decode_partial(self, audio: np.ndarray, received: float) -> None:
        epoch = self._epoch
        try:
            words = await self._transcribe(audio)
        except Exception:
            log.exception("partial decode failed")
            return
        if epoch != self._epoch:
            return                            # a commit landed meanwhile; this is stale
        self.partial = words
        await self._publish(received)

    async def _commit_worker(self) -> None:
        while (item := await self._queue.get()) is not None:
            audio, received = item
            try:
                words = await self._transcribe(audio)
            except Exception:
                log.exception("segment decode failed")
                continue
            start = len(self.committed)
            self.committed.extend(words)
            self._epoch += 1
            self.partial = []
            hyp_range = range(start, start + len(words))
            if self.scorer is not None and words:
                # Hold suspect words as "checking" from their first appearance.
                al = align(self.ref_norm, self.committed, self.ayah_starts)
                self._checking |= {(i, w.heard) for i, w in enumerate(al.words)
                                   if w.hyp_index in hyp_range and w.status in (CLOSE, WRONG)}
                self._checking_extras |= set(hyp_range)
            await self._publish(received)
            if self.scorer is not None and words:
                task = asyncio.create_task(self._score(audio, hyp_range))
                self._pron_tasks.add(task)
                task.add_done_callback(self._pron_tasks.discard)

    async def _score(self, audio: np.ndarray, hyp_range: range) -> None:
        al = align(self.ref_norm, self.committed, self.ayah_starts)
        spoken = [(i, w) for i, w in enumerate(al.words)
                  if w.hyp_index in hyp_range and w.status in (CORRECT, CLOSE, WRONG)]
        extras = [e for e in al.extras if e["hyp_index"] in hyp_range]
        said = [self.ref[i] for i, w in spoken if w.status in (CORRECT, CLOSE)]
        # What this audio contains, in spoken order: expected words where the learner said
        # them (the hypothesis to verify against), and the extra words as heard.
        seq = sorted([(w.hyp_index, 0, i, self.ref_norm[i]) for i, w in spoken] +
                     [(e["hyp_index"], 1, e["hyp_index"], e["heard"]) for e in extras])
        context = [text for *_, text in seq]
        pos = {(kind, key): k for k, (_, kind, key, _) in enumerate(seq)}
        suspects = [(pos[(0, i)], i, w.heard) for i, w in spoken if w.status in (CLOSE, WRONG)]
        heard = self.committed[hyp_range.start:hyp_range.stop]
        if not seq:
            self._checking_extras -= set(hyp_range)
            return

        def work():
            logp = self.scorer.log_probs(audio)
            letters = self.scorer.score_logp(logp, said, self.level) if said else {}
            deltas = {(i, heard): self.scorer.verify(logp, context, k, heard) for k, i, heard in suspects}
            # Support of each heard word within what the ASR heard in this audio.
            support = {hyp_range.start + k: self.scorer.extra_support(logp, heard, k) for k in range(len(heard))}
            return letters, deltas, support

        try:
            result, deltas, support = await asyncio.get_running_loop().run_in_executor(PRON_POOL, work)
        except Exception:
            log.exception("pronunciation scoring failed")
            result, deltas, support = {}, {}, {}
        finally:
            self._checking -= {(i, heard) for _, i, heard in suspects}
            self._checking_extras -= set(hyp_range)
        self.verdicts.update(deltas)
        self.extra_support.update(support)
        for ref_i, letters in result.items():
            self.pron[ref_i] = [s.to_dict() for s in letters]
        await self.emit({"type": "pronunciation",
                         "words": {str(k): v for k, v in self.pron.items() if k in result}})
        await self.emit({**self.state(final=self._finishing), "type": "state"})

    # ---- output --------------------------------------------------------------

    async def _publish(self, received: float) -> None:
        self.latencies.append(time.perf_counter() - received)
        await self.emit(self.state())

    def state(self, final: bool = False) -> dict:
        hyp = self.committed + self.partial
        self.alignment = al = align(self.ref_norm, hyp, self.ayah_starts)
        n_committed = len(self.committed)
        if not final and self.partial:
            # The newest partial word is often cut mid-way; judging it now flashes a
            # false "wrong". Hold its verdict until the segment is committed.
            last = len(hyp) - 1
            held = [i for i, w in enumerate(al.words) if w.hyp_end == last and w.status in (WRONG, CLOSE)]
            al = Alignment([WordResult(PENDING) if i in held else w for i, w in enumerate(al.words)],
                           [e for e in al.extras if e["hyp_index"] != last],
                           min([al.position, *held]))
        # Apply verification: cleared suspects become correct, pending ones "checking".
        verified: set[int] = set()
        checking: set[int] = set()
        adjusted = []
        for i, w in enumerate(al.words):
            if w.status in (CLOSE, WRONG) and w.hyp_index is not None and w.hyp_index < n_committed:
                key = (i, w.heard)
                if key in self._checking:
                    checking.add(i)
                    w = WordResult(PENDING, w.heard, w.hyp_index, w.similarity)
                elif key in self.verdicts and self.verdicts[key] < VERIFY_MARGIN:
                    verified.add(i)
                    w = WordResult(CORRECT, w.heard, w.hyp_index, w.similarity)
            adjusted.append(w)
        # Extra words: live (partial) ones are not shown; committed ones only once the audio
        # was checked to contain them. Unsupported ones were noise and are dropped.
        shown, dropped = [], []
        for e in al.extras:
            if e["hyp_index"] >= n_committed or e["hyp_index"] in self._checking_extras:
                continue
            sup = self.extra_support.get(e["hyp_index"])
            if self.scorer is not None and sup is not None and sup < EXTRA_MARGIN:
                dropped.append({"after": e["after"], "heard": e["heard"], "support": round(sup, 2)})
                continue
            shown.append({**e, "support": None if sup is None else round(sup, 2)})
        al = Alignment(adjusted, shown, al.position)
        words = []
        for i, w in enumerate(al.words):
            status = "checking" if i in checking else w.status
            if final and status == PENDING:
                status = "not_reached"
            words.append({
                "index": i, "status": status, "heard": w.heard, "similarity": w.similarity,
                "tentative": w.hyp_index is not None and w.hyp_index >= n_committed,
                "verified": i in verified,
                "letters": self.pron.get(i, []),
            })
        return {
            "type": "final" if final else "state",
            "words": words,
            "extras": [{"after": e["after"], "heard": e["heard"], "tentative": False,
                        "support": e["support"]} for e in al.extras],
            "dropped_extras": dropped,
            "position": al.position,
            "transcript": {"committed": " ".join(self.committed), "partial": " ".join(self.partial)},
            "feedback": feedback.messages(self.ref, al, self.pron, final),
            "stats": self.stats(),
        }

    def stats(self) -> dict:
        def ms(xs: list[float], q: float) -> float | None:
            return round(float(np.quantile(xs, q)) * 1000) if xs else None
        return {
            "audio_seconds": round(self.audio_seconds, 2),
            "latency_ms_p50": ms(self.latencies, 0.5),
            "latency_ms_p90": ms(self.latencies, 0.9),
            "decode_ms_p50": ms(self.decode_times, 0.5),
        }
