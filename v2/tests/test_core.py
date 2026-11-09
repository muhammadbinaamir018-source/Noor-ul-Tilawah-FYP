import asyncio

import numpy as np

from app import arabic, quran
from app.alignment import CLOSE, CORRECT, MISSING, PENDING, WRONG, align
from app.asr import ScriptedASR
from app.pronunciation import EXTRA_MARGIN, ctc_force_align
from app.session import RecitationSession
from app.vad import SAMPLE_RATE, EnergyDetector, Segmenter

FATIHA_2 = "الحمد لله رب العالمين"


def test_normalize_strips_marks_and_unifies_letters():
    assert arabic.normalize("الْحَمْدُ لِلَّهِ") == "الحمد لله"
    assert arabic.normalize("إِيَّاكَ") == "اياك"
    assert arabic.normalize("الصَّلَاةَ") == arabic.normalize("الصلاه")


def test_clusters_keep_marks_with_their_letter():
    assert arabic.clusters("قُلْ") == ["قُ", "لْ"]


def test_reference_strips_basmala_and_counts_words():
    ref = quran.reference(112)
    assert [w.ayah for w in ref[:4]] == [1, 1, 1, 1]
    assert ref[0].norm == "قل"
    assert len(quran.reference(1, 2, 2)) == 4


def ref_words(text):
    return arabic.words(text)


def test_align_perfect_prefix_leaves_rest_pending():
    al = align(ref_words(FATIHA_2), ref_words("الحمد لله"))
    assert [w.status for w in al.words] == [CORRECT, CORRECT, PENDING, PENDING]
    assert al.position == 2 and not al.extras


def test_align_detects_missing_word():
    al = align(ref_words(FATIHA_2), ref_words("الحمد رب العالمين"))
    assert [w.status for w in al.words] == [CORRECT, MISSING, CORRECT, CORRECT]


def test_align_detects_substitution_and_close_match():
    al = align(ref_words(FATIHA_2), ref_words("الحمد لله ملك العلمين"))
    assert al.words[2].status == WRONG and al.words[2].heard == "ملك"
    assert al.words[3].status == CLOSE


def test_align_detects_extra_word():
    al = align(ref_words(FATIHA_2), ref_words("الحمد لله لله رب العالمين"))
    assert all(w.status == CORRECT for w in al.words)
    assert [e["heard"] for e in al.extras] == ["لله"]


def test_ctc_force_align_assigns_frames_in_order():
    # Vocab: 0 blank, 1 a, 2 b. Frames clearly say: a a _ b b
    probs = np.full((5, 3), 0.05)
    for t, k in enumerate([1, 1, 0, 2, 2]):
        probs[t, k] = 0.9
    frames = ctc_force_align(np.log(probs), [1, 2], blank=0)
    assert frames == [[0, 1], [3, 4]]
    assert ctc_force_align(np.log(probs[:1]), [1, 2], blank=0) is None


def tone(seconds, amp=0.3):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), np.float32)


def test_segmenter_splits_on_pauses():
    seg = Segmenter(EnergyDetector(), gate=None)   # tones stand in for speech
    audio = np.concatenate([silence(0.5), tone(1.0), silence(1.0), tone(0.8), silence(0.2)])
    done = []
    for k in range(0, len(audio), 1600):
        done += seg.feed(audio[k:k + 1600])
    assert len(done) == 1 and 1.0 <= len(done[0]) / SAMPLE_RATE <= 1.6
    assert seg.current() is not None
    last = seg.flush()
    assert last is not None and len(last) / SAMPLE_RATE >= 0.8


def test_segmenter_ignores_silence():
    seg = Segmenter(EnergyDetector(), gate=None)   # tones stand in for speech
    assert seg.feed(silence(3)) == [] and seg.flush() is None


def test_session_streams_and_finalizes():
    async def run():
        sent = []

        async def emit(m):
            sent.append(m)

        ref = quran.reference(1, 2, 2)
        # Two utterances: first correct, second skips a word.
        asr = ScriptedASR(["الحمد لله", "العالمين"])
        s = RecitationSession(ref, asr, emit, partial_interval=100, segmenter=Segmenter(EnergyDetector(), gate=None))
        audio = np.concatenate([tone(1.0), silence(1.0), tone(0.8), silence(1.0)])
        for k in range(0, len(audio), 1600):
            await s.feed(audio[k:k + 1600])
        return await s.finish(), sent

    final, sent = asyncio.run(run())
    assert final["type"] == "final"
    assert [w["status"] for w in final["words"]] == [CORRECT, CORRECT, MISSING, CORRECT]
    assert any(m["kind"] == "word" and m["ref_index"] == 2 for m in final["feedback"])
    assert final["stats"]["latency_ms_p50"] is not None
    assert sum(m["type"] == "state" for m in sent) >= 2


def test_align_skipped_ayah_before_similar_ayah():
    # Al-Falaq with ayah 4 skipped: ayah 5 also starts with "ومن شر".
    ref = [w.norm for w in quran.reference(113)]
    ayah4 = [w.index for w in quran.reference(113) if w.ayah == 4]
    hyp = [r for k, r in enumerate(ref) if k not in ayah4]
    starts = frozenset(w.index for k, w in enumerate(quran.reference(113))
                       if k == 0 or quran.reference(113)[k - 1].ayah != w.ayah)
    # "ومن شر" opens ayahs 4 and 5: word-level cost alone cannot tell which copy was
    # skipped, the ayah-boundary preference picks the whole ayah.
    al = align(ref, hyp, starts)
    assert [i for i, w in enumerate(al.words) if w.status == MISSING] == ayah4
    assert not al.extras


def test_align_repeated_word_prefers_first_occurrence():
    # An-Nas 1-3 then ayah 4, with ayah 3 ("اله الناس") skipped.
    ref = arabic.words("قل اعوذ برب الناس ملك الناس اله الناس من شر")
    hyp = arabic.words("قل اعوذ برب الناس ملك الناس من شر")
    al = align(ref, hyp)
    assert [i for i, w in enumerate(al.words) if w.status == MISSING] == [6, 7]


def test_repeated_phrase_is_one_run_of_extras():
    # Al-Ikhlas with ayah 3 recited twice; "ولم" also opens ayah 4.
    ref = quran.reference(112)
    hyp = arabic.words("قل هو الله احد الله الصمد لم يلد ولم يولد لم يلد ولم يولد ولم يكن له كفوا احد")
    starts = frozenset(w.index for k, w in enumerate(ref) if k == 0 or ref[k - 1].ayah != w.ayah)
    al = align([w.norm for w in ref], hyp, starts)
    assert [e["heard"] for e in al.extras] == ["لم", "يلد", "ولم", "يولد"]
    assert len({e["after"] for e in al.extras}) == 1
    assert all(w.status == CORRECT for w in al.words)


class FakeScorer:
    """Letter scores empty; verification returns a fixed log-likelihood advantage."""

    def __init__(self, delta, support=10.0):
        self.delta = delta
        self.support = support
        self.verified = []
        self.supported = []

    def log_probs(self, audio):
        return np.zeros((10, 3), np.float32)

    def score_logp(self, logp, words, level="beginner"):
        return {}

    def verify(self, logp, words, i, heard):
        self.verified.append((words[i], heard))
        return self.delta

    def extra_support(self, logp, words, i):
        self.supported.append(words[i])
        return self.support


def run_session(asr_outputs, scorer):
    async def run():
        sent = []

        async def emit(m):
            sent.append(m)

        s = RecitationSession(quran.reference(1, 2, 2), ScriptedASR(asr_outputs), emit, scorer,
                              partial_interval=100, segmenter=Segmenter(EnergyDetector(), gate=None))
        audio = np.concatenate([tone(1.5), silence(1.0)])
        for k in range(0, len(audio), 1600):
            await s.feed(audio[k:k + 1600])
        return await s.finish(), sent

    return asyncio.run(run())


def test_verification_clears_asr_drift():
    # ASR spells "العالمين" as "العلمين"; the expected word fits the audio as well.
    scorer = FakeScorer(delta=-3.0)
    final, sent = run_session(["الحمد لله رب العلمين"], scorer)
    assert scorer.verified == [("العالمين", "العلمين")]
    assert [w["status"] for w in final["words"]] == [CORRECT] * 4
    assert final["words"][3]["verified"] and not final["feedback"]
    # The suspect showed as "checking", never as "close", before the verdict.
    statuses = [m["words"][3]["status"] for m in sent if m.get("words")]
    assert "checking" in statuses and CLOSE not in statuses


def test_verification_keeps_real_mistake():
    scorer = FakeScorer(delta=12.0)
    final, _ = run_session(["الحمد لله رب العلمين"], scorer)
    assert final["words"][3]["status"] == CLOSE and not final["words"][3]["verified"]
    assert any(m["ref_index"] == 3 for m in final["feedback"])


def test_numpy_ctc_matches_torch():
    torch = __import__("pytest").importorskip("torch")
    from app.pronunciation import ctc_log_likelihood

    rng = np.random.default_rng(0)
    logp = np.log(rng.dirichlet(np.ones(6), size=40)).astype(np.float32)
    for targets in ([1, 2, 3], [2, 2, 5, 1], [4]):   # includes a repeated label
        ref = -torch.nn.functional.ctc_loss(
            torch.from_numpy(logp).unsqueeze(1), torch.tensor([targets]), torch.tensor([40]),
            torch.tensor([len(targets)]), blank=0, reduction="sum")
        assert abs(ctc_log_likelihood(logp, targets, 0) - float(ref)) < 1e-3


def test_align_joins_words_the_asr_split_or_merged():
    # Al-Fatiha 3-4 as the ASR wrote it in a real test: "مالك" split into "ما لك".
    ref = arabic.words("الرحمن الرحيم مالك يوم الدين")
    al = align(ref, arabic.words("الرحمن الرحيم ما لك يوم الدين"))
    assert [w.status for w in al.words] == [CORRECT] * 5 and not al.extras
    assert al.words[2].heard == "ما لك" and (al.words[2].hyp_index, al.words[2].hyp_end) == (2, 3)
    # The reverse: two reference words heard as one.
    al = align(arabic.words("قل يا ايها الكافرون"), arabic.words("قل ياايها الكافرون"))
    assert [w.status for w in al.words] == [CORRECT] * 4 and not al.extras
    # A genuinely extra word is still reported.
    al = align(ref, arabic.words("الرحمن الرحيم مالك مالك يوم الدين"))
    assert [e["heard"] for e in al.extras] == ["مالك"]


def test_silero_ignores_household_noise():
    __import__("pytest").importorskip("onnxruntime")
    import sys
    from pathlib import Path
    from app.vad import SileroDetector
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from make_noisy_testset import events, fan

    rng = np.random.default_rng(0)
    noise = 0.05 * fan(5 * SAMPLE_RATE, rng) + events(5 * SAMPLE_RATE, rng, 0.2)
    # The loudness detector takes this noise for speech; the Silero gate discards it.
    loud = Segmenter(EnergyDetector(), gate=None)
    assert loud.feed(noise.astype(np.float32)) or loud.flush() is not None
    seg = Segmenter(EnergyDetector(), gate=SileroDetector())
    assert seg.feed(noise.astype(np.float32)) == [] and seg.current() is None and seg.flush() is None
    assert seg.rejected >= 1


def test_extra_word_from_noise_is_dropped():
    # After the ayah the ASR turned noise into "وما"; the audio does not support it.
    scorer = FakeScorer(delta=0.0, support=EXTRA_MARGIN - 10)
    final, sent = run_session(["الحمد لله رب العالمين وما"], scorer)
    assert "وما" in scorer.supported
    assert final["extras"] == [] and [d["heard"] for d in final["dropped_extras"]] == ["وما"]
    assert not final["feedback"]
    # It never appeared as an extra while being checked.
    assert all(not m["extras"] for m in sent if m.get("words"))


def test_spoken_extra_word_is_kept():
    scorer = FakeScorer(delta=0.0, support=EXTRA_MARGIN + 10)
    final, _ = run_session(["الحمد لله لله رب العالمين"], scorer)
    assert [e["heard"] for e in final["extras"]] == ["لله"] and final["extras"][0]["support"] == EXTRA_MARGIN + 10
    assert any(m["kind"] == "extra" for m in final["feedback"])


def test_level_accepts_beginner_approximation_of_dad():
    from app.pronunciation import CtcScorer
    from app.quran import RefWord

    vocab = {"<pad>": 0, "|": 1, "ض": 2, "ظ": 3, "ل": 4}
    scorer = CtcScorer(vocab, blank=0)
    word = RefWord(0, 1, 1, "ضَل", "ضل", ["ضَ", "ل"])
    # The learner's ض came out as ظ: frame 1 is ظ, frame 3 is ل, the rest blank.
    probs = np.full((5, 5), 0.01)
    for t, k in enumerate([0, 3, 0, 4, 0]):
        probs[t, k] = 0.96
    logp = np.log(probs / probs.sum(axis=1, keepdims=True))
    dad = lambda level: scorer.score_logp(logp, [word], level)[0][0]  # noqa: E731
    assert not dad("beginner").flagged and dad("beginner").level_score > 0.9
    assert dad("advanced").flagged and dad("advanced").score < 0.05


def test_extras_are_verified_when_a_later_segment_moves_them():
    # A phrase recited twice in two utterances: which copy is "extra" is only settled
    # once the second segment arrives; the extra shown must still carry its support.
    async def run():
        async def emit(m):
            pass
        s = RecitationSession(quran.reference(1, 2, 2), ScriptedASR(["الحمد لله", "الحمد لله رب العالمين"]),
                              emit, FakeScorer(delta=0.0, support=EXTRA_MARGIN + 10), partial_interval=100,
                              segmenter=Segmenter(EnergyDetector(), gate=None))
        audio = np.concatenate([tone(1.0), silence(1.0), tone(1.5), silence(1.0)])
        for k in range(0, len(audio), 1600):
            await s.feed(audio[k:k + 1600])
        return await s.finish()

    final = asyncio.run(run())
    assert [e["heard"] for e in final["extras"]] == ["الحمد", "لله"]
    assert all(e["support"] == EXTRA_MARGIN + 10 for e in final["extras"])
