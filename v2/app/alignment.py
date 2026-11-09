"""Word-level alignment of a (partial) recitation against the reference ayahs.

Semi-global edit distance: every hypothesis word must be explained, but the
reference may end early (words the learner has not reached yet are free).
Matching uses normalised-word similarity, so an ASR spelling variant or a small
pronunciation slip is a cheap "close" match rather than a hard substitution.
"""

from dataclasses import dataclass, field

from .arabic import similarity

CORRECT, CLOSE, WRONG, MISSING, PENDING = "correct", "close", "wrong", "missing", "pending"

CLOSE_THRESHOLD = 0.6     # similarity at or above this is a near-miss, not a different word
DEL_COST = 0.9            # skipped reference word; below INS_COST so a skipped word
INS_COST = 1.0            # beats "extra word + stop" when the next word matches
SUB_WEIGHT = 1.2          # > DEL_COST so "skip + match next" beats "substitute"
TAIL_COST = 0.5           # per reference word not reached yet; < DEL_COST so stopping early
                          # stays cheaper than skipping, but a skipped ayah followed by the
                          # next one is no longer explained as "wrong word + extras + stop"
GAP_OPEN = 0.01           # tiny: only decides between otherwise equal explanations
JOIN_COST = 0.1           # one reference word vs two heard words (or the reverse): the ASR
                          # often splits or joins words ("مالك" heard as "ما لك")
JOIN_MIN_SIM = 0.85       # ...only when the joined text nearly matches: a join must not
                          # absorb a genuinely skipped word ("الحمد لله" vs "الحمد" is 0.62)
EPS = 1e-9


@dataclass
class WordResult:
    status: str = PENDING
    heard: str | None = None       # hypothesis word aligned to this reference word
    hyp_index: int | None = None
    similarity: float | None = None
    hyp_end: int | None = None     # last hypothesis word, when two heard words form this word

    def __post_init__(self):
        if self.hyp_end is None:
            self.hyp_end = self.hyp_index


@dataclass
class Alignment:
    words: list[WordResult]
    extras: list[dict] = field(default_factory=list)   # {"after": ref index or -1, "heard": str}
    position: int = 0                                  # next expected reference word

    def errors(self) -> list[dict]:
        out = [{"type": w.status, "ref_index": i, "heard": w.heard}
               for i, w in enumerate(self.words) if w.status in (CLOSE, WRONG, MISSING)]
        out += [{"type": "extra", "ref_index": e["after"], "heard": e["heard"]} for e in self.extras]
        return sorted(out, key=lambda e: e["ref_index"])


def _sub_cost(sim: float) -> float:
    return 0.0 if sim >= 1.0 else SUB_WEIGHT * (1.0 - sim)


def align(ref: list[str], hyp: list[str], ayah_starts: frozenset[int] = frozenset()) -> Alignment:
    """Affine-gap (Gotoh) alignment with three states: M (word aligned), D (reference word
    skipped), I (extra word). Opening a gap costs GAP_OPEN extra, so among equally cheap
    explanations a skipped ayah or a repeated phrase is reported as one contiguous run.
    A run that starts on an ayah boundary (`ayah_starts`: reference indices of each
    ayah's first word) opens at half cost: learners skip and repeat whole ayahs.
    Remaining ties prefer gaps later in the reference: the first occurrence of a
    repeated word is the one matched, the later copy is reported skipped or extra.
    The M state is also entered by a join: two heard words matched as one reference word
    (the ASR split it) or one heard word as two reference words (the ASR merged them)."""
    n, m = len(ref), len(hyp)
    sims = [[similarity(r, h) for h in hyp] for r in ref]
    inf = float("inf")
    M, D, I = 0, 1, 2
    cost = [[[inf] * (m + 1) for _ in range(n + 1)] for _ in range(3)]
    back = [[[-1] * (m + 1) for _ in range(n + 1)] for _ in range(3)]
    how = [[(1, 1)] * (m + 1) for _ in range(n + 1)]   # M's step: (ref words, hyp words) consumed
    cost[M][0][0] = 0.0

    def best(i: int, j: int, prefer: tuple[int, ...] = (I, D, M)) -> tuple[float, int]:
        # Lowest cost state at a cell; ties go to the earlier state in `prefer`.
        return min(((cost[st][i][j], k, st) for k, st in enumerate(prefer)))[::2]

    def relax(st: int, i: int, j: int, v: float, prev: int, strict: bool, step=(1, 1)) -> None:
        if v < cost[st][i][j] - EPS or (not strict and v <= cost[st][i][j] + EPS):
            cost[st][i][j], back[st][i][j] = v, prev
            if st == M:
                how[i][j] = step

    for i in range(n + 1):
        for j in range(m + 1):
            c, st = best(i, j)
            if c == inf:
                continue
            if i < n and j < m:
                relax(M, i + 1, j + 1, c + _sub_cost(sims[i][j]), st, strict=True)
            if i < n and j + 1 < m:     # two heard words form one reference word
                sim = similarity(ref[i], hyp[j] + hyp[j + 1])
                if sim >= JOIN_MIN_SIM:
                    relax(M, i + 1, j + 2, c + JOIN_COST + _sub_cost(sim), st, strict=True, step=(1, 2))
            if i + 1 < n and j < m:     # one heard word covers two reference words
                sim = similarity(ref[i] + ref[i + 1], hyp[j])
                if sim >= JOIN_MIN_SIM:
                    relax(M, i + 2, j + 1, c + JOIN_COST + _sub_cost(sim), st, strict=True, step=(2, 1))
            # A gap opened here starts at ref[i] (skip) or sits just before it (extra).
            gap_open = GAP_OPEN / 2 if i in ayah_starts or i == n else GAP_OPEN
            if i < n:
                v, prev = min((cost[D][i][j], 0, D), (cost[M][i][j] + gap_open, 1, M),
                              (cost[I][i][j] + gap_open, 2, I))[::2]
                relax(D, i + 1, j, v + DEL_COST, prev, strict=False)
            if j < m:
                v, prev = min((cost[I][i][j], 0, I), (cost[M][i][j] + gap_open, 1, M),
                              (cost[D][i][j] + gap_open, 2, D))[::2]
                relax(I, i, j + 1, v + INS_COST, prev, strict=False)

    # The reference may end early (not recited yet); ties prefer the shortest prefix.
    end = min(range(n + 1), key=lambda i: (round(best(i, m)[0] + TAIL_COST * (n - i), 9), i))
    result = Alignment(words=[WordResult() for _ in ref], position=end)

    i, j = end, m
    st = best(i, j)[1]
    extras: list[dict] = []
    while i > 0 or j > 0:
        prev = back[st][i][j]
        if st == M:
            dr, dh = how[i][j]
            i, j = i - dr, j - dh
            said = " ".join(hyp[j:j + dh])
            sim = sims[i][j] if (dr, dh) == (1, 1) else similarity("".join(ref[i:i + dr]), "".join(hyp[j:j + dh]))
            status = CORRECT if sim >= 1.0 else CLOSE if sim >= CLOSE_THRESHOLD else WRONG
            for k in range(dr):
                result.words[i + k] = WordResult(status, said, j, round(sim, 3), j + dh - 1)
        elif st == D:
            i -= 1
            result.words[i] = WordResult(MISSING)
        else:
            j -= 1
            extras.append({"after": i - 1, "heard": hyp[j], "hyp_index": j})
        st = prev
    result.extras = extras[::-1]
    return result
