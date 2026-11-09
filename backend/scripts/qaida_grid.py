"""Find the ruled grid on a Qaida page scan.

The scans are clean black-on-white tables. A horizontal rule is a pixel row
that is mostly ink across the page; a vertical rule is a pixel column that is
mostly ink from the top to the bottom of one band. Everything here returns
pixel coordinates; callers convert to the 0-1 fractions the content JSON uses.
"""

from pathlib import Path

import av
import numpy as np

INK = 128          # gray value below which a pixel counts as ink
H_RULE = 0.70      # fraction of the frame width a horizontal rule must cover
V_RULE = 0.90      # fraction of a band's height a vertical rule must cover
MERGE_GAP = 14     # rules closer than this (double lines) are one rule
MIN_BAND = 40      # bands thinner than this are the gap inside a double line


def load_gray(path: Path) -> np.ndarray:
    with av.open(str(path)) as c:
        return next(c.decode(video=0)).to_ndarray(format="gray")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True values."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def _merge(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in runs:
        if out and s - out[-1][1] < MERGE_GAP:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def frame(ink: np.ndarray) -> tuple[int, int, int, int]:
    """Inner edge of the page border: (left, top, right, bottom)."""
    cols = _merge(_runs(ink.mean(axis=0) > 0.6))
    rows = _merge(_runs(ink.mean(axis=1) > 0.6))
    return cols[0][1], rows[0][1], cols[-1][0], rows[-1][0]


def is_blank(ink: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> bool:
    """True when a cell holds no glyph (an 8 px margin keeps its rules out)."""
    return ink[y0 + 8 : y1 - 8, x0 + 8 : x1 - 8].mean() < 0.003


def bands(ink: np.ndarray, rows: list[tuple[int, int]] | None = None) -> list[dict]:
    """Horizontal bands between rules, each with the cells its vertical rules make.

    `rows` gives explicit (y0, y1) bands instead of detecting them, for pages
    whose cells contain underlines that would otherwise read as rules.
    """
    left, top, right, bottom = frame(ink)
    if rows is None:
        inner = ink[:, left:right]
        rules = _merge(_runs(inner.mean(axis=1) > H_RULE))
        rules = [(top - 1, top)] + [r for r in rules if top < r[0] and r[1] < bottom] + [(bottom, bottom + 1)]
        rows = [(y0, y1) for (_, y0), (y1, _) in zip(rules, rules[1:])]
    out = []
    for y0, y1 in rows:
        if y1 - y0 < MIN_BAND:
            continue
        body = ink[y0 + 3 : y1 - 3, left:right]
        vrules = _merge(_runs(body.mean(axis=0) > V_RULE))
        xs = [0] + [x for r in vrules for x in r] + [right - left]
        cells = [(left + a, left + b) for a, b in zip(xs[::2], xs[1::2]) if b - a > MIN_BAND]
        out.append({"y0": int(y0), "y1": int(y1), "cells": [(int(a), int(b)) for a, b in cells]})
    return out


def split_row(ink: np.ndarray, y0: int, y1: int, x0: int, x1: int, n: int,
              trim: bool = False, equal: bool = False) -> list[tuple[int, int]]:
    """Cut an unruled row into n items, left to right.

    trim:  drop the decoration at each end first (the widest blank gap in
           each half of the row separates it from the items).
    equal: items are equal width (circled letters, which often touch).
    Otherwise cuts go where the ink is thinnest: dynamic programming picks
    n-1 columns minimising ink in a 13 px window (minus a bonus for distance
    from the nearest ink, so wide gaps are cut in the middle), each item at least 35 % of
    the average width, so touching words still separate at their joins.
    """
    body = ink[y0 + 4 : y1 - 4, x0:x1]
    profile = body.sum(axis=0).astype(float)
    cols = np.flatnonzero(profile > 1)
    lo, hi = int(cols[0]), int(cols[-1]) + 1
    if trim:
        blank = profile <= 1
        mid = (lo + hi) // 2
        left = [g for g in _runs(blank) if lo < g[0] and g[1] <= mid]
        right = [g for g in _runs(blank) if g[0] >= mid and g[1] < hi]
        if left and right:
            lo = max(left, key=lambda g: g[1] - g[0])[1]
            hi = max(right, key=lambda g: g[1] - g[0])[0]
            inner = np.flatnonzero(profile[lo:hi] > 1)
            lo, hi = lo + int(inner[0]), lo + int(inner[-1]) + 1
    pad = 6
    if equal or n == 1:
        step = (hi - lo) / n
        cuts = [lo + round(step * i) for i in range(n + 1)]
        spans = list(zip(cuts, cuts[1:]))
    else:
        # thin ink is cheap; the middle of a wide blank gap is cheapest of all
        has_ink = np.flatnonzero(profile > 1)
        dist = np.abs(np.arange(len(profile))[:, None] - has_ink[None, :]).min(axis=1)
        cost = np.convolve(profile, np.ones(13), mode="same") - 4.0 * np.minimum(dist, 40)
        minw = int(0.35 * (hi - lo) / n)
        xs = list(range(lo + minw, hi - minw + 1, 2))
        # best[k][i]: cheapest way to place k+1 cuts with the last at xs[i]
        best = [[cost[x] if x - lo >= minw else np.inf for x in xs]]
        back = []
        for _ in range(n - 2):
            prev = best[-1]
            row, ptr = [], []
            for i, x in enumerate(xs):
                cands = [(prev[j] + cost[x], j) for j in range(i) if x - xs[j] >= minw]
                c, j = min(cands) if cands else (np.inf, -1)
                row.append(c)
                ptr.append(j)
            best.append(row)
            back.append(ptr)
        last = [best[-1][i] if hi - x >= minw else np.inf for i, x in enumerate(xs)]
        i = int(np.argmin(last))
        cuts = [xs[i]]
        for ptr in reversed(back):
            i = ptr[i]
            cuts.append(xs[i])
        cuts = sorted(cuts)
        edges = [lo] + cuts + [hi]
        spans = list(zip(edges, edges[1:]))
        pad = 2
    return [(x0 + max(a - pad, 0), x0 + min(b + pad, x1 - x0)) for a, b in spans]


def split_cell(ink: np.ndarray, y0: int, y1: int, x0: int, x1: int, n: int) -> list[tuple[int, int]]:
    """Split one unruled cell into n items at its n-1 widest blank column runs.

    Used for rows of words or circled letters with no rules between them.
    Returns (x0, x1) per item, right-to-left reading order is left to the caller.
    """
    if n <= 1:
        return [(x0, x1)]
    body = ink[y0 + 4 : y1 - 4, x0 + 4 : x1 - 4]
    blank = body.sum(axis=0) <= 1
    ink_cols = np.flatnonzero(~blank)
    if len(ink_cols) == 0:
        return [(x0, x1)]
    lo, hi = ink_cols[0], ink_cols[-1] + 1
    gaps = [(s, e) for s, e in _runs(blank) if s > lo and e < hi]
    gaps = sorted(sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[: n - 1])
    if len(gaps) < n - 1:  # touching items: fall back to equal widths
        step = (hi - lo) / n
        cuts = [lo + round(step * i) for i in range(n + 1)]
        spans = list(zip(cuts, cuts[1:]))
    else:
        edges = [lo] + [x for g in gaps for x in g] + [hi]
        spans = list(zip(edges[::2], edges[1::2]))
    pad = 6
    return [(x0 + 4 + max(a - pad, 0), x0 + 4 + min(b + pad, x1 - x0 - 8)) for a, b in spans]
