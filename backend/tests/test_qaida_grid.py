"""Unit tests for the scan-grid utilities behind scripts/map_qaida.py.

They draw small synthetic "scans" (ink = True) so no image files are needed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from qaida_grid import bands, is_blank, split_cell, split_row  # noqa: E402


def page(h=400, w=300):
    """Blank page with a 3 px border, like the scans."""
    ink = np.zeros((h, w), dtype=bool)
    ink[:3, :] = ink[-3:, :] = True
    ink[:, :3] = ink[:, -3:] = True
    return ink


def hrule(ink, y):
    ink[y : y + 2, :] = True


def vrule(ink, x, y0, y1):
    ink[y0:y1, x : x + 2] = True


def blob(ink, x0, x1, y0, y1):
    ink[y0:y1, x0:x1] = True


def test_bands_follow_horizontal_rules():
    ink = page()
    hrule(ink, 100)
    hrule(ink, 200)
    got = bands(ink)
    assert [(b["y0"], b["y1"]) for b in got] == [(3, 100), (102, 200), (202, 397)]


def test_bands_split_cells_on_vertical_rules():
    ink = page()
    hrule(ink, 100)
    vrule(ink, 150, 3, 100)
    cells = bands(ink)[0]["cells"]
    assert len(cells) == 2
    assert cells[0][1] <= 150 and cells[1][0] >= 152


def test_bands_rows_override_ignores_internal_rules():
    """Pages whose cells contain their own underlines (rasm-ul-khat) are
    described by explicit row ranges; rules inside a row are not bands."""
    ink = page()
    hrule(ink, 100)
    hrule(ink, 150)  # underline inside the cell row 100-200
    hrule(ink, 200)
    got = bands(ink, rows=[(102, 200), (202, 397)])
    assert [(b["y0"], b["y1"]) for b in got] == [(102, 200), (202, 397)]


def test_is_blank_cell():
    ink = page()
    blob(ink, 50, 80, 50, 80)
    assert not is_blank(ink, 20, 120, 20, 120)
    assert is_blank(ink, 150, 250, 20, 120)


def test_split_row_equal_widths():
    ink = page()
    for x0 in (20, 120, 220):
        blob(ink, x0, x0 + 60, 20, 80)
    spans = split_row(ink, 10, 90, 3, 297, 3, equal=True)
    assert len(spans) == 3
    assert all(a < b for a, b in spans)
    assert spans[0][0] <= 20 and spans[-1][1] >= 280


def test_split_row_cuts_in_blank_gaps():
    ink = page()
    blob(ink, 20, 90, 20, 80)
    blob(ink, 130, 170, 20, 80)
    blob(ink, 210, 280, 20, 80)
    spans = split_row(ink, 10, 90, 3, 297, 3)
    cuts = [spans[0][1], spans[1][1]]
    assert 90 <= cuts[0] <= 130 and 170 <= cuts[1] <= 210
    assert spans[0][0] <= 20 and spans[1][0] <= 130 and spans[2][1] >= 280


def test_split_row_trim_drops_end_decorations():
    ink = page()
    blob(ink, 10, 30, 30, 70)  # decoration
    blob(ink, 90, 130, 20, 80)
    blob(ink, 170, 210, 20, 80)
    blob(ink, 270, 290, 30, 70)  # decoration
    spans = split_row(ink, 10, 90, 3, 297, 2, trim=True)
    assert len(spans) == 2
    assert spans[0][0] >= 60 and spans[-1][1] <= 240


def test_split_cell_vertical_stack_via_transpose():
    """vsplit bands use split_cell on the transposed image."""
    ink = page()
    blob(ink, 20, 280, 20, 60)
    blob(ink, 20, 280, 120, 160)
    rows = split_cell(ink.T, 3, 297, 3, 397, 2)
    assert len(rows) == 2
    assert rows[0][1] <= 120 and rows[1][0] >= 60


@pytest.mark.parametrize("n", [1, 2, 4])
def test_split_row_returns_n_spans_even_when_items_touch(n):
    ink = page()
    blob(ink, 20, 280, 20, 80)
    assert len(split_row(ink, 10, 90, 3, 297, n)) == n
