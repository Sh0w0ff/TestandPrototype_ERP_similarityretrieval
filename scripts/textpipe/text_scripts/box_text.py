"""box_text — VECTOR text per title-block/table cell (the primary box-text source).

For drawings whose text is real vector glyphs (pymupdf sees them), each kept cell's text is the
pymupdf words whose centre falls inside that cell, in reading order. Cells come from region (the
single geometry source); the OCR fallback for path-rendered drawings is box_ocr.py.

`pymupdf_words_in` copied verbatim from the frozen legacy reference
(scripts/part1/ocr_box_experiment.py) — parity-guarded by tests/test_boxtext_parity.py.

Public API:
  words_in_cell(page, poly)  -> str   (pymupdf words inside one cell, reading order)
  box_text(stem)            -> list of {"bbox_pt": [x0,y0,x1,y1], "text": str} for kept cells
"""
from shapely.geometry import Point

from text_scripts import region as R


def words_in_cell(page, poly):
    """pymupdf words whose centre falls inside the cell, in reading order."""
    hits = []
    for w in page.get_text("words"):
        wx0, wy0, wx1, wy1, txt = w[:5]
        if poly.contains(Point((wx0 + wx1) / 2, (wy0 + wy1) / 2)):
            hits.append((w[5], w[6], w[7], txt))
    return " ".join(t for _, _, _, t in sorted(hits))


def box_text(stem, page_idx=0):
    """Per-cell vector text for every kept cell, sorted top-to-bottom then left-to-right."""
    page, img, sx, sy = R.render(stem, page_idx)
    cells = sorted(R.cells_kept(page, img, sx, sy), key=lambda q: (q.bounds[1], q.bounds[0]))
    out = []
    for p in cells:
        minx, miny, maxx, maxy = p.bounds
        out.append({"bbox_pt": [round(minx, 1), round(miny, 1), round(maxx, 1), round(maxy, 1)],
                    "text": words_in_cell(page, p)})
    return out
