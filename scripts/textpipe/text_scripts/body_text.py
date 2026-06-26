"""body_text — VECTOR body text: the drawing-interior words that fall OUTSIDE every detected
cell (notes, dimensions, callouts) — i.e. what survives in the super-removed raster.

Canonical reproduction of vector_lines_poly.py's polygon_removed_text.txt:
  * outside set = region.detect_cells(page) — ALL detected cells (area+edge-connected), NOT the
    ink/containment-filtered subset;
  * words deduped on (text, x0, y0) like the original;
  * grouped by pymupdf block_no, each block carries its bbox + reading-order text.

Geometry comes from region (single source). OCR fallback for path-rendered drawings is body_ocr.py.

Public API:
  body_words(stem)   -> deduped outside-word tuples (block,line,word,x0,y0,x1,y1,txt)
  body_blocks(stem)  -> [{"bbox_pt":[x0,y0,x1,y1], "text": str}] grouped by block (reading order)
  body_text(stem)    -> flat body text stream (blocks joined)
"""
import pymupdf as fitz
from shapely.geometry import Point

from text_scripts import region as R


def _open_page(stem, page_idx=0):
    return fitz.open(R.find_pdf(stem))[page_idx]


def _dedup_words(page):
    """pymupdf words, deduped on (text, x0, y0) — exactly the original's de-dup key."""
    seen, words = set(), []
    for w in page.get_text("words"):
        k = (w[4], round(w[0], 1), round(w[1], 1))
        if k in seen:
            continue
        seen.add(k)
        words.append(w)
    return words


def body_words(stem, page_idx=0):
    """Words whose centre is in NO detected cell (canonical 'outside' set = all detect_cells)."""
    page = _open_page(stem, page_idx)
    polys = R.detect_cells(page)
    out = []
    for w in _dedup_words(page):
        wx0, wy0, wx1, wy1, txt = w[:5]
        block_no = w[5] if len(w) > 5 else 0
        line_no = w[6] if len(w) > 6 else 0
        word_no = w[7] if len(w) > 7 else 0
        pt = Point((wx0 + wx1) / 2, (wy0 + wy1) / 2)
        if not any(p.contains(pt) for p in polys):
            out.append((block_no, line_no, word_no, wx0, wy0, wx1, wy1, txt))
    return out


def body_blocks(stem, page_idx=0):
    """Group outside words by block_no -> reading-order text + bbox (== polygon_removed_text.txt)."""
    by_block = {}
    for block_no, line_no, word_no, wx0, wy0, wx1, wy1, txt in body_words(stem, page_idx):
        by_block.setdefault(block_no, []).append((line_no, word_no, wx0, wy0, wx1, wy1, txt))
    blocks = []
    for key in sorted(by_block):
        ws = sorted(by_block[key])
        xs0 = min(w[2] for w in ws); ys0 = min(w[3] for w in ws)
        xs1 = max(w[4] for w in ws); ys1 = max(w[5] for w in ws)
        blocks.append({"bbox_pt": [round(xs0, 1), round(ys0, 1), round(xs1, 1), round(ys1, 1)],
                       "text": " ".join(w[6] for w in ws)})
    return blocks


def body_text(stem):
    """Flat body text stream — blocks joined in block reading order."""
    return " ".join(b["text"] for b in body_blocks(stem))
