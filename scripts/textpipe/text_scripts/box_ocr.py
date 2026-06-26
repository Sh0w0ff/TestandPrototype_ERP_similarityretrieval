"""box_ocr — OCR fallback text per title-block/table cell, for path-rendered drawings whose
glyphs pymupdf can't see (box_text returns empty). Each kept cell is cropped to its bbox, masked
to its TRUE polygon (so an L/T-shaped cell doesn't read a neighbour's text in its notch), then
PaddleOCR'd. Cells come from region (single geometry source).

`paddle_ocr` + the crop/mask logic copied verbatim from the frozen legacy reference
(scripts/part1/ocr_box_experiment.py) — parity-guarded by tests/test_boxocr_parity.py.

Public API:
  paddle_ocr(crop_bgr)  -> (text, scores)
  box_ocr(stem)         -> list of {"bbox_pt", "text", "scores"} for kept cells
"""
import numpy as np
import cv2

from text_scripts import region as R

PAD = 4   # small AA margin; the crop is masked to the true polygon anyway

_PADDLE = None


def paddle_ocr(crop_bgr):
    global _PADDLE
    if _PADDLE is None:
        from paddleocr import PaddleOCR
        # crops are already clean, axis-aligned, upright renders — skip the heavy
        # doc-orientation / UVDoc-unwarp / textline-orientation preprocessing stack.
        _PADDLE = PaddleOCR(lang="en",
                            use_doc_orientation_classify=False,
                            use_doc_unwarping=False,
                            use_textline_orientation=False)
    if crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 8:
        return "", []
    r = _PADDLE.predict(input=crop_bgr)[0]
    texts = list(r.get("rec_texts") or [])
    scores = [round(float(s), 2) for s in (r.get("rec_scores") or [])]
    return " ".join(texts), scores


def _poly_px(p, sx, sy):
    return np.array([[int(x * sx), int(y * sy)] for x, y in p.exterior.coords], dtype=np.int32)


def box_ocr(stem, page_idx=0):
    """Per-cell OCR text for every kept cell, sorted top-to-bottom then left-to-right."""
    page, img, sx, sy = R.render(stem, page_idx)
    H, W = img.shape[:2]
    cells = sorted(R.cells_kept(page, img, sx, sy), key=lambda q: (q.bounds[1], q.bounds[0]))
    out = []
    for p in cells:
        minx, miny, maxx, maxy = p.bounds
        x0, y0 = max(int(minx * sx) - PAD, 0), max(int(miny * sy) - PAD, 0)
        x1, y1 = min(int(maxx * sx) + PAD, W), min(int(maxy * sy) + PAD, H)
        pts = _poly_px(p, sx, sy)
        crop = img[y0:y1, x0:x1].copy()
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts - [x0, y0]], 255)
        crop[mask == 0] = 255
        text, scores = paddle_ocr(crop)
        out.append({"bbox_pt": [round(minx, 1), round(miny, 1), round(maxx, 1), round(maxy, 1)],
                    "text": text, "scores": scores})
    return out
