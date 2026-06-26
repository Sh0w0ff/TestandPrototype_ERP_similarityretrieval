"""body_ocr — OCR fallback for the drawing BODY on path-rendered drawings (body_text returns
nothing because pymupdf sees no glyphs). Tiles mobile-det OCR over region.vector_removed (the
super-removed raster — title-block/BOM cells masked out, frame cropped), clusters lines into
reading groups.

paddle_lines / group_lines / tiled_lines / order_members copied verbatim from the frozen legacy
reference (scripts/part1/ocr_body_experiment.py). The raster comes from region.vector_removed,
which is pixel-identical to the legacy raster build (proven). No files are written — pure in/out.

Public API:
  body_groups(stem, grid=(2,2))  -> [{"bbox":[x0,y0,x1,y1], "text", "lines":[...]}] reading order
  body_text(stem, grid=(2,2))    -> flat body text stream (groups joined)
"""
import numpy as np
import cv2

from text_scripts import region as R

_PADDLE = None


def paddle_lines(img_bgr, max_side=3000):
    """Return list of (text, score, [x0,y0,x1,y1]) for every detected text line."""
    global _PADDLE
    if _PADDLE is None:
        from paddleocr import PaddleOCR
        _PADDLE = PaddleOCR(lang="en",
                            text_detection_model_name="PP-OCRv5_mobile_det",  # FAST det
                            text_det_limit_side_len=960, text_det_limit_type="max",
                            use_doc_orientation_classify=False,
                            use_doc_unwarping=False,
                            use_textline_orientation=False)
    H, W = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(H, W))
    proc = (cv2.resize(img_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
            if scale < 1.0 else img_bgr)
    r = _PADDLE.predict(input=proc)[0]
    texts = list(r.get("rec_texts") or [])
    scores = list(r.get("rec_scores") or [])
    boxes = r.get("rec_boxes")
    polys = r.get("rec_polys")
    inv = 1.0 / scale
    out = []
    for i, t in enumerate(texts):
        if boxes is not None and len(boxes):
            b = boxes[i]; bb = [b[0], b[1], b[2], b[3]]
        else:
            pl = polys[i]; xs = [p[0] for p in pl]; ys = [p[1] for p in pl]
            bb = [min(xs), min(ys), max(xs), max(ys)]
        bb = [int(v * inv) for v in bb]
        out.append((t, round(float(scores[i]), 2) if i < len(scores) else None, bb))
    return out


def group_lines(lines, gx_factor=1.2, gy_factor=0.8):
    """Cluster text lines into groups via union-find on margin-overlap."""
    n = len(lines)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    exp = []
    for _, _, (x0, y0, x1, y1) in lines:
        h = max(y1 - y0, 1)
        exp.append((x0 - gx_factor * h, y0 - gy_factor * h,
                    x1 + gx_factor * h, y1 + gy_factor * h))

    def overlap(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    for i in range(n):
        for j in range(i + 1, n):
            if overlap(exp[i], exp[j]):
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def tiled_lines(img, grid=(2, 2), overlap=0.06, max_side=4000):
    """Run mobile-det over an NxM grid of overlapping tiles, map to full coords, dedupe by IoU."""
    H, W = img.shape[:2]; ny, nx = grid
    tw, th = W // nx, H // ny; ox, oy = int(tw * overlap), int(th * overlap)
    alll = []
    for iy in range(ny):
        for ix in range(nx):
            x0, y0 = max(ix * tw - ox, 0), max(iy * th - oy, 0)
            x1, y1 = min((ix + 1) * tw + ox, W), min((iy + 1) * th + oy, H)
            for t, sc, bb in paddle_lines(img[y0:y1, x0:x1], max_side=max_side):
                alll.append((t, sc, [bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0]))

    def iou(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0); inter = iw * ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0
    kept = []
    for L in sorted(alll, key=lambda l: -(l[2][2]-l[2][0])*(l[2][3]-l[2][1])):
        if all(iou(L[2], K[2]) < 0.5 for K in kept):
            kept.append(L)
    return kept


def order_members(members):
    """Reading order: bucket into rows by y-band, then left-to-right within each row."""
    hs = [m[2][3] - m[2][1] for m in members]
    band = max(float(np.median(hs)) * 0.7, 1.0)
    return sorted(members, key=lambda m: (round(((m[2][1] + m[2][3]) / 2) / band), m[2][0]))


def body_groups(stem, grid=(2, 2), page_idx=0):
    """Tile-OCR the super-removed raster, cluster into reading groups (== ocr_body_experiment.run
    minus the file writes). Returns group records top-to-bottom."""
    vr, _, _ = R.vector_removed(stem, page_idx)
    lines = tiled_lines(vr, grid=grid)
    groups = group_lines(lines)
    groups.sort(key=lambda g: min(lines[i][2][1] for i in g))
    out = []
    for g in groups:
        members = order_members([lines[i] for i in g])
        xs0 = min(m[2][0] for m in members); ys0 = min(m[2][1] for m in members)
        xs1 = max(m[2][2] for m in members); ys1 = max(m[2][3] for m in members)
        out.append({"bbox": [xs0, ys0, xs1, ys1],
                    "text": " ".join(m[0] for m in members),
                    "lines": [{"text": m[0], "score": m[1], "bbox": m[2]} for m in members]})
    return out


def body_text(stem, grid=(2, 2)):
    """Flat body text stream — groups joined in reading order."""
    return " ".join(g["text"] for g in body_groups(stem, grid=grid))
