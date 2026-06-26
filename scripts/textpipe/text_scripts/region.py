"""region — the geometric foundation: find title-block/table CELLS and build the vector-removed
BODY raster. THE single source of truth for geometry; box_text / box_ocr / body_text / body_ocr /
bom all consume what this module returns (so the cell finder is never forked).

Functions copied verbatim from the frozen legacy reference (scripts/part1/ocr_box_experiment.py +
ocr_body_experiment.py) — see tests/test_region_parity.py which asserts they still match.

Public API:
  find_pdf(stem)                         -> Path to the source PDF
  render(stem)                           -> (page, img_bgr, sx, sy)
  detect_cells(page)                     -> all cell polygons (shapely, PDF-pt space)
  cells_deg(page)                        -> non-degenerate subset (bbox < 0.7 page) — the fill set
  cells_kept(page, img, sx, sy)          -> ink+containment-filtered subset — what to read/OCR
  vector_removed(stem)                   -> (raster_bgr, sx, sy) body image with all cells masked
"""
import warnings
from pathlib import Path

import numpy as np
import cv2
import pymupdf as fitz
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union

warnings.filterwarnings("ignore")

ROOT = Path("/Users/sh0w0ff/FYP")
DPI = 300
AXIS_TOL = 0.5
MIN_LEN = 14.0
EDGE_TOL = 30.0
MAX_AREA_FRAC = 0.30
DEG_FRAC = 0.7          # cells whose bbox spans >= this fraction of the page are degenerate


def find_pdf(stem):
    for ext in (".pdf", ".PDF"):
        p = ROOT / "PDF drawings" / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def page_count(stem):
    """Number of pages in the source PDF (for drawing-level all-pages aggregation)."""
    return fitz.open(find_pdf(stem)).page_count


def render(stem, page_idx=0):
    """Open the given page (default page 1), render @ DPI, return (page, img_bgr, sx, sy)."""
    page = fitz.open(find_pdf(stem))[page_idx]
    pw, ph = page.rect.width, page.rect.height
    pix = page.get_pixmap(dpi=DPI)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
    H, W = img.shape[:2]
    return page, img, W / pw, H / ph


def detect_cells(page):
    """Return list of kept cell polygons (shapely) in PDF-point space — the
    vector_lines_poly.py algorithm: axis-aligned segs -> polygonize -> area filter
    -> keep only components connected (edge-sharing) to a page-edge-touching cell."""
    pw, ph = page.rect.width, page.rect.height
    segs = []
    for dr in page.get_drawings():
        for item in dr.get("items", []):
            if item[0] != "l":
                continue
            x1, y1 = item[1].x, item[1].y
            x2, y2 = item[2].x, item[2].y
            is_h = abs(y1 - y2) <= AXIS_TOL and abs(x1 - x2) > AXIS_TOL
            is_v = abs(x1 - x2) <= AXIS_TOL and abs(y1 - y2) > AXIS_TOL
            if not (is_h or is_v):
                continue
            length = abs(x2 - x1) if is_h else abs(y2 - y1)
            if length < MIN_LEN:
                continue
            segs.append(LineString([(x1, y1), (x2, y2)]))
    polys = list(polygonize(unary_union(segs)))
    page_area = pw * ph
    polys_pre = [p for p in polys if p.area <= MAX_AREA_FRAC * page_area]

    n = len(polys_pre)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        bi = polys_pre[i].boundary
        for j in range(i + 1, n):
            if not polys_pre[i].intersects(polys_pre[j]):
                continue
            inter = bi.intersection(polys_pre[j].boundary)
            if hasattr(inter, "length") and inter.length > 0.5:
                union(i, j)

    def touches_edge(p):
        minx, miny, maxx, maxy = p.bounds
        return (minx <= EDGE_TOL or miny <= EDGE_TOL
                or maxx >= pw - EDGE_TOL or maxy >= ph - EDGE_TOL)

    comp_has_edge = {}
    for i, p in enumerate(polys_pre):
        r = find(i)
        if touches_edge(p):
            comp_has_edge[r] = True
        comp_has_edge.setdefault(r, False)
    return [p for i, p in enumerate(polys_pre) if comp_has_edge[find(i)]]


def axis_segments(page):
    """Axis-aligned line segments (x1,y1,x2,y2) in PDF-point space — same extraction
    detect_cells uses; exposed for the border-mask/crop step."""
    segs = []
    for dr in page.get_drawings():
        for item in dr.get("items", []):
            if item[0] != "l":
                continue
            x1, y1 = item[1].x, item[1].y
            x2, y2 = item[2].x, item[2].y
            is_h = abs(y1 - y2) <= AXIS_TOL and abs(x1 - x2) > AXIS_TOL
            is_v = abs(x1 - x2) <= AXIS_TOL and abs(y1 - y2) > AXIS_TOL
            if not (is_h or is_v):
                continue
            length = abs(x2 - x1) if is_h else abs(y2 - y1)
            if length < MIN_LEN:
                continue
            segs.append((x1, y1, x2, y2))
    return segs


def border_mask_crop(img, page, sx, sy, page_noise=0.01, edge_pct=0.04, pad=4):
    """Port of vector_lines_poly.py's content-edge frame mask + tight crop, so the OCR
    vector_removed 'zooms in' on the drawing exactly like the original."""
    pw, ph = page.rect.width, page.rect.height
    segs = axis_segments(page)
    nx, ny = page_noise * pw, page_noise * ph
    xs, ys = [], []
    for x1, y1, x2, y2 in segs:
        page_edge_only = ((x1 < nx and x2 < nx) or (x1 > pw - nx and x2 > pw - nx)
                          or (y1 < ny and y2 < ny) or (y1 > ph - ny and y2 > ph - ny))
        if page_edge_only:
            continue
        xs += [x1, x2]; ys += [y1, y2]
    if not xs:
        return img
    bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
    ex, ey = edge_pct * (bx1 - bx0), edge_pct * (by1 - by0)
    for x1, y1, x2, y2 in segs:
        near = ((x1 < bx0 + ex and x2 < bx0 + ex) or (x1 > bx1 - ex and x2 > bx1 - ex)
                or (y1 < by0 + ey and y2 < by0 + ey) or (y1 > by1 - ey and y2 > by1 - ey))
        if near:
            cv2.line(img, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)),
                     (255, 255, 255), 4)
    non_white = (img < 250).any(axis=-1)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    if rows.size and cols.size:
        y0 = max(int(rows.min()) - pad, 0); y1c = min(int(rows.max()) + pad, img.shape[0])
        x0 = max(int(cols.min()) - pad, 0); x1c = min(int(cols.max()) + pad, img.shape[1])
        img = img[y0:y1c, x0:x1c]
    return img


def ink_frac(p, img, sx, sy):
    """Fraction of dark pixels inside the cell interior (inset past the border)."""
    minx, miny, maxx, maxy = p.bounds
    x0, y0 = int(minx * sx) + 4, int(miny * sy) + 4
    x1, y1 = int(maxx * sx) - 4, int(maxy * sy) - 4
    h, w = img.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float((g < 180).mean())


def filter_cells(cells, img, sx, sy):
    """Two cheap pre-OCR filters:
      (1) INK  — drop cells with ~no dark content inside (empty grid cells, noise).
      (2) CONTAINMENT — drop 'aggregate' cells that wholly contain >=2 smaller cells."""
    inked = [(p, ink_frac(p, img, sx, sy)) for p in cells]
    keep_ink = [p for p, f in inked if f >= 0.003]
    bounds = [p.bounds for p in keep_ink]
    areas = [p.area for p in keep_ink]
    cx = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in bounds]
    kept = []
    for i, bi in enumerate(bounds):
        children = sum(1 for j in range(len(bounds))
                       if j != i and areas[j] < areas[i] * 0.95
                       and bi[0] <= cx[j][0] <= bi[2] and bi[1] <= cx[j][1] <= bi[3])
        if children < 2:
            kept.append(keep_ink[i])
    return kept, len(cells), len(keep_ink), len(kept)


# ---- composed helpers everything else consumes --------------------------------

def cells_deg(page):
    """Non-degenerate cells: bbox spans < DEG_FRAC of the page. This is the FULL set
    white-filled to make the vector-removed body raster (matches vector_lines_poly)."""
    pw, ph = page.rect.width, page.rect.height
    return [p for p in detect_cells(page)
            if (p.bounds[2] - p.bounds[0]) < DEG_FRAC * pw
            and (p.bounds[3] - p.bounds[1]) < DEG_FRAC * ph]


def cells_kept(page, img, sx, sy):
    """The ink+containment-filtered cells — WHAT to read/OCR (title block, BOM, fields)."""
    kept, *_ = filter_cells(cells_deg(page), img, sx, sy)
    return kept


def vector_removed(stem, page_idx=0):
    """Build the BODY raster: page render with every non-degenerate cell white-filled +
    outer frame masked + tight-cropped. Returns (raster_bgr, sx, sy). This is what
    body_ocr tiles OCR over."""
    page, img, sx, sy = render(stem, page_idx)
    vr = img.copy()
    for p in cells_deg(page):
        pts = np.array([[int(x * sx), int(y * sy)] for x, y in p.exterior.coords], np.int32)
        cv2.fillPoly(vr, [pts], (255, 255, 255))
        cv2.polylines(vr, [pts], True, (255, 255, 255), 4)  # clear residual boundary stroke
    vr = border_mask_crop(vr, page, sx, sy)
    return vr, sx, sy
