"""Vector-rendered polygon_removed_super.png.

Faithful port of vector_lines_poly.py's super step, changing ONLY the rendering
medium: geometry is drawn from the PDF's vector paths (get_drawings replayed via
the pymupdf Shape API) instead of rasterizing page1.png and painting white boxes
over text. The damaging part of the old method was the per-word white-box wipe,
which erased any drawing line a dimension number / balloon overlapped. Here text
is removed glyph-tight at the vector level, so geometry is never eaten.

Everything else mirrors the original:
  - polygon detection (axis-aligned segs -> polygonize -> keep edge-connected,
    <30% area) identifies the title-block / frame cells,
  - those polygons are WHITE-FILLED on the rendered raster (area fill is
    geometry-safe; the old path-suppression-by-polygon attempt ate view slots),
  - content-edge lines are masked,
  - crop to non-white pixels -> zoom to the views.

NO lineweight filtering (ABB-only; would skew ABB vs KC).

Output: polygon_removed_super_vec.png  (comparison name; doesn't overwrite the
existing super). --debug draws suppressed text paths in red and skips the
polygon white-fill so text removal can be inspected in isolation.
"""
import sys
from pathlib import Path

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union

ARGV = [a for a in sys.argv[1:] if not a.startswith("--")]
DEBUG = "--debug" in sys.argv
STEM = ARGV[0] if ARGV else "3AUA0000038918"

ROOT = Path(__file__).resolve().parents[2]          # repo root (portable: works wherever the repo lives)
sys.path.insert(0, str(ROOT / "scripts")); import paths
PDF = paths.PDF_DIR / f"{STEM}.pdf"
if not PDF.exists():
    PDF = paths.PDF_DIR / f"{STEM}.PDF"
OUT_DIR = paths.visual_dir(STEM)
REF_PNG = OUT_DIR / "page1.png"
OUT = OUT_DIR / ("super_vec_debug.png" if DEBUG else "polygon_removed_super_vec.png")

# --- constants (mirror vector_lines_poly.py) -------------------------------
AXIS_TOL = 0.5
MIN_LEN = 14.0
EDGE_TOL = 30.0
MAX_AREA_FRAC = 0.30
PAGE_NOISE_PCT = 0.01
EDGE_PCT = 0.04
WORD_PAD = 1.0        # pt pad around each suppressed word bbox
SUPPRESS_FRAC = 0.60  # drop a path only if >60% of its (thickness-floored) bbox
                      # sits inside a word box
LINE_FLOOR = 1.5      # pt: minimum bbox thickness for the containment test, so a
                      # long thin view edge can't be "mostly inside" a small word
                      # box (that bug ate the 38918 square-view right edge).

doc = fitz.open(PDF)
page = doc[0]
pw, ph = page.rect.width, page.rect.height
page_area = pw * ph
drawings = page.get_drawings()

# --- 1. Title-block / frame polygons ---------------------------------------
segs = []
for dr in drawings:
    for item in dr.get("items", []):
        if item[0] != "l":
            continue
        x1, y1 = item[1].x, item[1].y
        x2, y2 = item[2].x, item[2].y
        is_h = abs(y1 - y2) <= AXIS_TOL and abs(x1 - x2) > AXIS_TOL
        is_v = abs(x1 - x2) <= AXIS_TOL and abs(y1 - y2) > AXIS_TOL
        if not (is_h or is_v):
            continue
        if (abs(x2 - x1) if is_h else abs(y2 - y1)) < MIN_LEN:
            continue
        segs.append(LineString([(x1, y1), (x2, y2)]))

polys = list(polygonize(unary_union(segs)))
polys_pre = [p for p in polys if p.area <= MAX_AREA_FRAC * page_area]
n = len(polys_pre)
parent = list(range(n))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
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
def _touches_edge(p):
    minx, miny, maxx, maxy = p.bounds
    return (minx <= EDGE_TOL or miny <= EDGE_TOL
            or maxx >= pw - EDGE_TOL or maxy >= ph - EDGE_TOL)
comp_has_edge = {}
for i, p in enumerate(polys_pre):
    r = find(i)
    if _touches_edge(p):
        comp_has_edge[r] = True
    comp_has_edge.setdefault(r, False)
polys_kept = [p for i, p in enumerate(polys_pre) if comp_has_edge[find(i)]]

# --- 2. Words: drawing-body text outside the title-block polygons -----------
raw_words = page.get_text("words")
seen = set(); words = []
for w in raw_words:
    k = (w[4], round(w[0], 1), round(w[1], 1))
    if k in seen:
        continue
    seen.add(k); words.append(w)
outside_word_rects = []
for w in words:
    wx0, wy0, wx1, wy1 = w[:4]
    cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
    if any(p.contains(Point(cx, cy)) for p in polys_kept):
        continue  # inside title block -> removed by polygon white-fill below
    outside_word_rects.append((wx0 - WORD_PAD, wy0 - WORD_PAD,
                               wx1 + WORD_PAD, wy1 + WORD_PAD))
print(f"polys_kept={len(polys_kept)}  outside-words={len(outside_word_rects)}",
      file=sys.stderr)

# --- 3. glyph-tight suppression test ---------------------------------------
def rect_inter(a, b):
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)

def suppressed(r):
    # Inflate to a minimum thickness so long thin lines get a length-proportional
    # area (a small word box can't then exceed SUPPRESS_FRAC of a long edge).
    x0, y0, x1, y1 = r
    if x1 - x0 < LINE_FLOOR:
        m = (x0 + x1) / 2; x0, x1 = m - LINE_FLOOR / 2, m + LINE_FLOOR / 2
    if y1 - y0 < LINE_FLOOR:
        m = (y0 + y1) / 2; y0, y1 = m - LINE_FLOOR / 2, m + LINE_FLOOR / 2
    infl = (x0, y0, x1, y1)
    area = (x1 - x0) * (y1 - y0)
    for wr in outside_word_rects:
        if rect_inter(infl, wr) / area > SUPPRESS_FRAC:
            return True
    return False

# --- 4. Render geometry from vector paths, dropping text glyph paths --------
ref_w, ref_h = Image.open(REF_PNG).size
zoom = ref_w / pw
newdoc = fitz.open()
newpage = newdoc.new_page(width=pw, height=ph)
shape = newpage.new_shape()
def replay(d):
    for it in d.get("items", []):
        op = it[0]
        try:
            if op == "l":
                shape.draw_line(it[1], it[2])
            elif op == "c":
                shape.draw_bezier(it[1], it[2], it[3], it[4])
            elif op == "re":
                shape.draw_rect(it[1])
            elif op == "qu":
                shape.draw_quad(it[1])
        except Exception:
            continue
n_kept = n_supp = 0
for d in drawings:
    rr = d.get("rect")
    is_supp = rr is not None and suppressed((rr.x0, rr.y0, rr.x1, rr.y1))
    w = d.get("width") or 0.0
    if is_supp:
        n_supp += 1
        if not DEBUG:
            continue
        replay(d)
        shape.finish(color=(1, 0, 0), fill=None, width=max(w, 0.5), closePath=False)
    else:
        n_kept += 1
        replay(d)
        shape.finish(color=d.get("color"), fill=d.get("fill"),
                     width=max(w, 0.1), closePath=False)
shape.commit()
print(f"paths kept={n_kept} suppressed={n_supp}", file=sys.stderr)

pix = newpage.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
arr = arr[:, :, :3].copy()
im = Image.fromarray(arr)
draw = ImageDraw.Draw(im)
sx = sy = zoom

# --- 5. White-fill title-block / frame polygons (area fill = geometry-safe) -
if not DEBUG:
    for p in polys_kept:
        xs, ys = p.exterior.xy
        pts = [(x * sx, y * sy) for x, y in zip(xs, ys)]
        draw.polygon(pts, fill=(255, 255, 255))
        draw.line(pts + [pts[0]], fill=(255, 255, 255), width=4)

# --- 6. Mask content-edge lines (frame/border near the content bbox) --------
nx, ny = PAGE_NOISE_PCT * pw, PAGE_NOISE_PCT * ph
all_x, all_y = [], []
for seg in segs:
    (x1, y1), (x2, y2) = seg.coords[0], seg.coords[-1]
    if ((x1 < nx and x2 < nx) or (x1 > pw - nx and x2 > pw - nx)
            or (y1 < ny and y2 < ny) or (y1 > ph - ny and y2 > ph - ny)):
        continue
    all_x += [x1, x2]; all_y += [y1, y2]
if not DEBUG and all_x:
    bx0, by0, bx1, by1 = min(all_x), min(all_y), max(all_x), max(all_y)
    ex, ey = EDGE_PCT * (bx1 - bx0), EDGE_PCT * (by1 - by0)
    for seg in segs:
        (x1, y1), (x2, y2) = seg.coords[0], seg.coords[-1]
        if ((x1 < bx0 + ex and x2 < bx0 + ex) or (x1 > bx1 - ex and x2 > bx1 - ex)
                or (y1 < by0 + ey and y2 < by0 + ey) or (y1 > by1 - ey and y2 > by1 - ey)):
            draw.line([x1 * sx, y1 * sy, x2 * sx, y2 * sy], fill=(255, 255, 255), width=4)

# --- 7. Crop to non-white content ------------------------------------------
arr = np.array(im)
non_white = (arr < 250).any(axis=-1)
rows = np.where(non_white.any(axis=1))[0]
cols = np.where(non_white.any(axis=0))[0]
if rows.size and cols.size:
    pad = 4
    y0, y1 = max(int(rows.min()) - pad, 0), min(int(rows.max()) + pad, arr.shape[0])
    x0, x1 = max(int(cols.min()) - pad, 0), min(int(cols.max()) + pad, arr.shape[1])
    Image.fromarray(arr[y0:y1, x0:x1]).save(OUT)
    print(f"crop=({x0},{y0},{x1},{y1}) -> {OUT}")
else:
    im.save(OUT)
    print(f"no content crop -> {OUT}")
