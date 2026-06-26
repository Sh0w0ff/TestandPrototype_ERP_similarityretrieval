"""bom — find + parse the embedded BOM table into KC-schema rows. Two layouts:
  (a) whole-ROW: one cell per BOM line -> anchor-parse by field type (parse_row);
  (b) per-COLUMN: Description/Specification/Item-No are separate cells -> find the header
      band, snap data cells above it to columns (parse_grid_columns).

Parser copied verbatim from the frozen legacy reference (scripts/part1/ocr_table_parse.py).
`cell_text` is adapted to read the new unified {"bbox_pt","text"} cell shape produced by
box_text/box_ocr, while still accepting the legacy {"pymupdf","paddle"} shape — so the same
parser runs on both. Parity-guarded by tests/test_bom_parity.py.

Public API:
  bom_rows(cells)  -> list of KC-schema row dicts. cells = [{"bbox_pt":[x0,y0,x1,y1], "text":...}]
"""
import re

SCHEMA = ["pos", "qty", "description", "specification",
          "width", "length", "item_no", "drawing_no", "weight_kg"]

# --- field-TYPE recognizers (the anchors) -------------------------------------
RE_INT      = re.compile(r"^\d{1,3}$")
RE_ITEM     = re.compile(r"^\d{7,9}$")
RE_WEIGHT   = re.compile(r"^\d{1,4}[.,]\d{1,2}$|^\d{1,4}$")
RE_DRAWING  = re.compile(r"-DRW\d|-A$|-B$|PDF\d|^D[S]?\d", re.I)
RE_SPEC     = re.compile(r"EN\d|S355|S235|PL\d|SQ[A-Z]|HDG|DX5", re.I)


def cell_text(c):
    """Text for a cell, source-agnostic: the unified 'text' key (box_text/box_ocr), falling
    back to the legacy 'pymupdf'/'paddle' keys so the parser runs on either cell shape."""
    return (c.get("text") or c.get("pymupdf") or c.get("paddle") or "").strip()


def looks_like_bom_row(text):
    """A data row: starts with an integer pos AND carries a material/spec token."""
    toks = text.split()
    return bool(toks) and RE_INT.match(toks[0]) and any(RE_SPEC.search(t) for t in toks)


def parse_row(text):
    """Anchor-parse one whole-row BOM string into the KC schema (best-effort)."""
    toks = text.split()
    row = {k: "" for k in SCHEMA}
    row["pos"] = toks[0]
    body = toks[1:]
    if body and RE_WEIGHT.match(body[-1]):
        row["weight_kg"] = body.pop()
    if body and RE_DRAWING.search(body[-1]):
        row["drawing_no"] = body.pop()
    for i in range(len(body) - 1, -1, -1):
        if RE_ITEM.match(body[i]):
            row["item_no"] = body.pop(i)
            break
    spec_i = next((i for i, t in enumerate(body) if RE_SPEC.search(t)), None)
    if spec_i is not None:
        row["description"] = " ".join(body[:spec_i])
        row["specification"] = body[spec_i]
        rest = body[spec_i + 1:]
    else:
        row["description"] = " ".join(body)
        rest = []
    ints = [t for t in rest if RE_INT.match(t)]
    if len(ints) >= 1:
        row["width"] = ints[0]
    if len(ints) >= 2:
        row["length"] = ints[1]
    return row


# --- per-COLUMN layout --------------------------------------------------------
HEADER_VOCAB = {
    "specification": "specification", "description": "description",
    "drawing no": "drawing_no", "drawing id": "drawing_no", "drawing": "drawing_no",
    "item no": "item_no", "item id": "item_no",
    "weight kg": "weight_kg", "weight": "weight_kg",
    "width": "width", "length": "length", "qty": "qty", "pos": "pos",
}


def _norm(t):
    return re.sub(r"[^a-z ]", " ", t.lower()).strip()


def header_label(text):
    n = _norm(text)
    for lab in sorted(HEADER_VOCAB, key=len, reverse=True):
        if n == lab or n.startswith(lab + " "):
            return HEADER_VOCAB[lab]
    return None


def parse_grid_columns(cells):
    """Find the BOM header band (>=3 header labels at one y) and snap data cells above it to
    columns by x-center. Returns (rows, header_cols, consumed_bboxes) — consumed_bboxes is the
    set of cell bboxes (tuples) the grid layout absorbs (header + snapped data), so callers can
    separate the embedded BOM (a TARGET label) from the title-block input feature."""
    items = [(c["bbox_pt"], cell_text(c), header_label(cell_text(c))) for c in cells]
    hdr = [(b, lab) for b, txt, lab in items if lab]
    if len(hdr) < 3:
        return [], [], set()
    hdr.sort(key=lambda t: t[0][1])
    bands = []
    for b, lab in hdr:
        yc = (b[1] + b[3]) / 2
        if bands and abs(yc - bands[-1][0]) <= (b[3] - b[1]):
            bands[-1][1].append((b, lab))
        else:
            bands.append((yc, [(b, lab)]))
    hy, hcols = max(bands, key=lambda bd: len(bd[1]))
    if len(hcols) < 3:
        return [], [], set()
    header_vis = [(b[0], b[1], b[2], b[3], lab) for b, lab in hcols]
    cols = [(b[0], b[2], lab) for b, lab in hcols]
    cols.sort()
    hx0, hx1 = min(c[0] for c in cols), max(c[1] for c in cols)
    hh = max(b[3] - b[1] for b, _ in hcols)
    data = [(b, txt) for b, txt, lab in items
            if lab is None and hx0 - 2 <= (b[0] + b[2]) / 2 <= hx1 + 2
            and hh * 0.3 < (hy - (b[1] + b[3]) / 2) <= 12 * hh]
    data.sort(key=lambda t: (t[0][1] + t[0][3]) / 2)
    drows = []
    for b, txt in data:
        yc = (b[1] + b[3]) / 2
        ch = b[3] - b[1]
        if drows and abs(yc - drows[-1][0]) <= 0.6 * ch:
            drows[-1][1].append((b, txt))
        else:
            drows.append([yc, [(b, txt)]])
    out = []
    consumed = {tuple(b) for b, _lab in hcols}   # header-label cells are part of the BOM table
    for yc, members in drows:
        row = {k: "" for k in SCHEMA}
        for b, txt in members:
            cx = (b[0] + b[2]) / 2
            col = next((lab for x0, x1, lab in cols if x0 - 2 <= cx <= x1 + 2), None)
            if col:
                row[col] = (row[col] + " " + txt).strip()
        if any(row[k] for k in ("description", "specification", "item_no")):
            bbox = [min(b[0] for b, _ in members), min(b[1] for b, _ in members),
                    max(b[2] for b, _ in members), max(b[3] for b, _ in members)]
            out.append((bbox, row))
            consumed.update(tuple(b) for b, _ in members)   # data cells kept into a row
    return out, header_vis, consumed


def bom_header(cells):
    """The detected BOM column-header labels, left-to-right (per-COLUMN grid layout). Falls back
    to the KC SCHEMA when there's no header band but data rows exist (whole-ROW layout)."""
    _rows, hdr, _consumed = parse_grid_columns(cells)
    if hdr:
        return [lab for *_b, lab in sorted(hdr, key=lambda h: h[0])]
    return list(SCHEMA) if bom_rows(cells) else []


def bom_rows(cells):
    """cells: list of {bbox_pt, text}. Returns list of KC-schema row dicts.
    Try whole-ROW layout first; if nothing, fall back to per-COLUMN grid."""
    rows = []
    for c in sorted(cells, key=lambda c: c["bbox_pt"][1]):
        t = cell_text(c)
        if looks_like_bom_row(t):
            rows.append(parse_row(t))
    if not rows:
        grid, _hdr, _consumed = parse_grid_columns(cells)
        rows = [r for _b, r in grid]
    return rows


def bom_consumed_bboxes(cells):
    """Set of cell bboxes (as tuples) the BOM parser absorbs — used to separate the embedded BOM
    (a prediction TARGET / training label) from the title-block INPUT feature, so the same table
    text never appears on both sides (leakage). Mirrors bom_rows' layout choice: whole-ROW cells
    first, else the per-COLUMN grid's header + data cells."""
    whole = {tuple(c["bbox_pt"]) for c in cells if looks_like_bom_row(cell_text(c))}
    if whole:
        return whole
    _out, _hdr, consumed = parse_grid_columns(cells)
    return consumed
