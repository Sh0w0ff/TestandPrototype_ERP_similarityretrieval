"""Orchestrator: build the unified per-drawing SIGNAL from the self-contained text_scripts modules.

Routes vector vs OCR by pymupdf word-count band (<=50 -> OCR), assembles
cache/signal_v2/<stem>/signal.json + a raw-vs-signal dump-compare png.

Layers:
  region/box_text/box_ocr  -> box (title-block/table) cell text  [+ zone-marker filter]
  bom                      -> structured BOM rows (target label / baseline / eval)
  body_text/body_ocr       -> raw body text  -> _util.light_clean -> kept notes + routed dims
  extract_text (legacy)    -> fields/standards/RAL/treatments  (interpret layer; RAL scans body too)
  vocab.ALL_STANDARDS_LIBS -> vocab_tags over box text + notes (the 14 standard databases)

Usage:  python build_signal.py STEM [STEM ...]
"""
import sys, csv, json, functools
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))          # text_scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "part1"))  # legacy extract_text, vocab

from text_scripts import region as R
from text_scripts import box_text as BT
from text_scripts import box_ocr as BO
from text_scripts import body_text as BDT
from text_scripts import body_ocr as BDO
from text_scripts import bom as BOM
from text_scripts import vocab_tags as VT
from text_scripts import field_tags as FLD
from text_scripts._util import light_clean, is_zone_marker
import extract_text as X     # interpret layer (deferred decision) — fields/standards

ROOT = Path("/Users/sh0w0ff/FYP")
sys.path.insert(0, str(ROOT / "scripts"))
import paths
OUT = paths.TEXT_PIPE                          # <stem>/signal.json
WC_CSV = ROOT / "cache" / "pymupdf_word_counts.csv"
BAND = 50
DEBUG_DUMP = "--debug" in sys.argv             # signal_dump.png only when asked (was always-on bloat)
ALL_PAGES = "--all-pages" in sys.argv          # DRAWING-level: extract every page + UNION (52% are multipage)
SIG_NAME = "signal_allpages.json" if ALL_PAGES else "signal.json"   # keep the page-1 baseline intact


@functools.lru_cache(maxsize=1)
def _counts():
    d = {}
    if WC_CSV.exists():
        with open(WC_CSV) as f:
            r = csv.reader(f); next(r, None)
            for row in r:
                if len(row) >= 2:
                    try:
                        d[row[1]] = int(row[0])
                    except ValueError:
                        pass
    return d


def route(stem, page_idx=0):
    if page_idx == 0:
        wc = _counts().get(stem)              # the cached CSV is page-1 only
        if wc is None:
            import pymupdf as fitz
            wc = len(fitz.open(R.find_pdf(stem))[0].get_text("words"))
    else:
        import pymupdf as fitz
        wc = len(fitz.open(R.find_pdf(stem))[page_idx].get_text("words"))
    return ("ocr" if wc <= BAND else "vector"), wc


def _undouble(text):
    """Collapse exact text doubling: some ABB PDFs render text in two overlapping layers, so pymupdf
    reads every word twice -> a cell reads 'I.Kallio I.Kallio' / 'Title SUPPORT.. Title SUPPORT..'.
    If the token sequence is its first half repeated verbatim, keep one half. (Conservative: only
    EXACT whole-string doubling; partial trailing repeats are left alone.)"""
    toks = text.split()
    n = len(toks)
    if n >= 2 and n % 2 == 0 and toks[:n // 2] == toks[n // 2:]:
        return " ".join(toks[:n // 2])
    return text


def _extract_page(stem, page_idx):
    """Extract ONE page into its partial signal pieces. BOM routing happens PER PAGE so the
    grid parser sees a single page's cell grid (multipage BOMs sit on page 2+ for KC). Each
    record carries its source `page` for drawing-level provenance. Returns a dict of partials."""
    mode, wc = route(stem, page_idx)
    cells = BO.box_ocr(stem, page_idx) if mode == "ocr" else BT.box_text(stem, page_idx)
    for c in cells:                              # collapse overlapping-layer text doubling at the source
        c["text"] = _undouble((c.get("text") or "").strip())
    # BOM parses the FULL cell set (needs pos numbers); record which cells it absorbs so they are
    # kept OUT of the input feature (embedded BOM = a TARGET label, never an input — leakage guard).
    bom_input = [{"bbox_pt": c["bbox_pt"], "text": (c.get("text") or "")} for c in cells]
    rows = BOM.bom_rows(bom_input)
    bom_head = BOM.bom_header(bom_input)        # detected column headers (Pos/Qty/Description/...)
    bom_bboxes = BOM.bom_consumed_bboxes(bom_input)   # cells the BOM table occupies -> not input
    # zone-marker filter (recorded, not lost); BOM cells split off into bom_cells (the TARGET).
    box_texts, zone_markers, bom_text_cells = [], [], []
    for c in cells:
        t = (c.get("text") or "").strip()
        if not t:
            continue
        rec = {"bbox_pt": c["bbox_pt"], "text": t, "page": page_idx}
        if tuple(c["bbox_pt"]) in bom_bboxes:
            bom_text_cells.append(rec)                 # embedded-BOM table cell -> TARGET side
        elif is_zone_marker(t):
            zone_markers.append(rec)
        else:
            box_texts.append(rec)

    # BODY stays a LIST of spatial blocks (NOT flattened) — like title_block.cells. Each block keeps
    # its bbox + raw text, its own light_clean (notes/dims/dropped), and its own phrase-level tags.
    raw_blocks = (BDO.body_groups(stem, page_idx=page_idx) if mode == "ocr"
                  else BDT.body_blocks(stem, page_idx))
    body, dropped_blocks = [], []
    for b in raw_blocks:
        txt = _undouble(b.get("text", "").strip())
        if not txt:
            continue
        if len(txt) <= 2:                       # low-character blocks (a4, 04, A, II): noise
            dropped_blocks.append(txt); continue
        kept, dims, dropped = light_clean(txt)
        tags = VT.tag_unit(" ".join(kept))                # tag over the CLEANED notes (dims already out)
        # Prune-from-fingerprint flag (user rule 2026-06-04): a block is NOISE only if it has NO content
        # token, content = len>=4 OR it produced a vocab tag. Tagging ran FIRST, so weld/fit codes
        # (a4, H11 — len<=3, alone on a line) are tag-content and survive; section labels A-A/B-B,
        # grid refs, garble (no len>=4, no tag) are flagged. Stopwords ride in real sentences -> kept.
        noise = not (any(len(tok) >= 4 for tok in kept) or bool(tags))
        body.append({
            "bbox": b.get("bbox") or b.get("bbox_pt"), "page": page_idx,
            "text": txt,                                  # raw block text (verbatim)
            "notes": kept, "dims": dims, "dropped": dropped,
            "tags": tags, "noise": noise,                 # noise -> excluded from the notes fingerprint
        })
    return {"page": page_idx, "mode": mode, "wc": wc, "n_cells": len(cells),
            "box_texts": box_texts, "zone_markers": zone_markers, "bom_text_cells": bom_text_cells,
            "rows": rows, "bom_head": bom_head, "body": body, "dropped_blocks": dropped_blocks}


_NAMED_TEXT = ("part_name", "material", "coating", "general_tolerance", "specification", "based_on")


def _to_units(typed, tagged_notes):
    """UNIFIED 3-column structured input: every classified text becomes one row (reason | name |
    text). reason = how we understood it: 'named' (a label-driven field value), 'derived' (a vocab
    normalization of a field value, e.g. part_type), 'standard' (a recognised standard), 'tag' (a
    note carrying vocab tags, full text kept). NAME is the machine-readable key — the field name,
    or for a tag row the vocab CATEGORY(ies). This replaces the old split fields / vocab_tags /
    tagged_notes sections with one form (user request, thesis §9.8.27)."""
    rows = []
    for f in _NAMED_TEXT:                                   # named label-driven text values
        for v in typed.get(f) or []:
            rows.append({"reason": "named", "name": f, "text": v})
    for v in typed.get("scale") or []:
        rows.append({"reason": "named", "name": "scale", "text": v})
    if typed.get("weight_kg") is not None:
        rows.append({"reason": "named", "name": "weight_kg", "text": typed["weight_kg"]})
    for f in ("part_type", "material_class"):              # vocab-normalized derivations
        for v in typed.get(f) or []:
            rows.append({"reason": "derived", "name": f, "text": v})
    for s in typed.get("standards") or []:                 # recognised standards
        num = f"{s['family']} {s['number']}" + (f"-{s['suffix']}" if s.get("suffix") else "")
        rows.append({"reason": "standard", "name": s["family"], "text": num})
    for tn in tagged_notes:                                # vocab-tagged notes (full text kept)
        cats = sorted({c for r in tn.get("tags", []) for c in r.get("tags", {})})
        if not cats:                                       # no vocab tag -> its value is already a named/standard row
            continue
        txt = " ".join(tn.get("notes", [])) or tn.get("text", "")
        rows.append({"reason": "tag", "name": "+".join(cats), "text": txt,
                     "src": tn.get("src"), "page": tn.get("page"), "tags": tn.get("tags")})
    return rows


def build(stem):
    # Retrieval unit is the DRAWING (one PDF -> one ERP item). Page-1 only by default (legit v1);
    # --all-pages extracts every page and UNIONS them into ONE drawing-level fingerprint.
    pages = list(range(R.page_count(stem))) if ALL_PAGES else [0]
    per = []
    for pi in pages:
        try:
            per.append(_extract_page(stem, pi))
        except Exception as e:                       # an empty/corrupt later page must not sink the
            if pi == 0:                              # whole drawing; page 0 failing is fatal as before
                raise
            print(f"    [page {pi} skipped] {type(e).__name__}: {e}")
            per.append({"page": pi, "mode": "skip", "wc": 0, "n_cells": 0, "box_texts": [],
                        "zone_markers": [], "bom_text_cells": [], "rows": [], "bom_head": {},
                        "body": [], "dropped_blocks": []})

    # CROSS-PAGE dedup: multipage drawings repeat the SAME title block on every sheet (user obs
    # 2026-06-04), so the union would carry identical cells/notes N times. Drop a later page's
    # entry ONLY when its exact text already appeared on an EARLIER page (a real BOM continuation
    # sheet has different rows, so exact-text match is safe). page-0-only never filters -> single-
    # page output stays byte-identical to the page-1 baseline.
    n_dups = [0]
    def union_dedup(field):
        seen, out = set(), []
        for p in per:
            for rec in p[field]:
                k = " ".join((rec.get("text") or "").lower().split())
                if k and k in seen:
                    n_dups[0] += 1
                    continue
                out.append(rec)
            for rec in p[field]:
                k = " ".join((rec.get("text") or "").lower().split())
                if k:
                    seen.add(k)
        return out

    box_texts      = union_dedup("box_texts")
    zone_markers   = union_dedup("zone_markers")
    bom_text_cells = union_dedup("bom_text_cells")
    body           = union_dedup("body")
    rows           = [r for p in per for r in p["rows"]]
    dropped_blocks = [d for p in per for d in p["dropped_blocks"]]
    bom_head       = next((p["bom_head"] for p in per if p["bom_head"]), per[0]["bom_head"])
    n_cells        = sum(p["n_cells"] for p in per)
    mode, wc       = per[0]["mode"], per[0]["wc"]         # page-1 route is the headline (back-compat)

    # Vocab tagging runs AFTER light_clean (so dims/garbage are already routed out — no numeric
    # false-matches).
    # UNIFIED structured layer: LABEL-DRIVEN typed fields (part_name/part_type/material/coating/
    # general_tolerance/scale + nullable weight_kg) PLUS standards (content scan), all from field_tags.
    # Units = each title-block cell + each body CLAUSE (labels sit at a cell/clause start). This REPLACES
    # the old title_block.fields (dead colon-only extract_fields) + title_block.standards (thesis §9.8.25).
    units = [b["text"] for b in box_texts]
    for bl in body:
        if not bl["noise"]:
            units += VT.clauses(" ".join(bl["notes"]))
    typed = FLD.type_units(units)
    vendor = X.guess_vendor(stem)

    # ---- PARTITION every text unit into ONE bucket: classified / unclassified / bom / debug ----
    # A unit is CLASSIFIED if we made sense of it: it became a named field (-> classified.fields,
    # the value represents it) OR it carries a vocab tag / standard (-> classified.tagged_notes,
    # full text kept so nothing is lost). Otherwise it is residual text we could not classify yet
    # (-> unclassified.blocks / .body — the coverage gap). zone/noise/dropped/dims = debug only.
    def _named(text):
        t = FLD.type_units([text])
        named = (any(t[k] for k in ("part_name", "material", "coating", "general_tolerance",
                                     "scale", "specification")) or t["weight_kg"] is not None)
        return named, t["standards"]

    unclassified_blocks, tagged_notes, admin_cells = [], [], []
    for c in box_texts:                                   # title-block cells
        named, stds = _named(c["text"])
        if named:
            continue                                      # represented in classified.fields
        rec = {"text": c["text"], "page": c.get("page"), "bbox": c["bbox_pt"]}
        if FLD.is_admin(c["text"]) or FLD.is_person(c["text"]):  # admin chrome / author names -> debug only
            admin_cells.append(rec)
            continue
        ctags = VT.tag_unit(" ".join(light_clean(c["text"])[0]))
        if ctags or stds:
            tagged_notes.append({**rec, "src": "block", "tags": ctags})
        else:
            unclassified_blocks.append(rec)

    unclassified_body, noise_blocks, dims_all = [], [], []
    for b in body:                                        # body spatial blocks
        rec = {"text": b["text"], "page": b.get("page"), "bbox": b.get("bbox")}
        if b.get("dims"):
            dims_all.append({"page": b.get("page"), "dims": b["dims"]})
        if b["noise"]:
            noise_blocks.append({**rec, "notes": b["notes"]})
            continue
        btext = " ".join(b["notes"]) or b["text"]
        if FLD.is_admin(btext):                           # boilerplate / BOM-sheet header prose -> debug (leak-safe)
            admin_cells.append({**rec, "notes": b["notes"]})
            continue
        named, stds = _named(b["text"])
        if b["tags"] or named or stds:
            tagged_notes.append({**rec, "src": "body", "notes": b["notes"], "tags": b["tags"]})
        else:
            unclassified_body.append({**rec, "notes": b["notes"]})

    sig = {
        "stem": stem, "route": mode, "pymupdf_words": wc, "vendor": vendor,
        "n_pages": len(pages),
        "page_routes": [{"page": p["page"], "route": p["mode"], "words": p["wc"]} for p in per],
        "classified": {                # what we made sense of (the input signal proper)
            "units": _to_units(typed, tagged_notes),   # UNIFIED 3-column form (reason|name|text)
            "fields": typed,           #   typed backbone (numeric/normalized; powers retrieval tokens)
        },
        "unclassified": {              # residual text we could not classify (the coverage gap)
            "blocks": unclassified_blocks,
            "body": unclassified_body,
        },
        "bom": {                       # embedded BOM = prediction TARGET (never an input feature)
            "header": bom_head, "cells": bom_text_cells, "rows": rows,
        },
        "debug": {                     # filtered-out / dropped, kept for inspection only
            "zone_markers": zone_markers,
            "admin": admin_cells,      # recognized title-block administrative chrome (non-signal)
            "noise_blocks": noise_blocks,
            "dropped_blocks": dropped_blocks,
            "dims": dims_all,
        },
        "counts": {
            "cells": n_cells, "tagged_notes": len(tagged_notes),
            "unclassified_blocks": len(unclassified_blocks), "unclassified_body": len(unclassified_body),
            "admin": len(admin_cells),
            "bom_rows": len(rows), "bom_cells": len(bom_text_cells),
            "noise": len(noise_blocks), "crosspage_dups": n_dups[0],
        },
    }
    d = OUT / stem
    d.mkdir(parents=True, exist_ok=True)
    json.dump(sig, open(d / SIG_NAME, "w"), indent=1, ensure_ascii=False)
    if DEBUG_DUMP:
        try:
            dump(stem, sig)
        except Exception as e:
            print(f"  [dump warn] {type(e).__name__}: {e}")
    pg = f"pg={len(pages):>2} dup={n_dups[0]:>3} " if ALL_PAGES else ""
    print(f"  {stem[:40]:<42} {pg}route={mode:<6} cells={n_cells:>3} -> "
          f"classified(fields={sum(1 for k,v in typed.items() if v and k!='weight_kg' or (k=='weight_kg' and v is not None))} "
          f"tagged={len(tagged_notes):>2}) unclass(blk={len(unclassified_blocks):>2} body={len(unclassified_body):>2}) "
          f"bom={len(rows):>2} -> {d/SIG_NAME}")
    return sig


# ---- DRAWING | RAW PATHWAYS | FILTERED FINAL  3-panel dump --------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
COL_W, CHARS, LH, PAD = 560, 70, 19, 10

# BOM display columns: (schema key, short label, char width). Empty columns are dropped so the
# header and every row line up under the same fixed-width grid.
BOM_COLS = [("pos", "pos", 3), ("qty", "qty", 3), ("description", "descr", 11),
            ("specification", "specification", 16), ("width", "w", 4), ("length", "len", 5),
            ("item_no", "item_no", 9), ("drawing_no", "drawing_no", 11), ("weight_kg", "kg", 6)]


def _bom_table(rows):
    """Render bom_rows as a fixed-width table: header label row + each data row aligned under its
    column. Columns with no data across all rows are dropped to keep the line within the panel."""
    if not rows:
        return ["(none)"]
    keep = [(k, lab, w) for k, lab, w in BOM_COLS if any((r.get(k) or "").strip() for r in rows)]
    fmt = lambda vals: " ".join(f"{str(v)[:w]:<{w}}" for (k, lab, w), v in zip(keep, vals))
    out = [fmt([lab for _k, lab, _w in keep]),
           "-" * (sum(w for *_x, w in keep) + len(keep))]
    out += [fmt([r.get(k, "") for k, _lab, _w in keep]) for r in rows]
    return out


def _wrap(text, width=CHARS):
    out, line = [], ""
    for tok in str(text).split(" "):
        if len(line) + len(tok) + 1 > width:
            out.append(line); line = tok
        else:
            line = f"{line} {tok}".strip()
    if line:
        out.append(line)
    return out or [""]


def _block(title, lines, color=(20, 20, 20)):
    rows = [(title, (150, 40, 40), True), ("-" * CHARS, (205, 205, 205), False)]
    rows += [(ln, color, False) for ln in lines]
    rows.append(("", color, False))
    return rows


def _col(rows):
    h = PAD * 2 + LH * len(rows)
    canvas = np.full((h, COL_W, 3), 255, np.uint8)
    y = PAD + LH
    for text, color, is_hdr in rows:
        cv2.putText(canvas, text, (PAD, y), FONT, 0.48 if is_hdr else 0.44, color, 1, cv2.LINE_AA)
        y += LH
    return canvas


def _label(im, txt):
    bar = np.full((40, im.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, txt, (10, 28), FONT, 0.7, (120, 40, 40), 2)
    return np.vstack([bar, im])


def dump(stem, sig):
    npg = sig.get("n_pages", 1)
    pfx = lambda c: f"p{c.get('page', 0)}|" if npg > 1 else ""   # page tag only when multipage
    cl = sig.get("classified", {})
    un = sig.get("unclassified", {})
    dbg = sig.get("debug", {})
    bom = sig.get("bom", {})
    cnt = sig.get("counts", {})

    def _notes_inline(recs, color, title):
        lines = []
        for b in recs:                                     # no index numbering (user: meaningless on body)
            txt = " ".join(b.get("notes", [])) or b.get("text", "")
            w = _wrap(txt)
            lines.append(f"{pfx(b)}{w[0]}")
            lines += [f"    {x}" for x in w[1:]]
        return _block(title, lines or ["(none)"], color=color)

    # ---- CLASSIFIED column: ONE unified 3-column form (reason | name | text) + BOM(target) ----
    routes = "  ".join(f"p{r['page']}:{r['route']}({r['words']}w)" for r in sig.get("page_routes", []))
    cls = _block(f"STEM {stem[:54]}",
                 [f"route={sig['route']}  words={sig['pymupdf_words']}  n_pages={npg}", routes or "(single page)",
                  f"classified: {len(cl.get('units', []))} units  unclass: blk={cnt.get('unclassified_blocks',0)} "
                  f"body={cnt.get('unclassified_body',0)}  bom={cnt.get('bom_rows',0)}"])
    RW, NW = 9, 20                                          # reason / name column widths
    form = [f"{'REASON':<{RW}}{'NAME':<{NW}}CONTENT", "-" * CHARS]
    for u in cl.get("units", []):
        w = _wrap(str(u.get("text", "")), width=CHARS - RW - NW)
        form.append(f"{u.get('reason',''):<{RW}}{u.get('name','')[:NW-1]:<{NW}}{w[0]}")
        form += [f"{'':<{RW+NW}}{x}" for x in w[1:]]
    cls += _block(f"[CLASSIFIED] unified form - reason | name | content ({len(cl.get('units', []))})",
                  form, color=(0, 90, 120))
    cls += _block(f"[BOM] -> TARGET, excluded from input ({cnt.get('bom_rows',0)} rows)",
                  _bom_table(bom.get("rows", [])))

    # ---- UNCLASSIFIED + DEBUG column ----
    res = _block(f"[UNCLASSIFIED] title-block cells - no field ({len(un.get('blocks', []))})",
                 [f"{pfx(b)}{b['text']}" for b in un.get("blocks", [])] or ["(none)"], color=(150, 70, 0))
    res += _notes_inline(un.get("body", []), (150, 70, 0),
                         f"[UNCLASSIFIED] body notes - no tag/field ({len(un.get('body', []))})")
    res += _block(f"[DEBUG] admin chrome - title-block metadata ({len(dbg.get('admin', []))})",
                  [f"{pfx(b)}{b['text']}" for b in dbg.get("admin", [])] or ["(none)"], color=(120, 110, 0))
    res += _block(f"[DEBUG] zone_markers ({len(dbg.get('zone_markers', []))})",
                  _wrap(" ".join(c["text"] for c in dbg.get("zone_markers", []))) or ["(none)"], color=(0, 0, 190))
    res += _block(f"[DEBUG] noise blocks ({len(dbg.get('noise_blocks', []))})",
                  _wrap(" | ".join(" ".join(b.get("notes", [])) or b.get("text", "") for b in dbg.get("noise_blocks", []))) or ["(none)"],
                  color=(0, 0, 190))
    res += _block(f"[DEBUG] dropped low-char blocks ({len(dbg.get('dropped_blocks', []))})",
                  _wrap(" | ".join(dbg.get("dropped_blocks", []))) or ["(none)"], color=(0, 0, 190))
    n_dims = sum(len(d.get("dims", [])) for d in dbg.get("dims", []))
    res += _block(f"[DEBUG] routed-out dims ({n_dims})",
                  _wrap(" | ".join(x for d in dbg.get("dims", []) for x in d.get("dims", []))) or ["(none)"], color=(0, 0, 190))

    cr, cf = _col(cls), _col(res)
    H = max(cr.shape[0], cf.shape[0])
    cr = np.vstack([cr, np.full((H - cr.shape[0], COL_W, 3), 255, np.uint8)])
    cf = np.vstack([cf, np.full((H - cf.shape[0], COL_W, 3), 255, np.uint8)])
    # DRAWING panel: every page rendered with its kept cells overlaid, stacked top-to-bottom so
    # the all-pages signal can be eyeballed page-by-page against the union in the text columns.
    page_imgs = []
    for pi in range(npg):
        page, img, sx, sy = R.render(stem, pi)
        for p in R.cells_kept(page, img, sx, sy):
            pts = np.array([[int(x * sx), int(y * sy)] for x, y in p.exterior.coords], np.int32)
            cv2.polylines(img, [pts], True, (0, 170, 0), 3)
        page_imgs.append(_label(img, f"PAGE {pi + 1}/{npg}") if npg > 1 else img)
    PW = min(i.shape[1] for i in page_imgs)            # align all pages to a common width
    page_imgs = [cv2.resize(i, (PW, int(i.shape[0] * PW / i.shape[1]))) for i in page_imgs]
    stacked = np.vstack(page_imgs)
    pg = cv2.resize(stacked, (int(stacked.shape[1] * H / stacked.shape[0]), H))
    sep = np.full((H + 40, 3, 3), 150, np.uint8)
    combo = np.hstack([_label(pg, "DRAWING" + (f" ({npg}pp)" if npg > 1 else "")), sep,
                       _label(cr, "CLASSIFIED (+ BOM target)"), sep, _label(cf, "UNCLASSIFIED + DEBUG")])
    img_dir = paths.text_images(stem); img_dir.mkdir(parents=True, exist_ok=True)
    name = "signal_dump_allpages.png" if ALL_PAGES else "signal_dump.png"
    cv2.imwrite(str(img_dir / name), combo)


if __name__ == "__main__":
    stems = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not stems:
        print("usage: python build_signal.py STEM [STEM ...] [--debug]"); sys.exit(1)
    for s in stems:
        try:
            build(s)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERR {s[:48]}: {type(e).__name__}: {e}")
