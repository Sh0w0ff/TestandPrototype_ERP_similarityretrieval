"""raw_extract.py — Stage 1: extract raw text from every PDF page and cache to raw_pages.json.

Separates the expensive PDF/OCR work from the cheap classification work.
Once raw_pages.json exists for a stem, all classification rule changes (field_tags,
vocab_tags, admin filter, schema partition) can be re-run in seconds without touching OCR.

What lives here (stage 1 — never changes without a new PDF):
  - pymupdf word-count routing (vector vs OCR threshold)
  - box_text / box_ocr  → raw title-block cells
  - body_blocks / body_groups → raw body blocks (no light_clean, no tagging)
  - BOM table detection and row parsing (structural, position-based)
  - BOM bbox exclusion (cells inside the BOM grid flagged is_bom=True)
  - zone marker flagging (is_zone=True)
  - _undouble (ABB double-layer PDF artifact fix)

What stays in build_signal.py (stage 2 — changes often):
  - light_clean (noise routing, dims separation)
  - vocab_tags.tag_unit
  - field_tags.type_units
  - field_tags.is_admin / is_person
  - schema partition into classified / unclassified / bom / debug

Output: text_pipe/<stem>/raw_pages.json

Run:
  python scripts/textpipe/raw_extract.py                    # all stems, resumable
  python scripts/textpipe/raw_extract.py STEM [STEM ...]   # specific stems
  python scripts/textpipe/raw_extract.py --force            # re-extract even if cached
"""
import sys, csv, json, functools, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "part1"))

from text_scripts import region as R
from text_scripts import box_text as BT
from text_scripts import box_ocr as BO
from text_scripts import body_text as BDT
from text_scripts import body_ocr as BDO
from text_scripts import bom as BOM
from text_scripts._util import is_zone_marker
import extract_text as X     # vendor detection only

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import paths

OUT     = paths.TEXT_PIPE
WC_CSV  = ROOT / "cache" / "pymupdf_word_counts.csv"
BAND    = 50          # <=50 pymupdf words -> OCR route


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _wc_cache():
    d = {}
    if WC_CSV.exists():
        with open(WC_CSV) as f:
            r = csv.reader(f); next(r, None)
            for row in r:
                if len(row) >= 2:
                    try: d[row[1]] = int(row[0])
                    except ValueError: pass
    return d


def route(stem, page_idx=0):
    if page_idx == 0:
        wc = _wc_cache().get(stem)
        if wc is None:
            import pymupdf as fitz
            wc = len(fitz.open(R.find_pdf(stem))[0].get_text("words"))
    else:
        import pymupdf as fitz
        wc = len(fitz.open(R.find_pdf(stem))[page_idx].get_text("words"))
    return ("ocr" if wc <= BAND else "vector"), wc


# ---------------------------------------------------------------------------
# ABB double-layer artifact fix
# ---------------------------------------------------------------------------

def _undouble(text: str) -> str:
    toks = text.split()
    n = len(toks)
    if n >= 2 and n % 2 == 0 and toks[:n // 2] == toks[n // 2:]:
        return " ".join(toks[:n // 2])
    return text


# ---------------------------------------------------------------------------
# Per-page raw extraction
# ---------------------------------------------------------------------------

def extract_page_raw(stem: str, page_idx: int) -> dict:
    """Extract one page into raw data — no classification, no light_clean, no tagging."""
    mode, wc = route(stem, page_idx)

    # --- title-block cells ---
    raw_cells = BO.box_ocr(stem, page_idx) if mode == "ocr" else BT.box_text(stem, page_idx)
    for c in raw_cells:
        c["text"] = _undouble((c.get("text") or "").strip())

    # --- BOM table detection (structural, position-based) ---
    bom_input = [{"bbox_pt": c["bbox_pt"], "text": (c.get("text") or "")} for c in raw_cells]
    bom_rows    = BOM.bom_rows(bom_input)
    bom_head    = BOM.bom_header(bom_input)
    bom_bboxes  = BOM.bom_consumed_bboxes(bom_input)  # set of bbox_pt tuples inside BOM grid

    # --- annotate cells: is_bom, is_zone ---
    cells = []
    for c in raw_cells:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        cells.append({
            "text":    text,
            "bbox_pt": c["bbox_pt"],
            "page":    page_idx,
            "is_bom":  tuple(c["bbox_pt"]) in bom_bboxes,
            "is_zone": is_zone_marker(text),
        })

    # --- body blocks: raw text + words, NO light_clean, NO tagging ---
    raw_body = (BDO.body_groups(stem, page_idx=page_idx) if mode == "ocr"
                else BDT.body_blocks(stem, page_idx))
    body_blocks = []
    for b in raw_body:
        txt = _undouble((b.get("text") or "").strip())
        if not txt:
            continue
        body_blocks.append({
            "text":  txt,
            "bbox":  b.get("bbox") or b.get("bbox_pt"),
            "words": b.get("words", []) or txt.split(),
            "page":  page_idx,
        })

    return {
        "page_idx":   page_idx,
        "route":      mode,
        "wc":         wc,
        "cells":      cells,        # raw title-block cells (flagged is_bom / is_zone)
        "bom_rows":   bom_rows,     # parsed BOM table rows
        "bom_head":   bom_head,     # BOM column headers
        "body_blocks": body_blocks, # raw body blocks (no cleaning or tagging)
    }


# ---------------------------------------------------------------------------
# Drawing-level extraction → raw_pages.json
# ---------------------------------------------------------------------------

def extract_raw(stem: str) -> dict:
    n_pages = R.page_count(stem)
    vendor  = X.guess_vendor(stem)
    pages   = []
    for pi in range(n_pages):
        try:
            pages.append(extract_page_raw(stem, pi))
        except Exception as e:
            if pi == 0:
                raise
            print(f"    [page {pi} skipped] {type(e).__name__}: {e}")
            pages.append({"page_idx": pi, "route": "skip", "wc": 0,
                          "cells": [], "bom_rows": [], "bom_head": {},
                          "body_blocks": []})

    raw = {
        "stem":    stem,
        "vendor":  vendor,
        "n_pages": n_pages,
        "pages":   pages,
    }
    out_dir = OUT / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_pages.json").write_text(
        json.dumps(raw, indent=1, ensure_ascii=False)
    )
    return raw


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    force = "--force" in sys.argv
    stem_args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if stem_args:
        stems = stem_args
    else:
        # All stems that have a PDF but no raw_pages.json (or --force)
        all_stems = sorted(
            d.name for d in OUT.iterdir()
            if d.is_dir() and (d / "signal.json").exists()
        )
        if force:
            stems = all_stems
        else:
            stems = [s for s in all_stems if not (OUT / s / "raw_pages.json").exists()]

    print(f"raw_extract: {len(stems)} stems to process"
          + (" (forced)" if force else ""), flush=True)

    t0 = time.time()
    ok = err = 0
    for i, stem in enumerate(stems, 1):
        try:
            extract_raw(stem)
            ok += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERR {stem[:55]}: {type(e).__name__}: {e}", flush=True)
            err += 1
        if i % 50 == 0 or i == len(stems):
            dt = time.time() - t0
            rate = i / dt if dt else 0
            eta  = (len(stems) - i) / rate if rate else 0
            print(f"[{i}/{len(stems)}] ok={ok} err={err} "
                  f"{rate:.2f}/s ETA {eta/60:.1f}m", flush=True)

    print(f"DONE ok={ok} err={err} in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
