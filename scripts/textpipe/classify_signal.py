"""classify_signal.py — Stage 2: classify raw_pages.json into signal.json / signal_allpages.json.

Reads the cached raw extraction (raw_pages.json) and applies all classification logic:
light_clean, vocab_tags, field_tags, admin filter, schema partition, cross-page dedup.

The output is SEMANTICALLY IDENTICAL to the current build_signal.py output. The only
difference is that the OCR/pymupdf extraction is pre-cached — no re-extraction needed
when classification rules change.

Run:
  python scripts/textpipe/classify_signal.py                        # all stems, page-1
  python scripts/textpipe/classify_signal.py --all-pages            # all pages (allpages)
  python scripts/textpipe/classify_signal.py STEM [STEM ...]        # specific stems
  python scripts/textpipe/classify_signal.py --parity STEM [STEM ...] # diff vs existing signal.json

Parity check (run before migrating):
  python scripts/textpipe/classify_signal.py --parity 3AUA0000038918 3AUA0000074295
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "part1"))

from text_scripts import vocab_tags as VT
from text_scripts import field_tags as FLD
from text_scripts._util import light_clean, is_zone_marker
import extract_text as X

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import paths

OUT      = paths.TEXT_PIPE
ALL_PAGES = "--all-pages" in sys.argv
SIG_NAME  = "signal_allpages.json" if ALL_PAGES else "signal.json"

_NAMED_TEXT = ("part_name", "material", "coating", "general_tolerance",
               "specification", "based_on")


# ---------------------------------------------------------------------------
# Helpers (identical logic to build_signal.py)
# ---------------------------------------------------------------------------

def _to_units(typed, tagged_notes):
    rows = []
    for f in _NAMED_TEXT:
        for v in typed.get(f) or []:
            rows.append({"reason": "named", "name": f, "text": v})
    for v in typed.get("scale") or []:
        rows.append({"reason": "named", "name": "scale", "text": v})
    if typed.get("weight_kg") is not None:
        rows.append({"reason": "named", "name": "weight_kg", "text": typed["weight_kg"]})
    for f in ("part_type", "material_class"):
        for v in typed.get(f) or []:
            rows.append({"reason": "derived", "name": f, "text": v})
    for s in typed.get("standards") or []:
        num = f"{s['family']} {s['number']}" + (f"-{s['suffix']}" if s.get("suffix") else "")
        rows.append({"reason": "standard", "name": s["family"], "text": num})
    for tn in tagged_notes:
        cats = sorted({c for r in tn.get("tags", []) for c in r.get("tags", {})})
        if not cats:
            continue
        txt = " ".join(tn.get("notes", [])) or tn.get("text", "")
        rows.append({"reason": "tag", "name": "+".join(cats), "text": txt,
                     "src": tn.get("src"), "page": tn.get("page"), "tags": tn.get("tags")})
    return rows


def _named(text):
    t = FLD.type_units([text])
    named = (any(t[k] for k in ("part_name", "material", "coating", "general_tolerance",
                                 "scale", "specification")) or t["weight_kg"] is not None)
    return named, t["standards"]


# ---------------------------------------------------------------------------
# Classify one page from raw data
# ---------------------------------------------------------------------------

def _classify_page(page_raw: dict) -> dict:
    """Apply classification logic to one page of raw_pages.json data.
    Returns the same shape as build_signal._extract_page()."""
    page_idx = page_raw["page_idx"]
    mode     = page_raw["route"]
    wc       = page_raw["wc"]

    # --- cells: apply is_bom / is_zone flags from raw (already computed) ---
    box_texts, zone_markers, bom_text_cells = [], [], []
    for c in page_raw.get("cells", []):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        rec = {"bbox_pt": c["bbox_pt"], "text": text, "page": page_idx}
        if c.get("is_bom"):
            bom_text_cells.append(rec)
        elif c.get("is_zone"):
            zone_markers.append(rec)
        else:
            box_texts.append(rec)

    # --- body blocks: apply light_clean + noise filter + vocab tags ---
    body, dropped_blocks = [], []
    for b in page_raw.get("body_blocks", []):
        txt = (b.get("text") or "").strip()
        if not txt:
            continue
        if len(txt) <= 2:
            dropped_blocks.append(txt)
            continue
        kept, dims, dropped = light_clean(txt)
        tags = VT.tag_unit(" ".join(kept))
        noise = not (any(len(tok) >= 4 for tok in kept) or bool(tags))
        body.append({
            "bbox":  b.get("bbox"),
            "page":  page_idx,
            "text":  txt,
            "notes": kept, "dims": dims, "dropped": dropped,
            "tags":  tags, "noise": noise,
        })

    return {
        "page": page_idx, "mode": mode, "wc": wc,
        "n_cells":        len(page_raw.get("cells", [])),
        "box_texts":      box_texts,
        "zone_markers":   zone_markers,
        "bom_text_cells": bom_text_cells,
        "rows":           page_raw.get("bom_rows", []),
        "bom_head":       page_raw.get("bom_head", {}),
        "body":           body,
        "dropped_blocks": dropped_blocks,
    }


# ---------------------------------------------------------------------------
# Drawing-level classification
# ---------------------------------------------------------------------------

def classify(stem: str, raw: dict = None) -> dict:
    if raw is None:
        raw_path = OUT / stem / "raw_pages.json"
        if not raw_path.exists():
            raise FileNotFoundError(f"raw_pages.json missing for {stem} — run raw_extract.py first")
        raw = json.loads(raw_path.read_text())

    pages_raw = raw["pages"] if ALL_PAGES else raw["pages"][:1]
    per = [_classify_page(p) for p in pages_raw]

    # cross-page dedup (identical to build_signal.py)
    n_dups = [0]
    def union_dedup(field):
        seen, out = set(), []
        for p in per:
            for rec in p[field]:
                k = " ".join((rec.get("text") or "").lower().split())
                if k and k in seen:
                    n_dups[0] += 1; continue
                out.append(rec)
            for rec in p[field]:
                k = " ".join((rec.get("text") or "").lower().split())
                if k: seen.add(k)
        return out

    box_texts      = union_dedup("box_texts")
    zone_markers   = union_dedup("zone_markers")
    bom_text_cells = union_dedup("bom_text_cells")
    body           = union_dedup("body")
    rows           = [r for p in per for r in p["rows"]]
    dropped_blocks = [d for p in per for d in p["dropped_blocks"]]
    bom_head       = next((p["bom_head"] for p in per if p["bom_head"]), per[0]["bom_head"])
    n_cells        = sum(p["n_cells"] for p in per)
    mode, wc       = per[0]["mode"], per[0]["wc"]

    # field classification
    units = [b["text"] for b in box_texts]
    for bl in body:
        if not bl["noise"]:
            units += VT.clauses(" ".join(bl["notes"]))
            units.append(bl["text"])  # raw text preserves numbers that light_clean strips as dims
    typed  = FLD.type_units(units)
    vendor = raw.get("vendor") or X.guess_vendor(stem)

    # schema partition (identical to build_signal.py)
    unclassified_blocks, tagged_notes, admin_cells = [], [], []
    for c in box_texts:
        named, stds = _named(c["text"])
        if named:
            continue
        rec = {"text": c["text"], "page": c.get("page"), "bbox": c["bbox_pt"]}
        if FLD.is_admin(c["text"]) or FLD.is_person(c["text"]):
            admin_cells.append(rec); continue
        ctags = VT.tag_unit(" ".join(light_clean(c["text"])[0]))
        if ctags or stds:
            tagged_notes.append({**rec, "src": "block", "tags": ctags})
        else:
            unclassified_blocks.append(rec)

    unclassified_body, noise_blocks, dims_all = [], [], []
    for b in body:
        rec = {"text": b["text"], "page": b.get("page"), "bbox": b.get("bbox")}
        if b.get("dims"):
            dims_all.append({"page": b.get("page"), "dims": b["dims"]})
        if b["noise"]:
            noise_blocks.append({**rec, "notes": b["notes"]}); continue
        named, stds = _named(b["text"])
        if named:
            tagged_notes.append({**rec, "src": "body", "notes": b["notes"], "tags": b["tags"]})
            continue
        btext = " ".join(b["notes"]) or b["text"]
        if FLD.is_admin(btext):
            admin_cells.append({**rec, "notes": b["notes"]}); continue
        if b["tags"] or named or stds:
            tagged_notes.append({**rec, "src": "body", "notes": b["notes"], "tags": b["tags"]})
        else:
            unclassified_body.append({**rec, "notes": b["notes"]})

    sig = {
        "stem": stem, "route": mode, "pymupdf_words": wc, "vendor": vendor,
        "n_pages": len(pages_raw),
        "page_routes": [{"page": p["page"], "route": p["mode"], "words": p["wc"]} for p in per],
        "classified": {
            "units":  _to_units(typed, tagged_notes),
            "fields": typed,
        },
        "unclassified": {
            "blocks": unclassified_blocks,
            "body":   unclassified_body,
        },
        "bom": {
            "header": bom_head, "cells": bom_text_cells, "rows": rows,
        },
        "debug": {
            "zone_markers": zone_markers,
            "admin":        admin_cells,
            "noise_blocks": noise_blocks,
            "dropped_blocks": dropped_blocks,
            "dims":         dims_all,
        },
        "counts": {
            "cells":               n_cells,
            "tagged_notes":        len(tagged_notes),
            "unclassified_blocks": len(unclassified_blocks),
            "unclassified_body":   len(unclassified_body),
            "admin":               len(admin_cells),
            "bom_rows":            len(rows),
            "bom_cells":           len(bom_text_cells),
            "noise":               len(noise_blocks),
            "crosspage_dups":      n_dups[0],
        },
    }

    out_dir = OUT / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SIG_NAME).write_text(json.dumps(sig, indent=1, ensure_ascii=False))
    return sig


# ---------------------------------------------------------------------------
# Parity check — diff new output vs existing signal.json
# ---------------------------------------------------------------------------

def _flatten(obj, prefix=""):
    """Flatten nested dict/list into dot-path → value for diffing."""
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            items.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        items[prefix] = obj
    return items


def parity_check(stems):
    mode_label = "all-pages" if ALL_PAGES else "page-1"
    print(f"Parity check on {len(stems)} stems ({mode_label}) vs existing {SIG_NAME}\n")
    all_ok = True
    for stem in stems:
        raw_path = OUT / stem / "raw_pages.json"
        old_path = OUT / stem / SIG_NAME
        if not raw_path.exists():
            print(f"  SKIP {stem[:50]} — no raw_pages.json"); continue
        if not old_path.exists():
            print(f"  SKIP {stem[:50]} — no signal.json to compare"); continue

        new_sig = classify(stem)
        old_sig = json.loads(old_path.read_text())

        new_flat = _flatten(new_sig)
        old_flat = _flatten(old_sig)

        diffs = []
        all_keys = set(new_flat) | set(old_flat)
        for k in sorted(all_keys):
            nv = new_flat.get(k, "<MISSING>")
            ov = old_flat.get(k, "<MISSING>")
            if nv != ov:
                diffs.append((k, ov, nv))

        if not diffs:
            print(f"  OK  {stem[:60]}")
        else:
            all_ok = False
            print(f"  DIFF {stem[:55]}  ({len(diffs)} mismatches)")
            for k, ov, nv in diffs[:8]:
                print(f"       {k}")
                print(f"         old: {str(ov)[:80]}")
                print(f"         new: {str(nv)[:80]}")
            if len(diffs) > 8:
                print(f"       ... and {len(diffs)-8} more")

    print("\nPARITY: " + ("ALL MATCH" if all_ok else "DIFFERENCES FOUND — investigate before migrating"))
    return all_ok


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parity_mode = "--parity" in sys.argv
    stem_args   = [a for a in sys.argv[1:] if not a.startswith("--")]

    if parity_mode:
        if not stem_args:
            # default parity sample
            stem_args = ["3AUA0000038918", "3AUA0000074295"]
        parity_check(stem_args)
        return

    if stem_args:
        stems = stem_args
    else:
        stems = sorted(
            d.name for d in OUT.iterdir()
            if d.is_dir() and (d / "raw_pages.json").exists()
        )

    print(f"classify_signal ({SIG_NAME}): {len(stems)} stems", flush=True)
    t0 = time.time()
    ok = err = 0
    for i, stem in enumerate(stems, 1):
        try:
            sig = classify(stem)
            ok += 1
            n_u = len(sig["classified"]["units"])
            n_b = len(sig["unclassified"]["body"])
            n_r = len(sig["bom"]["rows"])
            print(f"  [{i:4d}/{len(stems)}] {stem[:48]:50s} "
                  f"units={n_u:>3} body={n_b:>3} bom={n_r:>2}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERR {stem[:55]}: {type(e).__name__}: {e}", flush=True)
            err += 1
        if i % 100 == 0:
            dt = time.time() - t0
            rate = i / dt if dt else 0
            print(f"  [{i}/{len(stems)}] {rate:.2f}/s  ETA {(len(stems)-i)/rate/60:.1f}m", flush=True)

    print(f"\nDONE ok={ok} err={err} in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
