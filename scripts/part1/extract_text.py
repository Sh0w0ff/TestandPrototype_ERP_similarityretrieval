"""
Part 1 textual-channel extractor.

Reads cached  cache/preprocess/<stem>/text.txt  (produced by preprocess.py)
and writes a structured sidecar  cache/preprocess/<stem>/extracted.json:

  {
    "stem": "...",
    "vendor_guess": "ABB" | "KC",
    "standards":  [ {family, number, suffix, context}, ... ],
    "fields":     { "MATERIAL": "...", "GEN TOL": "...", ... },   # raw uppercased keys
    "treatments": [ "HDG", "ZINC", ... ],
    "ral_codes":  [ "7021", ... ],
    "n_pages":    int,
    "n_chars":    int,
  }

Design: cast a wide net, no narrow commitments. Downstream code (similarity,
BOM generation, work-phase classification) selects whichever subset it needs.

Usage:
    python extract_text.py                 # all cached PDFs
    python extract_text.py <stem> ...      # specific cache dirs by stem
    python extract_text.py --force         # ignore existing sidecars
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from text_utils import clean_field_value, raw_tokens, rejoin_continuations

# Spatial-pairing heuristics — strict for now, easy to relax later.
# Geometry thresholds expressed as fractions of page width/height so they
# work across A3 (ABB) and A2 (KC) drawings without per-vendor tuning.
LABEL_MAX_CHARS = 30
LABEL_MIN_UPPER_FRAC = 0.8
DY_MAX_FRAC = 0.04   # value block must start within 4% page height below label
DX_MAX_FRAC = 0.08   # or within 8% page width to the right
XY_OVERLAP_FRAC = 0.3  # label & value column/row must overlap by ≥30%

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/ -> paths.py
import paths
CACHE = paths.PREPROCESS

STANDARDS_FAMILIES = ("ISO", "DIN", "EN", "ASTM", "GB", "ANSI", "JIS", "BS", "NF", "SFS")
STD_PAT = re.compile(
    r"\b(" + "|".join(STANDARDS_FAMILIES) + r")[\s-]?"
    r"(\d{2,6}[A-Za-z]?)"
    r"(?:[-/]([A-Za-z0-9]{1,6}))?",
    re.I,
)

# Title-block "Key: value" — left side is a short label of letters/spaces,
# right side starts with non-whitespace and goes to end of line.
KEYVAL_PAT = re.compile(r"(?m)^[ \t]*([A-Za-z][A-Za-z. /-]{1,30}?)\s*:\s+(\S[^\n]{0,80})")

# Inline RAL color code (not always inside a labelled field).
RAL_PAT = re.compile(r"RAL[\s-]?(\d{4})", re.I)

# Surface / heat-treatment keywords — presence-only signal (work-phase hint).
TREATMENT_KEYWORDS = (
    "HDG", "GALVANIZ", "ZINC", "PAINTED", "COATED", "ANODIZ", "HEAT TREAT",
    "POWDER COAT", "PRIMER", "CHROMATE", "PASSIVAT", "ELECTROPLATE",
    "SANDBLAST", "SHOT BLAST", "PICKL", "POLISH", "WELDED", "BRAZED",
)


def guess_vendor(stem: str) -> str:
    # Filename prefixes observed in the corpus. ABB uses 3AUA / 3AXD / etc.;
    # KC uses QD / SQD / SQCEX / numeric. Refine later if a mis-classification surfaces.
    return "ABB" if stem.upper().startswith("3A") else "KC"


def extract_standards(text: str) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict] = []
    for m in STD_PAT.finditer(text):
        fam = m.group(1).upper()
        num = m.group(2)
        suf = m.group(3) or ""
        key = (fam, num, suf)
        if key in seen:
            continue
        seen.add(key)
        start = max(0, m.start() - 5)
        end = min(len(text), m.end() + 30)
        out.append({
            "family": fam,
            "number": num,
            "suffix": suf or None,
            "context": text[start:end].replace("\n", " ").strip()[:80],
        })
    return out


def extract_fields(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in KEYVAL_PAT.findall(text):
        k_norm = k.strip().upper()
        # First-occurrence wins; multi-page repeats are deduped.
        if k_norm not in out:
            out[k_norm] = clean_field_value(v)
    return out


def extract_ral(text: str) -> list[str]:
    return sorted({c for c in RAL_PAT.findall(text)})


def extract_treatments(text_upper: str) -> list[str]:
    return [kw for kw in TREATMENT_KEYWORDS if kw in text_upper]


def is_label_block(text: str) -> bool:
    """Strict label detector — short text, mostly uppercase letters, not numeric.

    No hardcoded vocabulary. Generalisation-first; missed labels surface as
    'unknown_token' downstream rather than being silently mis-paired.
    """
    t = text.strip().replace("\n", " ").strip()
    if not (2 <= len(t) <= LABEL_MAX_CHARS):
        return False
    alpha = [c for c in t if c.isalpha()]
    if not alpha:
        return False
    upper_frac = sum(1 for c in alpha if c.isupper()) / len(alpha)
    return upper_frac >= LABEL_MIN_UPPER_FRAC


def _overlap_frac(a0: float, a1: float, b0: float, b1: float) -> float:
    """Fraction of the shorter span that overlaps the other (0..1)."""
    span = min(a1 - a0, b1 - b0)
    if span <= 0:
        return 0.0
    return max(0.0, min(a1, b1) - max(a0, b0)) / span


def pair_blocks_spatial(blocks: list[dict], page_sizes: list[tuple[float, float]]) -> dict[str, str]:
    """For each label block, pair with the nearest block below or right.

    Returns a flat dict label_text -> value_text. Multi-page drawings collapse
    keys across pages (first-page value wins on collision).

    DEPRECATED IN PIPELINE 2026-05-24 (§9.8.8): output still emitted to
    `extracted.json.fields_spatial` for debugging / future re-enable, but
    `extract_drawing.py` no longer consumes it. Whole-page pymupdf-block
    pairing proved unreliable (wrong-cell pairings on KC three-cell title
    blocks, compounded by substring label-match in the consumer). The
    spatial channel will be re-enabled when per-view + title-block OCR
    lands (§9.7 strategies B + C + Path X) with cleaner cell input.
    """
    out: dict[str, str] = {}
    by_page: dict[int, list[dict]] = {}
    for b in blocks:
        by_page.setdefault(b["page"], []).append(b)

    for page_no, page_blocks in by_page.items():
        if page_no - 1 >= len(page_sizes):
            continue
        pw, ph = page_sizes[page_no - 1]
        dy_max = DY_MAX_FRAC * ph
        dx_max = DX_MAX_FRAC * pw

        for label in page_blocks:
            if not is_label_block(label["text"]):
                continue
            lx0, ly0, lx1, ly1 = label["bbox"]

            best = None  # (distance, value_text)
            for cand in page_blocks:
                if cand is label:
                    continue
                cx0, cy0, cx1, cy1 = cand["bbox"]

                # Below: candidate starts below label, x-ranges overlap enough.
                dy = cy0 - ly1
                if 0 <= dy <= dy_max and _overlap_frac(lx0, lx1, cx0, cx1) >= XY_OVERLAP_FRAC:
                    if best is None or dy < best[0]:
                        best = (dy, cand["text"])

                # Right: candidate starts right of label, y-ranges overlap enough.
                dx = cx0 - lx1
                if 0 <= dx <= dx_max and _overlap_frac(ly0, ly1, cy0, cy1) >= XY_OVERLAP_FRAC:
                    if best is None or dx < best[0]:
                        best = (dx, cand["text"])

            if best is None:
                continue
            label_key = label["text"].strip().replace("\n", " ").strip()
            value = best[1].strip().replace("\n", " ").strip()
            if label_key and value and label_key not in out:
                out[label_key] = value
    return out


def extract_for(stem_dir: Path) -> dict:
    text_path = stem_dir / "text.txt"
    text_raw = text_path.read_text(errors="ignore")
    text = rejoin_continuations(text_raw)
    meta = json.loads((stem_dir / "meta.json").read_text())
    stem = meta["stem"]

    blocks_path = stem_dir / "blocks.json"
    blocks = json.loads(blocks_path.read_text()) if blocks_path.exists() else []
    page_sizes = meta.get("page_sizes", [])
    fields_spatial = pair_blocks_spatial(blocks, page_sizes) if blocks else {}

    return {
        "stem": stem,
        "vendor_guess": guess_vendor(stem),
        "standards": extract_standards(text),
        "fields": extract_fields(text),
        "fields_spatial": fields_spatial,
        "treatments": extract_treatments(text.upper()),
        "ral_codes": extract_ral(text),
        "raw_tokens": raw_tokens(text),
        "n_pages": meta["page_count"],
        "n_chars": len(text),
    }


def iter_targets(args) -> list[Path]:
    if args.stems:
        return [CACHE / s for s in args.stems]
    return sorted(p for p in CACHE.iterdir() if p.is_dir())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stems", nargs="*", help="specific cache-dir stems (default: all)")
    ap.add_argument("--force", action="store_true", help="overwrite existing extracted.json")
    args = ap.parse_args()

    targets = iter_targets(args)
    if not targets:
        ap.error(f"no cache dirs found under {CACHE}")

    n_done = n_skip = n_err = 0
    for d in targets:
        out_path = d / "extracted.json"
        if not args.force and out_path.exists():
            n_skip += 1
            continue
        try:
            record = extract_for(d)
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
            n_done += 1
        except Exception as e:
            n_err += 1
            print(f"  ERR  {d.name}  {type(e).__name__}: {e}")

    print(f"[extract_text] done={n_done} skip={n_skip} err={n_err}  "
          f"(cache={CACHE})")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
