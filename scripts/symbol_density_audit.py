"""
Symbol corpus density audit (2026-05-13).

Goal: count how many of the 1829 PDFs contain canonical text-extractable references
to each symbol class, per vendor. Tells us whether template-matching-on-rasters
is worth feasibility-testing (sub-B of §9.4) — i.e. is there enough actual symbol
content in the drawings, or are they mostly bare schematics?

Two-step:
  1. Cache `pdftotext` output for all 1829 PDFs to /tmp/pdftotext_cache/<hash>.txt
     (this artefact is reused by Part 1's preprocess.py later).
  2. Grep each cached output for canonical patterns per symbol class.
     Report: per-class count of PDFs with ≥1 match, broken down ABB / KC.

Symbol classes searched (TEXT proxy only — actual symbol detection will be visual):
  - welding_iso2553  : "ISO 2553" callouts
  - surface_finish   : Ra/Rz values, Sa blasting class, "ISO 1302/21920", Finnish "Pintakäsittely"
  - iso_2768         : general tolerance "ISO 2768" / "2768-m/f/v/c"
  - gd_t             : Unicode geometric-tolerance glyphs (⌖⊥∥▱◎⌒∠) and "Datum"
  - ral_color        : "RAL ####" — surface-treatment indicator
  - material_grade   : steel grades S###, EN 10025, AISI, etc.
  - diameter_signs   : "⌀" / "Ø" (sanity-check that *some* glyph extraction works)
  - finnish_present  : Finnish keywords ("Pintakäsittely", "Karheus", "Värisävy", "Mittakaava") — bilingual coverage check
"""
import csv, hashlib, re, subprocess
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PDF_DIR  = Path("/Users/sh0w0ff/FYP/PDF drawings")
CACHE    = Path("/tmp/pdftotext_cache")
CACHE.mkdir(exist_ok=True)

PDFS = sorted([p for p in PDF_DIR.glob("*.pdf")] + [p for p in PDF_DIR.glob("*.PDF")])
print(f"Total PDFs: {len(PDFS)}")


def is_abb(name: str) -> bool:
    return name.startswith(("3AUA", "3AXD"))


def cached_path(pdf: Path) -> Path:
    h = hashlib.sha1(pdf.name.encode("utf-8")).hexdigest()[:16]
    return CACHE / f"{h}.txt"


def extract_text(pdf: Path) -> bool:
    """Run pdftotext if not cached. Returns True on success."""
    out = cached_path(pdf)
    if out.exists() and out.stat().st_size > 0:
        return True
    try:
        subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), str(out)],
            check=True, capture_output=True, timeout=20,
        )
        return out.exists()
    except Exception as e:
        # write an empty marker so we don't retry
        out.write_text("")
        return False


print(f"Caching pdftotext output → {CACHE}/")
done = 0
with ThreadPoolExecutor(max_workers=8) as pool:
    for ok in pool.map(extract_text, PDFS):
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(PDFS)}")
print(f"  cached: {done}/{len(PDFS)}")


# ---- Pattern set ----
# Order matters only for the report. Use re.IGNORECASE except where casing is meaningful (RAL, S235).
PATTERNS: dict[str, list[str]] = {
    "welding_iso2553": [r"ISO[\s\-]*2553"],
    "surface_finish":  [r"ISO[\s\-]*1302", r"ISO[\s\-]*21920",
                        r"\bR[az]\s*[<=≤]?\s*\d+(\.\d+)?\b",
                        r"\bSa\s*2[½\.5]?\b",
                        r"Pintakäsittely", r"Karheus", r"Roughness"],
    "iso_2768":        [r"ISO[\s\-]*2768", r"\b2768-[mfvcMFVC]\b"],
    "gd_t":            [r"[⌖⌭⊥∥▱◎⌒∠]", r"\bDatum\b", r"\bGD&T\b",
                        r"\bPosition tol\b", r"\bFlatness\b", r"\bParallelism\b",
                        r"\bPerpendicularity\b"],
    "ral_color":       [r"\bRAL\s*\d{4}\b"],
    "material_grade":  [r"\bS\s*2[35]5\b", r"\bS\s*355\b", r"\bEN\s*10025\b",
                        r"\bAISI\s*\d{3}\b", r"\b1\.\d{4}\b",
                        r"\bDX\d{2,3}D[+\-][A-Z0-9\-+]+",     # DX51D+Z275-M-A-C etc.
                        r"\bHDG\s*STEEL", r"\bGalvani[sz]ed\b",
                        r"\bStainless\s*Steel", r"\bAluminium\b", r"\bAluminum\b",
                        r"Material:", r"Materiaali:"],
    "diameter_signs":  [r"[⌀Øø]", r"\bDIA\b", r"\bDiameter\b",
                        r"\bD\d{2,3}\b(?=[\s,\-\.])"],         # D50, D100 shaft codes
    "finnish_present": [r"Pintakäsittely", r"Värisävy", r"Mittakaava",
                        r"Karheus", r"Materiaali"],
}

# Case-sensitive sets (so RAL doesn't pick up "ral" inside descriptions).
CASE_SENSITIVE = {"ral_color", "material_grade"}

compiled = {}
for cls, pats in PATTERNS.items():
    flags = 0 if cls in CASE_SENSITIVE else re.IGNORECASE
    compiled[cls] = [re.compile(p, flags) for p in pats]


# ---- Scan ----
hits_abb: dict[str, int] = defaultdict(int)
hits_kc:  dict[str, int] = defaultdict(int)
text_byte_buckets = Counter()
empty_text = 0

per_pdf_hits: list[tuple[str, dict[str, int]]] = []

for pdf in PDFS:
    txt_path = cached_path(pdf)
    try:
        text = txt_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        text = ""
    if not text.strip():
        empty_text += 1
        continue
    text_byte_buckets[
        "≥1k" if len(text) >= 1000 else
        "100-999" if len(text) >= 100 else "<100"
    ] += 1

    pdf_hits: dict[str, int] = {}
    for cls, regs in compiled.items():
        count = sum(len(r.findall(text)) for r in regs)
        if count > 0:
            pdf_hits[cls] = count
            (hits_abb if is_abb(pdf.name) else hits_kc)[cls] += 1
    per_pdf_hits.append((pdf.name, pdf_hits))


n_abb = sum(1 for p in PDFS if is_abb(p.name))
n_kc  = len(PDFS) - n_abb

print(f"\n=== Text extraction quality ===")
print(f"Empty text (extraction failed):     {empty_text}/{len(PDFS)} ({empty_text/len(PDFS)*100:.1f}%)")
for k, v in text_byte_buckets.most_common():
    print(f"  text size {k:<8}: {v}")

print(f"\n=== Symbol-class hit rates ({len(PDFS)} PDFs: {n_abb} ABB / {n_kc} KC) ===")
print(f"{'class':<18} {'ABB hit %':>11} {'KC hit %':>11} {'overall':>10}")
print("-" * 54)
for cls in PATTERNS:
    a, k = hits_abb[cls], hits_kc[cls]
    overall = a + k
    pa = a / n_abb * 100 if n_abb else 0
    pk = k / n_kc  * 100 if n_kc  else 0
    po = overall / len(PDFS) * 100
    print(f"{cls:<18} {pa:>9.1f}% {pk:>10.1f}% {po:>8.1f}%  ({a}+{k}={overall})")

# Save per-PDF hit map for downstream pair/cluster construction.
OUT_TSV = Path("/tmp/symbol_density.tsv")
with open(OUT_TSV, "w") as f:
    cols = list(PATTERNS.keys())
    f.write("pdf\tvendor\t" + "\t".join(cols) + "\n")
    for name, hits in per_pdf_hits:
        vendor = "ABB" if is_abb(name) else "KC"
        row = [name, vendor] + [str(hits.get(c, 0)) for c in cols]
        f.write("\t".join(row) + "\n")
print(f"\nWrote {OUT_TSV}")
