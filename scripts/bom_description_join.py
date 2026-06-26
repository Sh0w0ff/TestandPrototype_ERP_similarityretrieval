"""
Description-aware BOM join — second pass after item-code join.

Origin: 2026-05-13. User spotted that unjoined PDF D113369 appears at BOM row 4926
as a *component* (item code 42993827) whose description embeds the string
"D113369-A/2--SHAFT...". The item-code-only join misses these because the drawing ID
is in the description field, not the item code.

Method:
  1. Pull the existing 58 'unjoined' PDFs from /tmp/unjoined_list.txt.
  2. For each, extract drawing-ID candidate tokens from the filename via the same
     tokeniser logic as bom_coverage_check.py.
  3. Search BOM's `Parent part description` and `Component description` columns
     for each candidate token using *boundary-respecting* substring match (token
     must be flanked by non-alphanumeric chars or string ends) to avoid false
     positives like "117" matching "1170".
  4. Report: per-PDF whether a match exists, which BOM row(s), which ERP item code
     it maps to, and the final residual list of genuinely orphan PDFs.

Output: /tmp/bom_description_join.tsv  +  /tmp/orphan_pdfs.txt
"""
import csv, re
from collections import defaultdict
from pathlib import Path

UNJOINED = Path("/tmp/unjoined_list.txt")
BOM_CSV  = Path("/Users/sh0w0ff/FYP/Bill_of_Materials.csv")
OUT_TSV  = Path("/tmp/bom_description_join.tsv")
ORPHANS  = Path("/tmp/orphan_pdfs.txt")

MIN_TOKEN_LEN = 5  # avoid "DRW1" / "1" etc.


def candidate_tokens(filename: str) -> set[str]:
    """Mirror of bom_coverage_check.py:tokens_from_filename heuristic — extract plausible drawing-ID tokens."""
    stem = Path(filename).stem
    out: set[str] = set()
    out.add(stem)
    # split by space/_/;
    for piece in re.split(r"[\s_;]+", stem):
        if piece:
            out.add(piece)
            # dash-range substrings of length>=2 components
            parts = piece.split("-")
            for i in range(len(parts)):
                for j in range(i + 1, len(parts) + 1):
                    sub = "-".join(parts[i:j])
                    if sub:
                        out.add(sub)
    # parenthesised content
    for m in re.findall(r"\(([^)]+)\)", stem):
        out.add(m)
    # explicit drawing-id shapes
    out.update(re.findall(r"3A[UX][AD]\d+", stem))               # ABB
    out.update(re.findall(r"D\d{4,}[-A-Z0-9]*", stem))           # KC D-series
    out.update(re.findall(r"\d{6,}-DRW\d+", stem))               # KC modern
    return {t for t in out if len(t) >= MIN_TOKEN_LEN}


def boundary_search(haystack: str, needle: str) -> bool:
    """needle appears in haystack with non-alphanumeric flanks (or string ends)."""
    pat = r"(?:^|[^A-Za-z0-9])" + re.escape(needle) + r"(?:[^A-Za-z0-9]|$)"
    return re.search(pat, haystack) is not None


def load_unjoined() -> list[str]:
    out = []
    for line in UNJOINED.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("---") or line.startswith("TOTAL"):
            continue
        out.append(line)
    return out


def load_bom() -> list[dict]:
    rows = []
    with open(BOM_CSV, encoding="utf-8", errors="replace") as f:
        rdr = csv.DictReader(f, delimiter=";")
        for i, r in enumerate(rdr):
            rows.append({
                "row": i + 2,  # +1 for header, +1 for 1-based
                "parent": (r.get("Parent part") or "").strip(),
                "parent_desc": (r.get("Parent part description") or "").strip(),
                "component": (r.get("Component") or "").strip(),
                "component_desc": (r.get("Component description") or "").strip(),
            })
    return rows


def main():
    unjoined = load_unjoined()
    bom = load_bom()
    print(f"Unjoined PDFs: {len(unjoined)}")
    print(f"BOM rows: {len(bom)}")

    matched_pdfs: list[tuple[str, str, list[dict]]] = []   # (pdf, token, hits)
    orphans: list[str] = []

    for pdf in unjoined:
        toks = candidate_tokens(pdf)
        # try longest tokens first — more specific
        toks_sorted = sorted(toks, key=len, reverse=True)
        best_hits: list[dict] = []
        best_tok = ""
        for tok in toks_sorted:
            hits = []
            for r in bom:
                if (boundary_search(r["parent_desc"], tok)
                    or boundary_search(r["component_desc"], tok)):
                    hits.append(r)
                    if len(hits) >= 5:  # cap; we only need to know it joined + a few examples
                        break
            if hits:
                best_hits = hits
                best_tok = tok
                break
        if best_hits:
            matched_pdfs.append((pdf, best_tok, best_hits))
        else:
            orphans.append(pdf)

    # Write detailed TSV
    with open(OUT_TSV, "w") as f:
        f.write("pdf\ttoken_matched\tbom_row\tparent_item\tcomponent_item\tparent_desc_snippet\tcomponent_desc_snippet\n")
        for pdf, tok, hits in matched_pdfs:
            for h in hits:
                f.write(f"{pdf}\t{tok}\t{h['row']}\t{h['parent']}\t{h['component']}\t"
                        f"{h['parent_desc'][:80]}\t{h['component_desc'][:80]}\n")

    # Write orphans
    ORPHANS.write_text("\n".join(orphans) + ("\n" if orphans else ""))

    # ERP-code recovery summary: for matched PDFs, what ERP codes do they point at?
    erp_recovered = defaultdict(set)
    for pdf, tok, hits in matched_pdfs:
        for h in hits:
            # If token appeared in component_desc, the ERP code is `component`; if in parent_desc, it's `parent`.
            if boundary_search(h["component_desc"], tok) and h["component"]:
                erp_recovered[pdf].add(("component", h["component"]))
            if boundary_search(h["parent_desc"], tok) and h["parent"]:
                erp_recovered[pdf].add(("parent", h["parent"]))

    print(f"\n=== Description-field join recovery ===")
    print(f"Matched: {len(matched_pdfs)}/{len(unjoined)}")
    print(f"Orphan : {len(orphans)}/{len(unjoined)}")
    print(f"\nPDFs with a recovered ERP code: {sum(1 for v in erp_recovered.values() if v)}")
    print(f"\nGenuine orphans (no description-field match in BOM):")
    for o in orphans:
        print(f"  {o}")

    print(f"\nWrote {OUT_TSV} and {ORPHANS}")

    # Sample examples
    print(f"\n--- 3 sample recoveries ---")
    for pdf, tok, hits in matched_pdfs[:3]:
        print(f"\n{pdf}")
        print(f"  token: {tok}")
        for h in hits[:1]:
            print(f"  row {h['row']}: parent={h['parent']!r} component={h['component']!r}")
            print(f"    parent_desc:    {h['parent_desc'][:100]}")
            print(f"    component_desc: {h['component_desc'][:100]}")


if __name__ == "__main__":
    main()
