"""textsim.py — shared text-similarity helpers (DECOUPLED from the product-structure pipeline).

Builds a per-drawing text that is RAW cell text (all pages) with one unsupervised, label-free
cleanup so a dense encoder isn't swamped by chrome:
  - drop zone markers (is_zone).
  - drop corpus-BOILERPLATE lines: any normalized line whose document-frequency exceeds DF_MAX
    (the identical ABB legal block / KC title-block labels). This is the dense-model analogue of
    TF-IDF's IDF down-weighting, done explicitly so SimCSE doesn't learn vendor identity.
  - KEEP embedded-BOM cells: the on-drawing BOM does NOT cleanly join the ERP product-structure
    target (established), so it is legitimate drawing text here, not a target leak.

No fingerprint, no typed fields, no vocab tags — this is intentionally NOT the e1 representation.
"""
import json, re, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
TEXT_PIPE = ROOT / "text_pipe"
DF_MAX = 0.30          # lines appearing in >30% of drawings = boilerplate, removed
CACHE = ROOT / "cache" / "textsim_corpus.json"


def _norm(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def _stem_lines(stem):
    """raw, deduped, non-zone, non-BOM cell lines for one drawing (all pages)."""
    f = TEXT_PIPE / stem / "raw_pages.json"
    if not f.exists():
        return []
    d = json.load(open(f))
    seen, out = set(), []
    for p in d.get("pages", []):
        for c in p.get("cells", []):
            if c.get("is_zone"):
                continue
            t = (c.get("text") or "").strip()
            n = _norm(t)
            if n and n not in seen:
                seen.add(n); out.append((n, t))
    return out


def build_corpus(df_max=DF_MAX, rebuild=False):
    """returns {stem: cleaned_text}. Caches to CACHE."""
    if CACHE.exists() and not rebuild:
        return json.load(open(CACHE))
    stems = sorted(d.name for d in TEXT_PIPE.iterdir() if (d / "raw_pages.json").exists())
    lines = {s: _stem_lines(s) for s in stems}
    df = collections.Counter()
    for s in stems:
        for n, _ in lines[s]:
            df[n] += 1
    nmax = df_max * len(stems)
    corpus = {}
    for s in stems:
        kept = [orig for n, orig in lines[s] if df[n] <= nmax]
        corpus[s] = " ".join(kept)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(corpus, open(CACHE, "w"))
    return corpus


if __name__ == "__main__":
    import sys
    corp = build_corpus(rebuild="--rebuild" in sys.argv)
    print(f"built cleaned text corpus: {len(corp)} drawings -> {CACHE}")
    lens = sorted(len(v) for v in corp.values())
    print(f"  text length chars: min {lens[0]}  median {lens[len(lens)//2]}  max {lens[-1]}")
    for s in ["3AXD50000325719", "SQDE-L384.5W377H376-7021_SUPPORT_69932539_2_69932539-DRW1"]:
        if s in corp:
            print(f"\n--- {s} (cleaned) ---\n{corp[s][:400]}")
