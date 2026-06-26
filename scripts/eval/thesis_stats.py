"""thesis_stats.py — corpus/ERP/label statistics for thesis tables + charts.
Read-only. Prints clean blocks the thesis tables/pgfplots charts are built from.
"""
import sys, json, glob, collections, statistics as st
from pathlib import Path
ROOT = Path("/Users/sh0w0ff/FYP")
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import paths
from erp_truth import ErpTruth

sig_files = sorted(glob.glob(str(paths.TEXT_PIPE / "*" / "signal.json")))
stems = [Path(f).parent.name for f in sig_files]
N = len(stems)

# ---- corpus composition ----
vend = collections.Counter(); pages = collections.Counter(); words = []
emb_bom_rows = collections.Counter(); emb_bom_any = collections.Counter(); rowcounts = []
multipage = 0
for f in sig_files:
    s = json.load(open(f))
    v = s.get("vendor", "?"); vend[v] += 1
    npg = s.get("n_pages", 1) or 1
    if npg > 1: multipage += 1
    w = s.get("pymupdf_words")
    if isinstance(w, int): words.append(w)
    b = s.get("bom", {}) or {}
    rows = b.get("rows") or []
    if rows:
        emb_bom_rows[v] += 1; rowcounts.append(len(rows))
print("=== CORPUS COMPOSITION ===")
print(f"total signal.json: {N}")
for v in sorted(vend): print(f"  vendor {v}: {vend[v]}  ({100*vend[v]/N:.1f}%)")
print(f"multipage drawings: {multipage} ({100*multipage/N:.1f}%)")
if words:
    words.sort()
    qs = [words[int(p*len(words))] for p in (0.25,0.5,0.75)]
    print(f"pymupdf words/drawing: min {min(words)} q25 {qs[0]} median {qs[1]} q75 {qs[2]} max {max(words)} (n={len(words)})")
    le50 = sum(1 for x in words if x<=50)
    print(f"  drawings <=50 words (OCR-fallback candidates): {le50} ({100*le50/len(words):.1f}%)")

print("\n=== EMBEDDED BOM TABLE (on-drawing) ===")
for v in sorted(vend):
    tot=vend[v]; hr=emb_bom_rows[v]
    print(f"  {v}: {hr}/{tot} have parsed BOM rows ({100*hr/tot:.1f}%)")
if rowcounts:
    print(f"  rows/embedded-table: median {st.median(rowcounts):.0f} mean {st.mean(rowcounts):.1f} max {max(rowcounts)}")

# ---- ERP linkage + BOM ----
erp = ErpTruth()
has_item=has_comp=multi=0; ncomp=[]; parents=set(); comps=set()
import csv
for r in csv.DictReader(open(ROOT/'Bill_of_Materials.csv',encoding='latin-1'),delimiter=';'):
    p=r['Parent part'].strip(); c=r['Component'].strip()
    if p: parents.add(p)
    if c: comps.add(c)
for sstem in stems:
    items=erp.stem_to_items(sstem); cs=erp.stem_to_components(sstem)
    if items: has_item+=1
    if len(items)>1: multi+=1
    if cs: has_comp+=1; ncomp.append(len(cs))
print("\n=== ERP LINKAGE ===")
print(f"  >=1 item: {has_item} ({100*has_item/N:.1f}%)")
print(f"  >=1 component (usable BOM): {has_comp} ({100*has_comp/N:.1f}%)")
print(f"  multi-item (ambiguous): {multi} ({100*multi/N:.1f}%)")
if ncomp:
    ncomp.sort()
    print(f"  components/drawing: median {st.median(ncomp):.0f} mean {st.mean(ncomp):.1f} max {max(ncomp)}")
    # histogram buckets for a chart
    buckets = collections.Counter()
    for x in ncomp:
        b = '1' if x==1 else '2-3' if x<=3 else '4-7' if x<=7 else '8-15' if x<=15 else '16+'
        buckets[b]+=1
    print("  component-count histogram:", {k:buckets[k] for k in ['1','2-3','4-7','8-15','16+']})
top = parents - comps
print(f"  top-level (parent never component): {len(top)} = {100*len(top)/len(parents):.1f}% of parents; {100*len(top)/len(parents|comps):.1f}% of all BOM items")

# ---- BOM long-tail (singletons) ----
comp_freq = collections.Counter()
for r in csv.DictReader(open(ROOT/'Bill_of_Materials.csv',encoding='latin-1'),delimiter=';'):
    c=r['Component'].strip()
    if c: comp_freq[c]+=1
singletons = sum(1 for c,n in comp_freq.items() if n==1)
print(f"\n=== BOM LONG TAIL ===\n  distinct components: {len(comp_freq)}  singletons: {singletons} ({100*singletons/len(comp_freq):.1f}%)")

# ---- production-line (G3 label) frequency over linked drawings ----
import phase_classify as pc
spl = pc.load_prod_lines()
plf = collections.Counter()
for sstem in stems:
    for pl in spl.get(sstem, set()):
        plf[pl]+=1
print("\n=== PRODUCTION-LINE FREQUENCY (drawings routing through each) ===")
for pl,c in plf.most_common():
    print(f"  {pl:12s} {c}")
