"""
Classify every PDF as top-level assembly / subassembly / leaf component /
uninstantiated, by propagating BOM parent-vs-component bookkeeping through the
now-complete PDF↔ERP linkage.

Buckets (per ERP item):
  - top_level     : appears in BOM as Parent only         (has children, nothing consumes it)
  - subassembly   : appears as both Parent and Component  (has children AND is consumed)
  - leaf          : appears as Component only             (atomic part / leaf in this snapshot)
  - uninstantiated: in Item_Basic_Data but in no BOM row  (catalogued but not produced here)

A PDF inherits the "richest" bucket among its items (top_level > subassembly >
leaf > uninstantiated), so revisions/variants don't get classified down.

Output:
  /tmp/pdf_buckets.tsv     — pdf<TAB>bucket<TAB>item_ids<TAB>example_parent_desc
  /tmp/item_buckets.tsv    — item<TAB>bucket
  console: distribution + sample of each bucket
"""
import csv, io, contextlib, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/Users/sh0w0ff/FYP/scripts")
with contextlib.redirect_stdout(io.StringIO()):
    import bom_coverage_check as bcc  # primary pass via item-code

# Merge in description-field second-pass recoveries so all 1829 PDFs get linked.
# The PDF *is* the item whose description carried the token — so add only that
# item (parent_item if token matched parent_desc, component_item if token matched
# component_desc), not both. Adding both would cause a SHAFT drawing whose token
# matched a row's component_desc to inherit the parent assembly's top_level bucket.
import re as _re
SECOND_PASS_TSV = Path("/tmp/bom_description_join.tsv")
if SECOND_PASS_TSV.exists():
    with open(SECOND_PASS_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            pdf = row["pdf"]
            tok = row["token_matched"]
            pat = _re.compile(r"(?:^|[^A-Za-z0-9])" + _re.escape(tok) + r"(?:[^A-Za-z0-9]|$)")
            if pat.search(row.get("parent_desc_snippet", "")):
                it = row["parent_item"].strip()
                if it: bcc.pdf_to_items[pdf].add(it)
            if pat.search(row.get("component_desc_snippet", "")):
                it = row["component_item"].strip()
                if it: bcc.pdf_to_items[pdf].add(it)

ITEM_CSV  = Path("/Users/sh0w0ff/FYP/Item_Basic_Data.csv")
BOM_CSV   = Path("/Users/sh0w0ff/FYP/Bill_of_Materials.csv")
OUT_PDF   = Path("/tmp/pdf_buckets.tsv")
OUT_ITEM  = Path("/tmp/item_buckets.tsv")

# Item universe = everything in Item_Basic_Data.csv (Stera's catalogue scope).
all_items: dict[str, str] = {}  # item_id -> description (for sampling)
with open(ITEM_CSV, encoding="latin-1") as f:
    for row in csv.DictReader(f, delimiter=";"):
        item = row["Item"].strip()
        if item:
            all_items[item] = (row.get("Item description") or "").strip()

# Parent / Component sets from BOM.
parents: set[str] = set()
components: set[str] = set()
with open(BOM_CSV, encoding="latin-1") as f:
    for row in csv.DictReader(f, delimiter=";"):
        p = row["Parent part"].strip()
        c = row["Component"].strip()
        if p: parents.add(p)
        if c: components.add(c)

# Classify each ERP item.
def bucket_of(item: str) -> str:
    is_p = item in parents
    is_c = item in components
    if is_p and is_c: return "subassembly"
    if is_p:          return "top_level"
    if is_c:          return "leaf"
    return "uninstantiated"

item_bucket: dict[str, str] = {it: bucket_of(it) for it in all_items}

# Also classify any items that appear in BOM but are missing from Item_Basic_Data
# (these exist — BOM references components that Stera doesn't catalogue separately,
# e.g. raw stock, externally-purchased parts). Tag them as `external`.
for it in (parents | components):
    if it not in item_bucket:
        item_bucket[it] = "external"
        all_items[it] = ""  # no description available

# Propagate to PDFs. A PDF inherits the strongest bucket among its items.
PRIORITY = {"top_level": 4, "subassembly": 3, "leaf": 2, "uninstantiated": 1, "external": 0}

pdf_bucket: dict[str, str] = {}
for fn in bcc.pdfs:
    items = bcc.pdf_to_items.get(fn, set())
    if not items:
        pdf_bucket[fn] = "no_link"
        continue
    best = max(items, key=lambda it: PRIORITY.get(item_bucket.get(it, "external"), 0))
    pdf_bucket[fn] = item_bucket.get(best, "external")

# Write item-level TSV.
with open(OUT_ITEM, "w") as f:
    f.write("item\tbucket\tdescription\n")
    for it, b in sorted(item_bucket.items()):
        f.write(f"{it}\t{b}\t{all_items.get(it, '')}\n")

# Write PDF-level TSV.
with open(OUT_PDF, "w") as f:
    f.write("pdf\tbucket\titem_ids\texample_desc\n")
    for fn in bcc.pdfs:
        b = pdf_bucket[fn]
        items = sorted(bcc.pdf_to_items.get(fn, []))
        items_str = ",".join(items) if items else ""
        desc = ""
        for it in items:
            d = all_items.get(it, "")
            if d:
                desc = d[:120]
                break
        f.write(f"{fn}\t{b}\t{items_str}\t{desc}\n")

# Report distributions.
def dist(d: dict[str, str]) -> dict[str, int]:
    out = defaultdict(int)
    for v in d.values(): out[v] += 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))

print("=== Item-level bucket distribution ===")
print(f"Total items in scope: {len(item_bucket)}")
for k, v in dist(item_bucket).items():
    print(f"  {k:<16} {v:>6}  ({v/len(item_bucket)*100:5.1f}%)")

print(f"\n=== PDF-level bucket distribution ({len(bcc.pdfs)} PDFs) ===")
for k, v in dist(pdf_bucket).items():
    print(f"  {k:<16} {v:>6}  ({v/len(bcc.pdfs)*100:5.1f}%)")

# Samples
print("\n=== Sample PDFs per bucket ===")
sample_by_bucket = defaultdict(list)
for fn, b in pdf_bucket.items():
    if len(sample_by_bucket[b]) < 3:
        sample_by_bucket[b].append(fn)
for b in ["top_level", "subassembly", "leaf", "uninstantiated", "external", "no_link"]:
    s = sample_by_bucket.get(b, [])
    if s:
        print(f"\n  [{b}]")
        for fn in s:
            items = sorted(bcc.pdf_to_items.get(fn, []))
            print(f"    {fn}  items={items}")

print(f"\nWrote {OUT_ITEM} and {OUT_PDF}")
