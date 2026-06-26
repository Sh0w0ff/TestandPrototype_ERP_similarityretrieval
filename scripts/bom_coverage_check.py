"""
Sanity check: how many of the 1829 PDFs have trainable BOM ground truth?

Pipeline:
  PDF filename --(token match against description prefix)--> Item ID
  Item ID --(lookup in Bill_of_Materials as Parent)--> list of component item IDs
"""
import csv, re
from pathlib import Path
from collections import defaultdict, Counter

PDF_DIR  = Path("/Users/sh0w0ff/FYP/PDF drawings")
ITEM_CSV = Path("/Users/sh0w0ff/FYP/Item_Basic_Data.csv")
BOM_CSV  = Path("/Users/sh0w0ff/FYP/Bill_of_Materials.csv")

pdfs = sorted([p.name for p in PDF_DIR.glob("*.pdf")] +
              [p.name for p in PDF_DIR.glob("*.PDF")])

DIGIT_ID_RE = re.compile(r"^\d{7,}$")
SUFFIX_VARIANTS = ("-DRW1", "-PDF1", "-NXD1", "-A", "-B",
                   "A", "B", "C", "D")  # no-dash suffix form e.g. 68229537B


def tokens_from_filename(fn):
    stem = re.sub(r"\.(pdf|PDF)$", "", fn)
    toks = {stem}
    # Pull content from parentheses: "...(DS015934-A2)..." -> DS015934-A2
    for m in re.findall(r"\(([^()]+)\)", stem):
        toks.add(m)
    # Split on _ ; whitespace
    for t in re.split(r"[_;\s]+", stem):
        if len(t) >= 4:
            toks.add(t)
        parts = t.split("-")
        if len(parts) >= 2:
            # All dash RANGE substrings (not just prefixes): handles "X-Y-A-1" -> "Y-A"
            for i in range(len(parts)):
                for j in range(i + 1, len(parts) + 1):
                    sub = "-".join(parts[i:j])
                    if len(sub) >= 4:
                        toks.add(sub)
                    # Trailing-digit strip on last part: "D113777-A3" -> "D113777-A"
                    last = parts[j - 1] if j > i else ""
                    stripped = re.sub(r"\d+$", "", last)
                    if stripped and stripped != last:
                        sub2 = "-".join(parts[i:j-1] + [stripped]) if j > i else stripped
                        if len(sub2) >= 4:
                            toks.add(sub2)
        # Synthetic suffix variants on pure-digit item IDs: "69738959" -> "69738959-DRW1" etc.
        for sub in list(toks):
            if DIGIT_ID_RE.match(sub):
                for suf in SUFFIX_VARIANTS:
                    toks.add(sub + suf)
    # Punctuation-insertion variants for compact KC-legacy filenames.
    # Source case: filename "6SAR03A1" but ERP description has "6SAR03-A1";
    # "7X1A01B12" vs "7X1A01/B12". Filenames drop the separator the ERP keeps.
    # Heuristic: at every digit->letter transition, generate "-" and "/" variants.
    for sub in list(toks):
        positions = [i for i in range(1, len(sub))
                     if sub[i-1].isdigit() and sub[i].isalpha()]
        for pos in positions:
            for sep in ("-", "/"):
                toks.add(sub[:pos] + sep + sub[pos:])
    return toks

token_to_files = defaultdict(list)
for fn in pdfs:
    for t in tokens_from_filename(fn):
        token_to_files[t].append(fn)

# Item_Basic_Data: build description-prefix → item
item_of_drawing = defaultdict(list)   # drawing token → list of item IDs
drawing_of_item = {}                  # item ID → drawing token
with open(ITEM_CSV, encoding="latin-1") as f:
    for row in csv.DictReader(f, delimiter=";"):
        item = row["Item"].strip()
        desc = row["Item description"] or ""
        # Index multiple head candidates per item:
        #   - "64616447/L--SECTIONING PLATE": short head "64616447" (split at /),
        #     long head "64616447/L" (split at --). Filenames usually carry the
        #     bare item id, so short head wins on the dominant case.
        #   - "7X1A01/B12--SUPPORT RAIL":     short head "7X1A01" (loses /B12),
        #     long head "7X1A01/B12". Filename "7X1A01B12.pdf" (separator-free)
        #     joins via dash/slash-insertion variant matching long head.
        # Indexing both keeps the dominant case unchanged and adds the rare one.
        heads = set()
        if "--" in desc:
            heads.add(desc.split("--", 1)[0].strip())
        heads.add(desc.split("/", 1)[0].strip())
        for head in heads:
            if head:
                item_of_drawing[head].append(item)
                drawing_of_item[item] = head

# Resolve each PDF → item IDs
pdf_to_items = defaultdict(set)
items_with_pdf = set()
for fn in pdfs:
    for t in tokens_from_filename(fn):
        for it in item_of_drawing.get(t, []):
            pdf_to_items[fn].add(it)
            items_with_pdf.add(it)

# BOM: parent → list of components
bom_children = defaultdict(list)
with open(BOM_CSV, encoding="latin-1") as f:
    for row in csv.DictReader(f, delimiter=";"):
        parent = row["Parent part"].strip()
        comp   = row["Component"].strip()
        if parent and comp:
            bom_children[parent].append(comp)

# How many PDFs have non-empty BOMs?
pdf_has_bom = 0
pdf_bom_sizes = []
pdf_components_drawn = []
for fn, items in pdf_to_items.items():
    children = []
    for it in items:
        children.extend(bom_children.get(it, []))
    if children:
        pdf_has_bom += 1
        pdf_bom_sizes.append(len(children))
        # How many of the children have their own drawing in the corpus?
        drawn_children = [c for c in children if c in items_with_pdf]
        pdf_components_drawn.append(len(drawn_children))

print(f"Total PDFs:                                 {len(pdfs)}")
print(f"PDFs joined to ≥1 ERP item:                 {len(pdf_to_items)}  ({len(pdf_to_items)/len(pdfs)*100:.1f}%)")
print(f"PDFs whose items appear as BOM Parent:      {pdf_has_bom}  ({pdf_has_bom/len(pdfs)*100:.1f}%)")
print()
print(f"BOM size distribution for those PDFs (components per parent):")
if pdf_bom_sizes:
    s = sorted(pdf_bom_sizes)
    print(f"  min/median/mean/max: {s[0]} / {s[len(s)//2]} / {sum(s)/len(s):.1f} / {s[-1]}")
    buckets = Counter()
    for n in s:
        if n == 1:        buckets["1"] += 1
        elif n <= 5:      buckets["2-5"] += 1
        elif n <= 20:     buckets["6-20"] += 1
        elif n <= 50:     buckets["21-50"] += 1
        else:             buckets["50+"] += 1
    for k in ["1","2-5","6-20","21-50","50+"]:
        print(f"  {k:8s} {buckets[k]}")
print()
print(f"Component-drawing coverage (how many BOM components have their own PDF):")
if pdf_components_drawn:
    s = sorted(pdf_components_drawn)
    print(f"  min/median/mean/max: {s[0]} / {s[len(s)//2]} / {sum(s)/len(s):.2f} / {s[-1]}")
    print(f"  Parents with ≥1 component drawn: {sum(1 for n in s if n>=1)}")
    print(f"  Parents with ≥5 components drawn: {sum(1 for n in s if n>=5)}")
    print(f"  Parents with ≥half their components drawn: "
          f"{sum(1 for n,t in zip(pdf_components_drawn, pdf_bom_sizes) if t>0 and n/t >= 0.5)}")
