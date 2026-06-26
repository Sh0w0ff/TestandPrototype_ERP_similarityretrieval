"""erp_truth — E1 ground-truth layer: join a drawing STEM to its ERP target labels.

Reuses the proven filename->item linkage from scripts/bom_coverage_check.py (the basis for the
96.9% linkage / 94.3%-have-BOM finding) by COPYING tokens_from_filename verbatim (project convention:
copy proven pure logic, don't import the script — which runs prints at import). Exposes, per stem:

  stem_to_items(stem)       -> set of ERP Item IDs the drawing represents
  stem_to_components(stem)  -> set of component Item IDs   (G2 BOM target)
  stem_to_phases(stem)      -> set of work-phase descriptions (G3 target)
  stem_to_family(stem)      -> Product family description ('ABB DRIVES' / 'KONECRANES') for G1 labels

These are TRAINING/EVAL labels (the prediction TARGET) — never an inference-time feature (CC3 leakage).
"""
import csv, os, re
from pathlib import Path
from collections import defaultdict

# ERP CSVs live in the data pool; default to the original FYP location, override with FYP_DATA_ROOT.
ROOT = Path(os.environ.get("FYP_DATA_ROOT", "/Users/sh0w0ff/FYP"))
ITEM_CSV = ROOT / "Item_Basic_Data.csv"
BOM_CSV = ROOT / "Bill_of_Materials.csv"
WP_CSV = ROOT / "Work_Phases.csv"

DIGIT_ID_RE = re.compile(r"^\d{7,}$")
SUFFIX_VARIANTS = ("-DRW1", "-PDF1", "-NXD1", "-A", "-B", "A", "B", "C", "D")


def tokens_from_filename(fn):
    """COPIED VERBATIM from bom_coverage_check.py — generates the candidate drawing-number tokens
    a filename could match against an ERP description head (both vendor formats, separator variants)."""
    stem = re.sub(r"\.(pdf|PDF)$", "", fn)
    toks = {stem}
    for m in re.findall(r"\(([^()]+)\)", stem):
        toks.add(m)
    for t in re.split(r"[_;\s]+", stem):
        if len(t) >= 4:
            toks.add(t)
        parts = t.split("-")
        if len(parts) >= 2:
            for i in range(len(parts)):
                for j in range(i + 1, len(parts) + 1):
                    sub = "-".join(parts[i:j])
                    if len(sub) >= 4:
                        toks.add(sub)
                    last = parts[j - 1] if j > i else ""
                    stripped = re.sub(r"\d+$", "", last)
                    if stripped and stripped != last:
                        sub2 = "-".join(parts[i:j - 1] + [stripped]) if j > i else stripped
                        if len(sub2) >= 4:
                            toks.add(sub2)
        for sub in list(toks):
            if DIGIT_ID_RE.match(sub):
                for suf in SUFFIX_VARIANTS:
                    toks.add(sub + suf)
    for sub in list(toks):
        positions = [i for i in range(1, len(sub)) if sub[i - 1].isdigit() and sub[i].isalpha()]
        for pos in positions:
            for sep in ("-", "/"):
                toks.add(sub[:pos] + sep + sub[pos:])
    return toks


def _rows(path):
    return list(csv.DictReader(open(path, encoding="latin-1"), delimiter=";"))


class ErpTruth:
    """Lazily-built ERP join indices. One instance reused across an E1 run."""

    def __init__(self):
        self.item_of_drawing = defaultdict(list)   # description head -> [item ids]
        self.item_family = {}                       # item -> product family desc
        self.bom_children = defaultdict(set)        # parent item -> {component items}
        self.item_phases = defaultdict(set)         # item -> {phase descriptions}
        self.item_desc = {}                         # item -> raw Item description
        for r in _rows(ITEM_CSV):
            item = r["Item"].strip()
            desc = r["Item description"] or ""
            self.item_desc[item] = desc
            self.item_family[item] = (r.get("Product family description") or "").strip()
            heads = set()
            if "--" in desc:
                heads.add(desc.split("--", 1)[0].strip())
            heads.add(desc.split("/", 1)[0].strip())
            for h in heads:
                if h:
                    self.item_of_drawing[h].append(item)
        for r in _rows(BOM_CSV):
            p, c = r["Parent part"].strip(), r["Component"].strip()
            if p and c:
                self.bom_children[p].add(c)
            # PASS 2 (bom_description_join): a drawing's ID may live ONLY in a BOM description
            # field (the drawing is someone else's parent/component). Index those heads -> item
            # so stems unmatched by Item_Basic_Data still join (lifts 96.9% -> ~99.9%).
            for col, item in (("Parent part description", p), ("Component description", c)):
                desc = r.get(col) or ""
                if not item:
                    continue
                heads = set()
                if "--" in desc:
                    heads.add(desc.split("--", 1)[0].strip())
                heads.add(desc.split("/", 1)[0].strip())
                for h in heads:
                    if h and item not in self.item_of_drawing[h]:
                        self.item_of_drawing[h].append(item)
        for r in _rows(WP_CSV):
            it, ph = r["Item"].strip(), (r["Work phase description"] or "").strip()
            if it and ph:
                self.item_phases[it].add(ph)
        # residual fallback corpus (bom_description_join): (lowered description, item) for a
        # boundary-respecting substring search when head-indexing misses (ID embedded mid-string).
        self._desc_items = []
        for r in _rows(BOM_CSV):
            for col, item in (("Parent part description", r["Parent part"].strip()),
                              ("Component description", r["Component"].strip())):
                d = (r.get(col) or "").lower()
                if item and d:
                    self._desc_items.append((d, item))

    def _substring_items(self, stem):
        """Last-resort join: drawing-ID token embedded mid-description, flanked by non-alnum."""
        toks = [t.lower() for t in tokens_from_filename(stem)
                if len(t) >= 6 and any(c.isdigit() for c in t)]
        out = set()
        for d, item in self._desc_items:
            for t in toks:
                i = d.find(t)
                if i >= 0 and (i == 0 or not d[i - 1].isalnum()) \
                        and (i + len(t) == len(d) or not d[i + len(t)].isalnum()):
                    out.add(item); break
        return out

    def stem_to_items(self, stem):
        out = set()
        for t in tokens_from_filename(stem):
            out.update(self.item_of_drawing.get(t, []))
        if not out:                          # residual: boundary-respecting substring fallback
            out = self._substring_items(stem)
        return out

    def stem_to_components(self, stem):
        out = set()
        for it in self.stem_to_items(stem):
            out |= self.bom_children.get(it, set())
        return out

    def stem_to_phases(self, stem):
        out = set()
        for it in self.stem_to_items(stem):
            out |= self.item_phases.get(it, set())
        return out

    def stem_to_family(self, stem):
        fams = {self.item_family.get(it, "") for it in self.stem_to_items(stem)}
        fams.discard("")
        return next(iter(fams), "")

    # coarse part-type vocabulary (head nouns; descriptions lead with the type, so first match wins)
    PART_TYPE_WORDS = ["COUNTERWEIGHT", "BUSDUCT", "BRACKET", "SUPPORT", "RADIATION", "SHIELD",
                       "CABINET", "PLINTH", "SHROUD", "PROFILE", "FLANGE", "SPACER", "WASHER",
                       "COVER", "GLAND", "MODULE", "FRAME", "PLATE", "BEAM", "SHAFT", "TUBE",
                       "RAIL", "GUIDE", "ANGLE", "DUCT", "ROOF", "BAR", "RING", "PIN", "LUG"]

    def stem_to_parttype(self, stem):
        """Coarse part-type label from the Item description (after the drawing-no '--' head),
        as the FIRST known head-noun encountered left-to-right (descriptions lead with the type)."""
        words = set(self.PART_TYPE_WORDS)
        for it in self.stem_to_items(stem):
            d = self.item_desc.get(it, "")
            tail = d.split("--", 1)[1] if "--" in d else d
            for tok in tail.upper().replace(",", " ").split():
                if tok in words:
                    return tok
        return ""


if __name__ == "__main__":
    import sys
    erp = ErpTruth()
    for st in (sys.argv[1:] or
               ["SQEM-R5000W5150H720-7021_BEAM; SUPPORT BEAM_69967516_1_69967516-DRW1",
                "3AXD50000017211", "68410134"]):
        print(f"\n{st[:50]}")
        print("  items     :", sorted(erp.stem_to_items(st)))
        print("  family    :", erp.stem_to_family(st))
        print("  components:", len(erp.stem_to_components(st)), sorted(erp.stem_to_components(st))[:6])
        print("  phases    :", sorted(erp.stem_to_phases(st)))
