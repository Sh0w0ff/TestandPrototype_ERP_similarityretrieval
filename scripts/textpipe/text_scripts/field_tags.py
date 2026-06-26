"""field_tags — LABEL-DRIVEN typing of structured title-block (and body) fields.

A drawing's title block is a FORM: each cell is "<Label> <value>" (ABB rarely uses a colon —
'Title PLINTH...', 'Scale 1:5', 'Weight kg 2.47'; some fields do: 'Material:', 'Coating:',
'Gen Tol:'). So the cell's LABEL fixes the value's semantic type — we read the value BY label
rather than guessing from content. This catches values we have no vocab for (e.g. the real
material 'HDG STEEL SHEET' / 'U-Stahl (DIN1026)') and is why this REPLACES the old colon-only
extract_fields (which returned {} on every drawing — see thesis §9.8.25).

Quantities are normalised to a canonical unit (weight -> kg; the unit may sit in the label
'Weight kg' or after the number 'Weight 500 g'). Weight is a STRUCTURED, NULLABLE field
(None when the drawing has no weight cell).

A small keyword vocab over the part-name value yields part_type (PLATE/BEAM/BRACKET...); over any
material value yields material_class (STEEL/STAINLESS/ALU...). These are the only content lookups;
everything else is pure label typing.

Public API:
  type_units(units) -> {part_name, part_type, material, material_class, coating,
                        general_tolerance, scale, weight_kg}   (lists for multi-valued; weight float|None)
"""
import re

# canonical field -> label regex matched at the START of a unit (optional ':'/spaces after).
# Order matters: more specific labels first ('Gen Tol' before a bare 'Tol').
LABELS = [
    ("part_name",         r"(?:Title|Benennung|Designation|Description)"),
    ("general_tolerance", r"(?:Gen\.?\s*Tol\.?|General\s+toleranc\w*|Allgemeintoleranz)"),
    ("material",          r"(?:Material|Werkstoff)"),
    # 'Surface' only as a FIELD label (+treatment/finish/protection or a colon) — never the prose word
    # 'surface' in '...the surface of the beams'. Coating/Finish/Oberfl are safe standalone labels.
    ("coating",           r"(?:Coating|Surface\s+(?:treatment|finish|protection|prot\w*)|Surface(?=\s*:)|Finish|Oberfl\w+|Beschichtung)"),
    ("scale",             r"(?:Scale|Ma\Sstab)"),
    ("specification",     r"(?:Specification|Spezifikation|Spec\.?)"),
    # 'Based on <parent-id>' = the drawing this part derives from. A PARENT reference is genuine
    # predictive signal (shared lineage tends to share a BOM), so we rescue it as a field rather
    # than letting it fall into the admin chrome. Bare 'Based on' (no id) becomes admin (is_admin).
    ("based_on",          r"Based\s*on"),
    # weight tolerates a leading qualifier so KC's 'Total weight kg 0.00' / 'Net weight' match too.
    ("weight",            r"(?:(?:Total|Net|Gross|Nominal|Approx\.?)\s+)?(?:Weight|Mass|Gewicht|Massa)"),
]
_LABEL_RX = [(f, re.compile(rf"^\s*{pat}\b\s*[:\-]?\s*(.+)$", re.I | re.S)) for f, pat in LABELS]

# weight: a number with an optional unit before (in the label, 'Weight kg 2.47') or after ('2.47 kg')
_WEIGHT_RX = re.compile(r"(?:(kg|kgs|g|gr|t|to?nnes?)\s*)?([0-9]+(?:[.,][0-9]+)?)\s*(kg|kgs|g|gr|t|to?nnes?)?", re.I)
_SCALE_RX = re.compile(r"\b(\d{1,3})\s*[:/]\s*(\d{1,3})\b")

# engineering-standard references (ISO 2768 / DIN 1026 / EN 10327 ...) — a CONTENT scan over every
# unit (not label-driven), folded in here so fields/standards/typed are ONE structured layer.
_STD_FAMILIES = ("ISO", "DIN", "EN", "ASTM", "GB", "ANSI", "JIS", "BS", "NF", "SFS")
_STD_RX = re.compile(r"\b(" + "|".join(_STD_FAMILIES) + r")[\s-]?(\d{2,6}[A-Za-z]?)(?:[-/]([A-Za-z0-9]{1,6}))?", re.I)

# coating value keywords (so a Coating/Finish cell value like 'PAINT' or 'HDG' is recognised even
# when the label was generic). Fixes the missed capital 'PAINT'.
_COATING_KW = re.compile(r"\b(paint|powder|primer|galvani[sz]ed|zinc|hdg|anodi[sz]ed|"
                         r"passivat\w*|degreas\w*|coating|e-?coat|ral\s*\d{3,4})\b", re.I)

# part-type keyword vocab over the part_name value (first match wins; longest-ish terms first).
_PART_TYPES = ["bracket", "support", "plinth", "shroud", "flange", "gusset", "housing", "cover",
               "guide", "frame", "panel", "plate", "beam", "shaft", "spacer", "washer", "mount",
               "base", "rail", "ring", "tube", "pipe", "bar", "rod", "lug", "clip", "bushing",
               "bush", "pin", "stud", "angle", "channel", "profile", "sheet", "weldment", "assembly",
               # ERP-grounded additions (top BOM component-description words, 2026-06-05)
               "stiffener", "foot", "rib", "insert", "pad", "clamp", "cap", "plug",
               "collar", "sleeve", "socket", "hook", "stop", "retainer", "arm", "lid"]
_PART_RX = re.compile(r"\b(" + "|".join(_PART_TYPES) + r")s?\b", re.I)

# material-class vocab over a material value.
# ERP-grounded additions (2026-06-05): DC01 (cold-rolled mild steel), S355MC/J2 variants,
# HR (hot-rolled) and CR (cold-rolled) process indicators, S235.
_MATERIALS = [("stainless",   r"stainless|inox|1\.4\d{3}|aisi\s*3\d\d|en\s*1\.4\d{3}"),
              ("aluminium",   r"alumini\w+|\balu\b|\bal\d{4}"),
              ("plastic",     r"plastic|pa6|pom|abs|ptfe|polyam\w+|nylon"),
              ("galv_steel",  r"hdg|galvani[sz]ed|dx51|z\d{3}\b|\+z\b|zinc.?coat\w*|sinkkipohja"),
              ("cold_rolled", r"\bdc0[1-6]\b|\bcr\b(?=\s+steel|\s+sheet)"),
              ("steel",       r"steel|stahl|s235|s355|s275|s420|s460|st\d{2}|fe\d{3}|"
                              r"din\s*1026|u-?stahl|\bhr\b(?=\s+s\d{3})|\bs355[a-z0-9+]*")]
_MAT_RX = [(name, re.compile(pat, re.I)) for name, pat in _MATERIALS]


def _to_kg(unit, num):
    v = float(num.replace(",", "."))
    u = (unit or "kg").lower()
    if u in ("g", "gr"):
        return round(v / 1000.0, 4)
    if u.startswith("t"):
        return round(v * 1000.0, 4)
    return round(v, 4)                      # kg / default


def parse_weight(value):
    """First number in a weight-field value -> kilograms (unit from label or trailing token)."""
    m = _WEIGHT_RX.search(value)
    if not m:
        return None
    unit = m.group(1) or m.group(3)
    return _to_kg(unit, m.group(2))


def parse_scale(value):
    m = _SCALE_RX.search(value)
    return f"{m.group(1)}:{m.group(2)}" if m else None


# value is clean only up to the first sentence end / next label keyword; specs are short.
_VALUE_CUT = re.compile(r"\.\s|\bWelding\b|\bColor\b|\bFINISH\b|\bSee\b|\bNote\b", re.I)
_NEXT_LABEL = re.compile(r"\b(?:Title|Material|Coating|Surface|Finish|Scale|Weight|Mass|"
                         r"Gen\.?\s*Tol|General\s+toleranc)\b", re.I)
_TOK_CAP = {"part_name": 12, "material": 8, "coating": 8, "general_tolerance": 6}
# a real general-tolerance value names a standard or a class — not a revision note like 'added'.
_GENTOL_OK = re.compile(r"ISO\s*2768|DIN\s*7?\d{3,4}|2768|7168|\b[mfcv][KLH]?\b|±|\d", re.I)
_NEG = re.compile(r"\b(no|without)\b|\bfree\b", re.I)
# a value that STARTS with a revision verb is a change-log entry ('Material changed to 60x60x8'),
# not a field statement — semantic guard (like negation), not a keyword filter.
_REV = re.compile(r"^(?:changed?|added|removed?|modified|corrected|updated?|deleted?|was|fixed|revised|moved)\b", re.I)

# a parent reference value must be a real id token (alphanumeric, >=5 chars), not bare 'Based on'.
_BASEDON_ID = re.compile(r"\b([A-Za-z0-9]*\d[A-Za-z0-9]{4,})\b")

# ---- ADMINISTRATIVE TITLE-BLOCK CHROME ----------------------------------------------------------
# The ABB title block is a fixed FORM. Beyond the few cells we type as fields, the rest is purely
# administrative metadata identical across the whole corpus: the rights statement, doc/DMS/change
# ids, author + role + date rows, the revision-index and revision descriptions, and the form layout
# labels (Doc. des. / Form / Sheet / Lang. / Customer / Total ...). None of it is predictive of the
# BOM or work phase, yet — because it is the SAME boilerplate on every ABB drawing — it acts as a
# corpus-wide stop-phrase that inflates ABB-to-ABB similarity. We recognise it and shelve it in
# debug.admin so it leaves BOTH the fingerprint and the unclassified coverage gap. Scoped to
# title-block CELLS only (body prose is never admin). thesis §9.8.27.
_ADMIN = [
    re.compile(r"reserve all rights|exclusive property of|information contained herein", re.I),  # legal boilerplate (ABB + KC)
    re.compile(r"BILL OF MATERIAL", re.I),                         # a BOM-sheet header block = TARGET side, never input
    re.compile(r"\bQINST\b|general manufacturing instruction", re.I),  # KC boilerplate cross-reference (every page)
    # ABB form/role labels
    re.compile(r"^(?:Prepared|Check\.?|Appr\.?|Resp\.?\s*dept|Rev\.?\s*ind|Lang\.?|"
               r"Doc\.?\s*des\.?|Doc\.?\s*no\.?|Cust\.?\s*Doc\.?\s*no\.?|DMS\s*Number|"
               r"Form|Sheet|Customer|Project\s*name|Total|Based\s*on)\b", re.I),
    # KC (Konecranes) revision-table + title-block labels (different template, same chrome).
    # \w* tails absorb OCR truncation ('Revised b') and word variants ('Owner department' vs 'Dept').
    # NOT 'Keywords' — that field carries part-classification signal (LONG VERSION / QC-codes), keep it.
    re.compile(r"^(?:Rev\b|Revis\w*|Description|Date\b|Design\w*|Check\w*|Appr\w*|Drawing\s*no|"
               r"Item\s*(?:ID|no)|Document\s*ID|Owner\s*[Dd]ep\w*|Folder|Size|KONECRANES)", re.I),
    re.compile(r"^3A[A-Z]{2}\d{5,}(?:\s*\(PART\))?$", re.I),       # ABB doc id (+ '(PART)')
    re.compile(r"^\d{9,}$"),                                       # 9+ digit change id
    re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{2,4}$"),                  # ABB date '19-May-14'
    re.compile(r"^\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?$"),         # KC ISO date/datetime '2012-05-09 10:20'
    re.compile(r"^mm$", re.I),                                     # bare unit cell (KC)
    re.compile(r"^[A-Z]\.[A-ZÄÖÅ][\wÄÖÅäöå]+$"),                   # author token 'I.Kallio'
    re.compile(r"^[A-Z]\.\d+\+?$"),                                # revision index 'H.1', 'J.2+'
    # pure change-log line (trailing verb + optional punctuation) — but NOT lines that carry a
    # current-state value ('Material changed to 60x60x8', 'Hole changed 18->20'): those keep a
    # residual token after the verb, so the trailing anchor leaves them in the coverage gap.
    re.compile(r"\b(?:added|removed|deleted|changed|modified|corrected|updated|revised)[.\s]*$", re.I),
]


def is_admin(text):
    """True if a text unit is administrative chrome (see _ADMIN). Applied to BOTH title-block cells
    and body blocks (the KC rights statement + a 'BILL OF MATERIAL' header block are prose). Caller
    routes these to debug.admin so they never enter the fingerprint or the unclassified coverage gap."""
    t = (text or "").strip()
    return bool(t) and any(rx.search(t) for rx in _ADMIN)


# A revision-table author cell: the 'Revised by'/'Design' column values KC leaves as standalone cells
# ('Ritala Mikko', 'EF', OCR-split 'Kir jokangas Timoari'). Distinguishable because KC PART names are
# ALL-CAPS ('END PLATE', 'BRACKET') while person names are Title-case. Scoped to title-block CELLS only.
_INITIALS = re.compile(r"^[A-ZÄÖÅ]{1,3}$")                         # 'EF', author initials
_NAMETOK = re.compile(r"^[A-ZÄÖÅ][a-zäöå]{2,}$")                   # a Title-case name token
_USERNAME = re.compile(r"^[a-zäöå]{3,}[0-9]{1,3}$")               # KC login id WITH a digit ('xkirjoti1');
# pure-letter ids ('simpari','etttva') are NOT caught on purpose — a bare lowercase word would eat real
# signal like 'holes'/'steel' ([[feedback_noise_over_overfiltering]]). Those stay in the coverage gap.


def is_person(text):
    """True if a title-block cell looks like a person name / author initials (revision-table author
    column). Guards: 1-3 tokens, NO digit, NO all-caps word (so ALL-CAPS part names never match)."""
    t = (text or "").strip()
    if not t:
        return False
    toks = t.split()
    if len(toks) == 1:                                            # 'EF' / login id 'xkirjoti1' (digit ok here)
        return bool(_INITIALS.match(toks[0]) or _USERNAME.match(toks[0]))
    if any(ch.isdigit() for ch in t):                            # a multi-token person name has no digits
        return False
    if not 2 <= len(toks) <= 3:
        return False
    if any(len(w) >= 3 and w.isupper() for w in toks):           # an ALL-CAPS word => a part name, not a person
        return False
    return sum(1 for w in toks if _NAMETOK.match(w)) >= 2         # >=2 Title-case name tokens


def _clean_value(field, val):
    """Trim a label's value to the actual spec: stop at a sentence end / a NEW label, cap tokens."""
    val = val.split("\n")[0].strip()
    if field in ("material", "coating", "general_tolerance"):
        cut = _VALUE_CUT.search(val)
        if cut and cut.start() > 0:
            val = val[:cut.start()].strip()
    # stop if a different field's label appears later in the same run-on cell
    nl = _NEXT_LABEL.search(val, 1)
    if nl:
        val = val[:nl.start()].strip()
    cap = _TOK_CAP.get(field)
    if cap:
        val = " ".join(val.split()[:cap])
    return val.strip(" :,-")


def _part_type(name):
    m = _PART_RX.search(name or "")
    return m.group(1).lower() if m else None


def _material_class(value):
    for name, rx in _MAT_RX:
        if rx.search(value or ""):
            return name
    return None


def type_units(units):
    """units: list of text strings (title-block cells + body clauses). Returns the typed field
    dict. Multi-valued fields (material/coating/part_name/...) collect a de-duplicated list;
    weight_kg is a single float (first found) or None."""
    out = {"part_name": [], "part_type": [], "material": [], "material_class": [],
           "coating": [], "general_tolerance": [], "scale": [], "weight_kg": None,
           "specification": [], "based_on": [], "standards": []}
    seen_std = set()
    for u in units:
        u = (u or "").strip()
        if not u:
            continue
        for m in _STD_RX.finditer(u):                       # content scan (not label-driven)
            fam, num, suf = m.group(1).upper(), m.group(2), (m.group(3) or "")
            if (fam, num, suf) not in seen_std:
                seen_std.add((fam, num, suf))
                out["standards"].append({"family": fam, "number": num, "suffix": suf or None})
        for field, rx in _LABEL_RX:
            m = rx.match(u)
            if not m:
                continue
            raw = m.group(1).strip()
            if field == "weight":
                if out["weight_kg"] is None:
                    out["weight_kg"] = parse_weight(raw)
                break
            if field == "scale":
                s = parse_scale(raw)
                if s and s not in out["scale"]:
                    out["scale"].append(s)
                break
            if field == "based_on":
                pid = _BASEDON_ID.match(raw)               # keep only a real parent id, not bare 'Based on'
                if pid and pid.group(1) not in out["based_on"]:
                    out["based_on"].append(pid.group(1))
                break
            val = _clean_value(field, raw)
            if not val:
                break
            if field == "part_name":
                if val not in out["part_name"]:
                    out["part_name"].append(val)
                pt = _part_type(val)
                if pt and pt not in out["part_type"]:
                    out["part_type"].append(pt)
            elif field == "material":
                if _REV.match(val):                     # 'Material changed to 60x60x8' = a change-log entry
                    break
                if val not in out["material"]:
                    out["material"].append(val)
                mc = _material_class(val)
                if mc and mc not in out["material_class"]:
                    out["material_class"].append(mc)
                if _COATING_KW.search(val) and not _NEG.search(val) and val not in out["coating"]:
                    out["coating"].append(val)          # ABB sometimes files paint under 'Material:'
            elif field == "coating":
                if _NEG.search(val) or _REV.match(val):  # absence, or a change-log entry (semantic)
                    break
                if val not in out["coating"]:           # else TRUST the label — keep value even w/o a known kw
                    out["coating"].append(val)
            elif field == "specification":
                if not _REV.match(val) and val not in out["specification"]:
                    out["specification"].append(val)
            elif field == "general_tolerance":
                if _GENTOL_OK.search(val) and val not in out["general_tolerance"]:
                    out["general_tolerance"].append(val)   # skip revision notes like 'added'
            break                                       # first matching label wins for this unit
    return out
