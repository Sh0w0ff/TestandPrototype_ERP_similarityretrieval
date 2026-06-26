"""vocab_tags — phrase-level standards tagging over spatial UNITS (box cells / body blocks),
NEVER over a flattened string. Each unit keeps its spatial identity (like a textbox); tagging
attaches whole-phrase, explainable, negation-aware tags.

A unit's text is split into clauses ONLY where it fuses sentences (a 'No.../NOTE!/See/Surface
treatment' boundary inside one block); otherwise the unit is one clause. Each clause is tagged by:
  * the 14 `vocab.ALL_STANDARDS_LIBS` databases (token lookups) — precise, standard-grounded;
  * a small curated SURFACE anchor set for what the vocab misses (RAL colour, KC paint code, the
    phrase 'surface treatment', paint systems).
The surface-domain vocab libs (corrosion, surface_texture) are FOLDED into one `surface` category,
so surface is just a vocab category — not a separate block.

Per-unit `tag_unit` keeps the clause text (provenance, inline display); the corpus-level `aggregate`
is a COMPACT presence summary — {cat: [{anchors, negated}]} with NO clause text. The old aggregate
stored the whole clause under every category it triggered, so a multi-topic note was copied 4-5×
(JSON bloat + an unreadable, repetitive tag section). The fingerprint only ever consumed `anchors`/
`negated`, never the clause, so dropping it leaves the typed tokens' derivation intact.

Public API:
  tag_unit(text)         -> [ {clause, negated, tags:{cat:[anchors]}} ]   (one spatial unit; keeps clause)
  aggregate(texts)       -> {cat: [ {anchors, negated} ]}                 (compact presence summary)
"""
import re
import vocab   # legacy 14-database package (scripts/part1/vocab)

RAL = re.compile(r"\bRAL[\s-]?\d{3,4}\b", re.I)
KCPAINT = re.compile(r"\bKC\d\b", re.I)
SURF_PHRASE = re.compile(r"surface treatment|paint system|coating|primer", re.I)
PAINT_SYS = re.compile(r"\bS\d\.\d\d\b")
SURFACE_ANCHORS = [RAL, KCPAINT, SURF_PHRASE, PAINT_SYS]

# Weld callouts (fillet throat 'a', leg 'z', penetration/butt 's'), e.g. a4, a6, s8a6, z6.
# Case-SENSITIVE lowercase so the title-block paper formats 'A2 Size'/'A4 Size' don't false-match.
WELD = re.compile(r"\b(?:[saz]\d{1,2})+\b")
# ISO hole-fit class, IT grade bounded to 1..18 (H7, H11) so spec heights like 'H720'/'H132'
# (digit run continues past the bound, killing the \b) never match. Lowercase shaft fits deferred.
FIT = re.compile(r"\bH(?:1[0-8]|[1-9])\b")

# ERP production-line process hints — anchored in drawing text (EN + Finnish).
# Finnish translations from ERP work_phase_description / work_center_description:
#   hitsaus=welding, maalaus=painting, puhallus=blasting, sahaus=sawing,
#   taivutus=bending, koneistus=machining, hionta=grinding, kierteytys=threading,
#   pemmaus=PEM-inserting, pesu=washing, irroitus=deburring/removal, setitys=setup.
# Each entry: (production_line_code, regex). Anchor added to `process` category.
# Guards: minimum 4-char word, no false-match on common short tokens.
_PROCESS_RULES = [
    ("laser",    re.compile(r"\blaser\b", re.I)),
    ("plasma",   re.compile(r"\bplasma\b", re.I)),
    ("weld",     re.compile(r"\bweld(?:ing|ed|s)?\b|\bhitsaus\b|\bheftaus\b|\bhitsataan\b", re.I)),
    ("bend",     re.compile(r"\bbend(?:ing|s|radius|radii|r=)?\b|\btaivutus\b|\btaivutetaan\b|\bbending\b", re.I)),
    ("blast",    re.compile(r"\bblast(?:ing)?\b|\bpuhallus\b|\bsand.?blast\b|\bshot.?blast\b", re.I)),
    ("paint",    re.compile(r"\bpaint(?:ing|ed)?\b|\bmaalaus\b|\bpowder.?coat\b|\bjauhemaalaus\b|\bpulver\b", re.I)),
    ("machine",  re.compile(r"\bmachin(?:ing|ed)?\b|\bkoneistus\b|\bmilling\b|\bturning\b|\bsorvaus\b|\bdrill(?:ing)?\b|\bporaus\b", re.I)),
    ("deburr",   re.compile(r"\bdeburr(?:ing)?\b|\bhionta\b|\bgrind(?:ing)?\b|\birroitus\b|\bchamfer(?:ing)?\b", re.I)),
    ("saw",      re.compile(r"\bsaw(?:ing)?\b|\bsahaus\b|\bkatkaisu\b|\bcutting\b", re.I)),
    ("thread",   re.compile(r"\bthread(?:ing)?\b|\bkierteytys\b|\btap(?:ping)?\b", re.I)),
    ("pack",     re.compile(r"\bpack(?:ing|aging)?\b|\bpakkaus\b", re.I)),
    ("wash",     re.compile(r"\bwash(?:ing)?\b|\bpesu\b|\bcleaning\b|\bdegreas(?:ing)?\b", re.I)),
]

NEG_START = re.compile(r"^(no|not|without)\b", re.I)
CLAUSE = re.compile(r"(?=\bNo\b)|(?=\bNot\b)|(?=NOTE!)|(?=\bSee\b)|(?=Surface treatment)", re.I)
FOLD = {"corrosion": "surface", "surface_texture": "surface"}
# NB: callers pass LIGHT-CLEANED text (dims already routed out by light_clean), so bare dimensions
# like '22' never reach the vocab lookups -> no numeric false-matches (no separate guard needed).


def clauses(text):
    """Split a spatial unit into clauses only at fused-sentence boundaries; else one clause."""
    return [p.strip() for p in CLAUSE.split(text) if p and p.strip()] or [text]


def tag_unit(text):
    """Tag one spatial unit (a box cell or a body block). Returns a list of per-clause records."""
    out = []
    for cl in clauses(text):
        neg = bool(NEG_START.match(cl))
        cats = {}
        for tk in cl.replace(",", " ").split():
            for name, _d, look in vocab.ALL_STANDARDS_LIBS:
                if look(tk):
                    cats.setdefault(FOLD.get(name, name), set()).add(tk)
        for rx in SURFACE_ANCHORS:
            m = rx.search(cl)
            if m:
                cats.setdefault("surface", set()).add(m.group(0))
        for m in WELD.finditer(cl):
            cats.setdefault("weld_quality", set()).add(m.group(0))
        for m in FIT.finditer(cl):
            cats.setdefault("fits", set()).add(m.group(0))
        for prod_line, rx in _PROCESS_RULES:
            if rx.search(cl):
                cats.setdefault("process", set()).add(prod_line)
        if cats:
            out.append({"clause": cl, "negated": neg,
                        "tags": {k: sorted(v) for k, v in cats.items()}})
    return out


def aggregate(texts):
    """texts: list of spatial-unit strings (box cells + body blocks). Returns a COMPACT presence
    summary {category: [ {anchors, negated} ]} merged across all units, de-duplicated — no clause
    text (that lives per-block in `tag_unit`; the fingerprint only reads anchors/negated)."""
    agg = {}
    for t in texts:
        for rec in tag_unit(t):
            for cat, anchors in rec["tags"].items():
                entry = {"anchors": anchors, "negated": rec["negated"]}
                bucket = agg.setdefault(cat, [])
                if entry not in bucket:
                    bucket.append(entry)
    return agg
