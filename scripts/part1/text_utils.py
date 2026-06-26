"""
Shared text-cleaning + tokenization helpers for Part 1 scripts.

Imported by extract_text.py and glossary.py. Keep small and dependency-free.
"""

from __future__ import annotations

import re

# Line-continuation: a `-` at end of line followed by \n + optional leading whitespace,
# where the next non-blank char is a letter or digit. We strip the newline + spaces but
# preserve the dash itself (so `EN10029-PL10-S355J2+N` survives wrap, and a legitimately
# hyphenated `ISO 2768-M` doesn't lose its dash).
CONTINUATION_PAT = re.compile(r"(?<=-)\n[ \t]*(?=[A-Za-z0-9])")

# Column-gap separator used by pdftotext -layout: 4+ consecutive whitespace chars
# (spaces or tabs) mark the boundary between adjacent fields on the same visual row.
COLUMN_GAP_PAT = re.compile(r"[ \t]{4,}")


def rejoin_continuations(text: str) -> str:
    """Stitch lines split by soft-hyphen-style wraps; the trailing `-` stays."""
    return CONTINUATION_PAT.sub("", text)


def clean_field_value(v: str) -> str:
    """Cut a captured title-block field value at the first column gap.

    pdftotext flattens adjacent title-block fields onto the same text row separated
    by a wide whitespace run. Without this cut, the captured value carries trailing
    content from the next field column.
    """
    parts = COLUMN_GAP_PAT.split(v.strip(), maxsplit=1)
    return parts[0].strip()


# --- Tokenization (used by glossary.py) ---

_RAW_TOKEN_PAT = re.compile(r"\S+")
_PART_SPLIT_PAT = re.compile(r"[-/]")
_CANON_STRIP_PAT = re.compile(r"[-/\s]")


# Boilerplate token set — words from ABB copyright + KC confidentiality blocks
# + ISO 5455 projection notes + generic stopwords + title-block field labels.
# Filtered out of raw_tokens because they swamp downstream vocab harvests
# without carrying manufacturing signal. Audit 2026-05-24 surfaced these as
# the top-frequency uncovered tokens across all production-line FN sets.
BOILERPLATE_WORDS: frozenset[str] = frozenset({
    # ABB copyright block ("We reserve all rights... Reproduction, dissemination...
    # without express written authorization is forbidden... third parties...")
    "ALL", "RIGHTS", "RESERVED", "RESERVE", "THIS", "THE", "AND", "FOR",
    "WITHOUT", "EXPRESS", "INFORMATION", "CONTAINED", "PROPRIETARY",
    "THIRD", "DOCUMENT", "USE", "DISCLOSURE", "AUTHORIZED", "AUTHORISED",
    "ANY", "NOT", "WRITTEN", "PROPERTY", "EXCLUSIVE", "REPRESENTS",
    "CONFIDENTIAL", "REPRODUCED", "DISCLOSED", "ALTERED", "TRANSFERRED",
    "FORBIDDEN", "PARTIES", "PARTY", "PERMISSION", "CONSENT", "PRIOR",
    "USED", "MADE", "AVAILABLE", "WHOLE", "WHATSOEVER", "OTHERS",
    "WITH", "WHEN", "AFTER", "BEFORE", "FROM", "INTO", "ONTO", "ONLY",
    "WE", "OUR", "ARE", "WAS", "WERE", "BEING", "BEEN", "HAVE", "WILL",
    "SHALL", "MUST", "MAY", "CAN", "HEREIN", "THEREIN", "EXCEPT",
    "WHERE", "WHICH", "WHILE", "SUCH", "OF", "TO", "IS", "BE", "IN",
    "ON", "OR", "AT", "BY", "AN", "AS",
    "REPRODUCTION", "DISSEMINATION", "COMMUNICATION", "UTILIZATION",
    "OFFENDERS", "LIABLE", "DAMAGES",
    # KC confidentiality ("This document and the information contained herein
    # is the exclusive property of Konecranes Plc and represents...")
    "KONECRANES", "ABB", "PLC", "TRADE", "SECRET",
    # ISO 5455 projection-symbol notes ("FIRST ANGLE PROJECTION...")
    "FIRST", "ANGLE", "PROJECTION", "CORRECT", "FACTOR", "ADDING",
    "DIMENSIONS",
    # Title-block headers and BOM-row labels
    "POS", "REVISED", "PLATE", "SIZE", "ITEM", "QTY", "QUANTITY",
    "DESCRIPTION", "SPECIFICATION", "WIDTH", "LENGTH", "WEIGHT",
    "DRAWN", "CHECKED", "APPROVED", "DESIGNED", "DEPT", "DEPARTMENT",
    "DATE", "SCALE", "REV", "REVISION", "TITLE", "SHEET", "PAGE",
    "FORMAT", "FRAME", "BORDER", "VIEW", "SECTION", "DETAIL",
    "PARTIAL", "ASSEMBLY",   # NB: assembly vocab module catches verbs ASSEMBLE/ASSEMBLED
    "DRAWING", "DRAWINGS", "DIMENSIONING", "TOLERANCING", "GENERAL",
    "TOLERANCE", "TOLERANCES",
    # Field labels (values get picked up by structured extractors, label words don't carry signal)
    "MATERIAL", "COATING", "REFERENCE", "PRODUCT", "PROJECT",
    "FOLDER", "KEYWORDS", "NUMBER", "STANDARDS", "STANDARD", "SURFACE",
    "TREATMENT", "TREATED", "FINISH", "FINISHED",
    # Position / orientation words
    "TOTAL", "EACH", "BOTH", "SAME", "REAR", "FRONT", "TOP",
    "BOTTOM", "LEFT", "RIGHT", "SIDE", "INNER", "OUTER", "OPEN", "CLOSED",
    "INSIDE", "OUTSIDE", "ABOVE", "BELOW", "ACROSS", "ALONG",
    # Finnish field labels
    "MATERIAALI", "PAINO", "MITTAKAAVA", "PÄIVÄYS", "PIIRT", "TARKAST",
    # ── Second-pass boilerplate surfaced by audit 2026-05-24 re-run ─────────
    # ABB "PREPARED BY... STRICTLY... AUTHORITY... DMS... UNMARKED DIMENSIONS"
    # block (sits below the main copyright span). None of these carry process
    # signal; all are template scaffolding.
    "SET", "AUTHORITY", "STRICTLY", "BASED", "FORM", "CUSTOMER", "NAME",
    "DMS", "GEN", "ORIGINAL", "PREPARED", "UNMARKED", "RADII",
    "BEND",  # word appears in 729 ABB drawings as 'BEND RADII' template
             # header, not as a process callout. Actual bend operations
             # use 'BENT' / 'BENDING' (vocab surface_forms).
    # KC continuation of the confidentiality block + designer/ECN line
    "THAT", "OTHERWISE", "EMPLOYED", "ECN", "MANNER", "SEE", "OWNER",
    "MANUFACTURING", "INSTRUCTIONS", "DESIGNER",
})


def _is_boilerplate(tok: str) -> bool:
    """Token is pure-alphabetic boilerplate. Strips trailing punct first."""
    t = tok.upper().rstrip(":;,.")
    return t in BOILERPLATE_WORDS


def raw_tokens(text: str) -> list[str]:
    """Whitespace-only split, with boilerplate tokens dropped.

    Compound identifiers (EN10029-PL10-S355J2+N) stay intact — only single-word
    boilerplate matches are removed.  See BOILERPLATE_WORDS above for source.
    """
    return [t for t in _RAW_TOKEN_PAT.findall(text) if not _is_boilerplate(t)]


def part_tokens(raw: str) -> list[str]:
    """Split a raw compound on `-` and `/`. Preserves `+` (metallurgy) and `,` (decimals)."""
    if "-" not in raw and "/" not in raw:
        return []
    return [p for p in _PART_SPLIT_PAT.split(raw) if p]


def canonical_key(tok: str) -> str:
    """Group-by key for variant clustering.

    Lowercase, replace decimal comma with dot, strip separators and trailing punctuation.
    Examples:
      EN10029 / EN 10029 / EN-10029  -> en10029
      1,5MM / 1.5MM                  -> 1.5mm
      S355J2+N                       -> s355j2+n    (+ preserved)
    """
    t = tok.lower().replace(",", ".")
    t = _CANON_STRIP_PAT.sub("", t)
    return t.rstrip(":;,.")
