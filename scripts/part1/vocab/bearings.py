"""Bearing designations — high-value BOM children, especially for Konecranes.

Bearing callouts follow tight ISO numbering grammar that can be regex-parsed:

    6201           radial deep-groove ball bearing, 12 mm bore
    6308-2RS       same, sealed both sides
    6310/C3        clearance class C3
    32208          tapered roller, ISO 355
    NU 312         cylindrical roller
    22220          spherical roller

Sourcing (full citation context in thesis_direction.md §7):
    Groover §32  — Mechanical Assembly (covers bearings as standard hardware
                   but does not enumerate ISO 15 codes).
    ISO 15 free  — Scope page lists the bearing series-code grammar (first
    preview        digit = type, last 2 = bore code where bore mm = 5 × digits
                   for codes 04-99).
    Wikipedia    — "Rolling-element bearing" / "Ball bearing" articles
                   reproduce the ISO 15 series structure with citations.

Bore-code convention (ISO 15):
    Code 00 = 10 mm, 01 = 12 mm, 02 = 15 mm, 03 = 17 mm
    Code 04 onwards: bore mm = code × 5  (e.g. 06 = 30 mm, 10 = 50 mm, 12 = 60 mm)
"""

from __future__ import annotations

import re


def _e(standard: str, canonical: str, source: str, family: str,
       surface_forms: list[str] = None) -> dict:
    return {
        "standard": standard,
        "canonical": canonical,
        "family": family,
        "source": source,
        "surface_forms": surface_forms or [],
    }


BEARING_FAMILIES: list[dict] = [
    # ── ISO 15 — radial bearings ────────────────────────────────────────────
    _e("ISO 15", "Deep groove ball bearing (6xxx series)",
       "ISO 15 type code 6 = single-row deep groove ball. Most common "
       "bearing on light/medium machinery.",
       "deep_groove_ball",
       ["6000", "6001", "6002", "6003", "6004", "6005", "6006", "6201", "6202",
        "6203", "6204", "6205", "6206", "6207", "6208", "6210", "6212", "6300",
        "6302", "6304", "6306", "6308", "6310", "6312"]),
    _e("ISO 15", "Angular contact ball bearing (7xxx series)",
       "ISO 15 type code 7. Combined radial + axial load.",
       "angular_contact_ball",
       ["7200", "7201", "7203", "7205", "7206", "7208", "7210"]),
    _e("ISO 15", "Self-aligning ball bearing (1xxx / 2xxx series)",
       "Two rows of balls on spherical raceway.",
       "self_aligning_ball",
       ["1200", "1203", "1205", "1208", "2200", "2203", "2205", "2208"]),
    _e("ISO 15", "Cylindrical roller bearing (N / NU / NJ / NUP / RNU)",
       "Higher radial load capacity than ball bearings. "
       "Prefix denotes flange configuration (N = no flanges on outer ring, "
       "NU = no flanges on inner ring, NJ / NUP = thrust-capable variants).",
       "cylindrical_roller",
       ["N 204", "NU 204", "NJ 205", "NUP 206", "N 308", "NU 310", "NU 312",
        "NU 2310"]),
    _e("ISO 15", "Spherical roller bearing (2xxxx series)",
       "Two rows of barrel-shaped rollers; self-aligning. Heavy radial load.",
       "spherical_roller",
       ["22205", "22208", "22210", "22212", "22215", "22220", "22308", "22310",
        "22312", "22320"]),
    _e("ISO 15", "Needle roller bearing (NA / RNA / HK / NK)",
       "Long thin rollers; high load capacity in compact section.",
       "needle_roller",
       ["NA 4900", "NA 4906", "NK 35/20", "HK 0810", "HK 1212"]),

    # ── ISO 355 — tapered roller bearings ───────────────────────────────────
    _e("ISO 355", "Tapered roller bearing (5-digit / 6-digit)",
       "ISO 355 designation system. Common for shafts taking combined "
       "radial + axial loads (e.g. crane wheel shafts).",
       "tapered_roller",
       ["30202", "30203", "30204", "30205", "30206", "30207", "30208", "30210",
        "30212", "32008", "32010", "32208", "32210", "32305", "32308", "33205"]),

    # ── ISO 104 — thrust bearings ───────────────────────────────────────────
    _e("ISO 104", "Thrust ball bearing (5xxxx series)",
       "Axial load only. ISO 104 type code starts with 5.",
       "thrust_ball",
       ["51100", "51101", "51103", "51105", "51108", "51110", "51200", "51203",
        "51205", "51208"]),
    _e("ISO 104", "Cylindrical roller thrust bearing (8xxxx series)",
       "Heavy axial load.",
       "thrust_roller",
       ["81100", "81102", "81104", "81108", "81110", "89308", "89310"]),
    _e("ISO 104", "Spherical roller thrust bearing (29xxx series)",
       "Self-aligning thrust bearing for very heavy axial loads. Common in "
       "crane hook blocks (Konecranes-relevant).",
       "spherical_roller_thrust",
       ["29320", "29412", "29420", "29432"]),
]


# ── Regex patterns ──────────────────────────────────────────────────────────

# Deep groove ball: 4 digits starting 60xx / 62xx / 63xx / 64xx (and 16xx miniature).
DEEP_GROOVE_RE = re.compile(r"\b(6[0-4]\d{2}|16\d{2})\b")

# Tapered roller: 5 digits starting 30xxx / 31xxx / 32xxx / 33xxx / T2xxx (or 6 digits).
TAPERED_ROLLER_RE = re.compile(r"\b(3[0-3]\d{3}|T2[A-Z]{2}\d{3})\b")

# Spherical roller: 5 digits starting 22xxx / 23xxx / 24xxx.
SPHERICAL_ROLLER_RE = re.compile(r"\b(2[2-4]\d{3})\b")

# Cylindrical roller with prefix: N / NU / NJ / NUP / NN / RNU + 3-4 digits.
CYLINDRICAL_ROLLER_RE = re.compile(r"\b(N|NU|NJ|NUP|NN|RNU)\s*(\d{3,4}(?:/\d{1,3})?)\b")

# Needle roller prefixes.
NEEDLE_RE = re.compile(r"\b(NA|RNA|HK|NK|BK)\s*(\d{3,5}(?:/\d{1,3})?)\b")

# Thrust ball: 5 digits starting 51xxx / 52xxx / 53xxx.
THRUST_BALL_RE = re.compile(r"\b(5[1-3]\d{3})\b")

# Suffixes that modify a bearing designation (do not consume the base — used
# as side annotations when present near a bearing number).
BEARING_SUFFIX_RE = re.compile(
    r"\b(?:2RS|2Z|ZZ|RS1|RS2|2RS1|2RS2|N|NR|/C\d|/P\d|K|K30)\b"
)


# ── Lookups ────────────────────────────────────────────────────────────────

_BY_FORM: dict[str, dict] = {}
for _entry in BEARING_FAMILIES:
    for _sf in _entry["surface_forms"]:
        _BY_FORM.setdefault(_sf.upper().replace(" ", ""), _entry)
        _BY_FORM.setdefault(_sf.upper(), _entry)


def lookup(token: str) -> dict | None:
    """Lookup a bearing designation by surface form (with or without spaces)."""
    t = token.upper().strip()
    return _BY_FORM.get(t) or _BY_FORM.get(t.replace(" ", ""))


def parse_bore_code(code: str) -> int | None:
    """Decode ISO 15 bore code to bore diameter in mm. None if not parsable."""
    if not code.isdigit() or len(code) != 2:
        return None
    n = int(code)
    if n <= 3:
        return {0: 10, 1: 12, 2: 15, 3: 17}[n]
    return n * 5  # codes 04..99 → bore mm = code × 5


SURFACE_FORMS: frozenset[str] = frozenset(_BY_FORM.keys())
