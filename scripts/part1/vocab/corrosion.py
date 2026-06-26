"""ISO 12944 / ISO 1461 — Corrosion-protection systems and classes.

Drawings frequently specify a corrosivity environment plus a paint
system: `C5-M MEDIUM` (industrial atmosphere, medium-durability paint),
`HDG 70µm` (hot-dip galvanized 70 µm thickness). These are direct
similarity features (parts with the same surface-protection spec serve
similar environments) and overlap with the existing schema fields
`corrosivity_class`, `paint_system`, `coating_thickness`.

Sourcing (full citation context in thesis_direction.md §7):
    ISO 12944 — Paints and varnishes — Corrosion protection of steel
                structures by protective paint systems. Multi-part.
                Free preview lists corrosivity classes C1..CX and
                durability ranges L/M/H/VH.
    ISO 1461  — Hot-dip galvanized coatings on iron and steel articles
                (specifications and test methods).
    ISO 9223  — Atmosphere corrosivity classification (parallel system).
    EN ISO 12944-5 — Specific paint-system codes (S2.05, S5.08, etc.).
"""

from __future__ import annotations

import re


def _e(category: str, canonical: str, source: str, surface_forms: list[str]) -> dict:
    return {
        "category": category,
        "canonical": canonical,
        "source": source,
        "surface_forms": surface_forms,
    }


CORROSION: list[dict] = [
    # ── ISO 12944-2 corrosivity classes ─────────────────────────────────────
    _e("corrosivity_class", "ISO 12944-2 corrosivity class C1 (very low)",
       "ISO 12944-2 §5. C1 = heated indoor (offices, dry).",
       ["C1"]),
    _e("corrosivity_class", "ISO 12944-2 corrosivity class C2 (low)",
       "ISO 12944-2. C2 = unheated indoor / rural outdoor.",
       ["C2"]),
    _e("corrosivity_class", "ISO 12944-2 corrosivity class C3 (medium)",
       "ISO 12944-2. C3 = urban / industrial.",
       ["C3"]),
    _e("corrosivity_class", "ISO 12944-2 corrosivity class C4 (high)",
       "ISO 12944-2. C4 = industrial / coastal.",
       ["C4"]),
    _e("corrosivity_class", "ISO 12944-2 corrosivity class C5 (very high)",
       "ISO 12944-2. C5 (formerly C5-I / C5-M for industrial / marine; "
       "2018 revision merged) = aggressive industrial / marine.",
       ["C5", "C5-I", "C5-M", "C5 I", "C5 M"]),
    _e("corrosivity_class", "ISO 12944-2 corrosivity class CX (extreme)",
       "ISO 12944-2 (2018 addition). CX = extreme offshore / sub-tropical / tropical.",
       ["CX", "C5-IX", "C5-MX"]),

    # ── ISO 12944-1 durability ranges ───────────────────────────────────────
    _e("durability_range", "ISO 12944-1 durability range",
       "L = low (up to 7 yr), M = medium (7-15), H = high (15-25), "
       "VH = very high (>25). Pair with a corrosivity class.",
       ["DURABILITY L", "DURABILITY M", "DURABILITY H", "DURABILITY VH",
        "LOW DURABILITY", "MEDIUM DURABILITY", "HIGH DURABILITY",
        "VERY HIGH DURABILITY"]),

    # ── ISO 12944-5 paint system codes ──────────────────────────────────────
    _e("paint_system", "ISO 12944-5 paint system code",
       "S<digit>.<digit><digit> = specific paint-system reference. "
       "First group identifies the binder family; full table in ISO 12944-5.",
       ["S2.05", "S2.07", "S3.02", "S3.09", "S5.01", "S5.03", "S5.08"]),

    # ── ISO 1461 — Hot-dip galvanizing ──────────────────────────────────────
    _e("galvanizing", "ISO 1461 — Hot-dip galvanized coating",
       "Specifies coating thickness by article thickness category. "
       "Common minimum thickness 45-85 µm depending on substrate.",
       ["ISO 1461", "HDG", "HOT DIP GALVANIZED", "HOT-DIP GALVANIZED",
        "GALVANIZED ISO 1461"]),

    # ── EN ISO 14713 — guidance on Zn/Al coatings for atmosphere protection ─
    _e("zinc_coating_guidance", "EN ISO 14713 — Zinc/Al coatings guidance",
       "Selection guidance for zinc / aluminium coatings, paired with "
       "corrosivity classes from ISO 9223 / ISO 12944.",
       ["EN ISO 14713", "ISO 14713"]),

    # ── EN 10346 — continuously hot-dip coated steel sheet ──────────────────
    _e("sheet_coating", "EN 10346 — Hot-dip coated continuous sheet",
       "Coating designation Z<weight> for zinc (e.g. Z275 = 275 g/m² total "
       "both sides), ZA = ZnAl, ZM = ZnMgAl, AZ = AlZn, AS = AlSi. "
       "Common on KC sheet metal drawings ('DX51D+Z275').",
       ["Z100", "Z140", "Z200", "Z275", "Z350", "Z450", "Z600",
        "ZA095", "ZA130", "ZM120", "ZM310", "AZ150", "AZ185"]),
    _e("sheet_grade", "EN 10346 — Coated sheet steel grades (DX/HX/DC)",
       "Substrate grade: DX51D / DX52D / DX53D / DX54D for forming, "
       "HX260YD etc. for high-strength, DC01/DC03/DC04 cold-rolled.",
       ["DX51D", "DX52D", "DX53D", "DX54D", "DC01", "DC03", "DC04"]),
]


# ── Regex patterns ──────────────────────────────────────────────────────────

# Corrosivity class with optional industry/marine suffix
CORROSIVITY_RE = re.compile(r"\b(C[1-5X])(?:[-\s]?([IMX]))?\b")

# Paint system code: S<digit>.<2 digits>
PAINT_SYSTEM_RE = re.compile(r"\bS(\d)\.(\d{2})\b")

# Coating designation Z275 etc. (often appears as steel-grade suffix +Z275)
COATING_DESIG_RE = re.compile(r"\b(?:\+)?([ZA][ASMZ]?|AZ|AS)(\d{2,3})\b")

# Coating thickness in µm: "70µm", "70 µm", "70um", "70 micron".
COATING_THICKNESS_RE = re.compile(
    r"\b(\d{2,3})\s*(?:µ|u|micro)m\b", re.IGNORECASE
)


_BY_FORM: dict[str, dict] = {}
for _entry in sorted(
    CORROSION, key=lambda e: -max(len(s) for s in e["surface_forms"])
):
    for _sf in _entry["surface_forms"]:
        _BY_FORM.setdefault(_sf.upper(), _entry)


def lookup(token: str) -> dict | None:
    return _BY_FORM.get(token.upper().strip())


SURFACE_FORMS: frozenset[str] = frozenset(_BY_FORM.keys())
