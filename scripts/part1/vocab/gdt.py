"""ISO 1101 / ASME Y14.5 — GD&T characteristic text labels.

The GD&T symbols themselves (⌖ ⊥ ∥ ⌒ ⌭ etc.) are visual and belong to
the visual channel. But the *English text labels* for the characteristics
often appear in notes and tolerance tables — `TRUE POSITION`, `FLATNESS`,
`PERPENDICULARITY`, `RUNOUT`, `PROFILE OF SURFACE` — and modifiers
(`MAX MATERIAL`, `LMC`, `MMC`, `FREE STATE`) are written out.

Capturing these gives the textual channel a piece of the GD&T signal
without needing the visual channel.

Sourcing (full citation context in thesis_direction.md §7):
    Groover §5.1.2 — Tolerances (covers GD&T characteristics + modifiers)
    ISO 1101:2017 free preview — lists the 14 characteristics
    ASME Y14.5-2018 — US equivalent; free preview of scope
    Wikipedia "Geometric dimensioning and tolerancing" — cross-checks
"""

from __future__ import annotations


def _e(category: str, canonical: str, source: str, surface_forms: list[str]) -> dict:
    return {
        "category": category,
        "canonical": canonical,
        "source": source,
        "surface_forms": surface_forms,
    }


GDT: list[dict] = [
    # ── Form tolerances (no datum needed) ───────────────────────────────────
    _e("form", "Straightness",
       "ISO 1101 §17.2; Groover §5.1.2.",
       ["STRAIGHTNESS"]),
    _e("form", "Flatness",
       "ISO 1101 §17.3; Groover §5.1.2.",
       ["FLATNESS"]),
    _e("form", "Circularity (roundness)",
       "ISO 1101 §17.4; Groover §5.1.2.",
       ["CIRCULARITY", "ROUNDNESS"]),
    _e("form", "Cylindricity",
       "ISO 1101 §17.5; Groover §5.1.2.",
       ["CYLINDRICITY"]),
    _e("form", "Profile of a line",
       "ISO 1101 §17.6.",
       ["PROFILE OF A LINE", "LINE PROFILE"]),
    _e("form", "Profile of a surface",
       "ISO 1101 §17.7.",
       ["PROFILE OF A SURFACE", "PROFILE OF SURFACE", "SURFACE PROFILE"]),

    # ── Orientation tolerances ──────────────────────────────────────────────
    _e("orientation", "Parallelism",
       "ISO 1101 §18.2; Groover §5.1.2.",
       ["PARALLELISM", "PARALLEL"]),
    _e("orientation", "Perpendicularity",
       "ISO 1101 §18.3.",
       ["PERPENDICULARITY", "PERPENDICULAR", "SQUARENESS"]),
    _e("orientation", "Angularity",
       "ISO 1101 §18.4.",
       ["ANGULARITY"]),

    # ── Location tolerances ─────────────────────────────────────────────────
    _e("location", "Position (true position)",
       "ISO 1101 §19.2; ASME Y14.5 §7.3. Most-used GD&T callout for "
       "hole patterns.",
       ["POSITION", "TRUE POSITION", "POSITIONAL TOLERANCE"]),
    _e("location", "Concentricity",
       "ISO 1101 §19.3 (deprecated in ASME Y14.5-2018, retained in ISO).",
       ["CONCENTRICITY", "CONCENTRIC"]),
    _e("location", "Symmetry",
       "ISO 1101 §19.4 (deprecated in ASME Y14.5-2018).",
       ["SYMMETRY", "SYMMETRICAL"]),

    # ── Runout tolerances ───────────────────────────────────────────────────
    _e("runout", "Circular runout",
       "ISO 1101 §20.2.",
       ["CIRCULAR RUNOUT", "RUNOUT"]),
    _e("runout", "Total runout",
       "ISO 1101 §20.3.",
       ["TOTAL RUNOUT"]),

    # ── Modifiers (material condition + special states) ─────────────────────
    _e("modifier", "Maximum material condition (MMC, Ⓜ)",
       "ISO 2692; ASME Y14.5 §3.1 §6.",
       ["MMC", "MAXIMUM MATERIAL", "MAX MATERIAL", "MAX MATL"]),
    _e("modifier", "Least material condition (LMC, Ⓛ)",
       "ISO 2692; ASME Y14.5.",
       ["LMC", "LEAST MATERIAL"]),
    _e("modifier", "Regardless of feature size (RFS)",
       "ASME Y14.5 — default condition unless MMC/LMC specified.",
       ["RFS", "REGARDLESS OF FEATURE SIZE"]),
    _e("modifier", "Projected tolerance zone (Ⓟ)",
       "ISO 10578; ASME Y14.5 §10.",
       ["PROJECTED TOLERANCE", "PROJECTED TOL"]),
    _e("modifier", "Free state (Ⓕ)",
       "ASME Y14.5 — applies to non-rigid parts measured unrestrained.",
       ["FREE STATE"]),
    _e("modifier", "Tangent plane (Ⓣ)",
       "ASME Y14.5 §6.5.",
       ["TANGENT PLANE"]),

    # ── Datum referencing terms ─────────────────────────────────────────────
    _e("datum", "Datum feature",
       "ISO 5459; ASME Y14.5 §4.",
       ["DATUM", "DATUM FEATURE", "DATUM REFERENCE"]),
    _e("datum", "Datum target",
       "ISO 5459.",
       ["DATUM TARGET"]),

    # ── Common tolerance-zone qualifiers ────────────────────────────────────
    _e("qualifier", "All around",
       "ISO 1101.",
       ["ALL AROUND", "ALL-AROUND"]),
    _e("qualifier", "Between (two-point indicator)",
       "ISO 1101.",
       ["BETWEEN"]),
    _e("qualifier", "Continuous feature",
       "ASME Y14.5 §2.7.3.",
       ["CONTINUOUS FEATURE", "CF"]),
]


_BY_FORM: dict[str, dict] = {}
for _entry in sorted(GDT, key=lambda e: -max(len(s) for s in e["surface_forms"])):
    for _sf in _entry["surface_forms"]:
        _BY_FORM.setdefault(_sf.upper(), _entry)


def lookup(token: str) -> dict | None:
    return _BY_FORM.get(token.upper().strip())


SURFACE_FORMS: frozenset[str] = frozenset(_BY_FORM.keys())
