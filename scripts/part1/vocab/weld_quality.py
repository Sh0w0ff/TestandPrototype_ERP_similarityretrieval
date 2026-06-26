"""Weld-quality and welded-structure tolerance vocabulary.

Complements `iso4063.py` (which captures the welding *process* number).
This module captures the *quality and tolerance* callouts that determine
how good the weld must be:

    ISO 5817 - B          stringent quality (limited imperfections)
    ISO 5817 - C          intermediate
    ISO 5817 - D          moderate
    ISO 13920 - A         tightest welded-structure dimensional tolerance
    ISO 13920 - BF        general tolerance class B (dimensions) + F (angles)

Sourcing (full citation context in thesis_direction.md §7):
    Groover §30 — Welding Processes (background only; does not enumerate
                  ISO 5817 / 13920 quality classes).
    ISO 5817 / 13920 free preview — defines the quality / tolerance class
                                    letters and their meaning.
    Welding-society material (TWI, IIW) — public-web articles enumerate
                                          ISO 5817 B/C/D acceptance criteria.
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


WELD_QUALITY: list[dict] = [
    # ── ISO 5817 quality levels (weld imperfection severity) ────────────────
    _e("quality_level", "ISO 5817 quality level B (stringent)",
       "ISO 5817 — Quality levels for imperfections in fusion-welded "
       "joints in steel/nickel/titanium. Level B = highest, lowest "
       "tolerance for imperfections.",
       ["ISO 5817 B", "ISO 5817-B", "EN 25817 B", "QUALITY B"]),
    _e("quality_level", "ISO 5817 quality level C (intermediate)",
       "ISO 5817 — Most common shop-floor quality class.",
       ["ISO 5817 C", "ISO 5817-C", "EN 25817 C", "QUALITY C"]),
    _e("quality_level", "ISO 5817 quality level D (moderate)",
       "ISO 5817 — Allows more imperfections; non-critical welds.",
       ["ISO 5817 D", "ISO 5817-D", "EN 25817 D", "QUALITY D"]),

    # ── ISO 13920 welded-structure dimensional / angular tolerances ─────────
    _e("dim_tolerance", "ISO 13920 tolerance class A (dimensions, tight)",
       "ISO 13920 — General tolerances for welded constructions, linear "
       "and angular dimensions. Class A = tightest.",
       ["ISO 13920 A", "ISO 13920-A"]),
    _e("dim_tolerance", "ISO 13920 tolerance class B (medium)",
       "ISO 13920 — most common.",
       ["ISO 13920 B", "ISO 13920-B"]),
    _e("dim_tolerance", "ISO 13920 tolerance class C (coarse)",
       "ISO 13920.",
       ["ISO 13920 C", "ISO 13920-C"]),
    _e("dim_tolerance", "ISO 13920 tolerance class D (very coarse)",
       "ISO 13920.",
       ["ISO 13920 D", "ISO 13920-D"]),
    _e("ang_tolerance", "ISO 13920 angular tolerance class E/F/G/H",
       "ISO 13920 angular-tolerance letter codes. Often paired with a "
       "dimensional class (e.g. 'ISO 13920-BF').",
       ["ISO 13920 E", "ISO 13920 F", "ISO 13920 G", "ISO 13920 H",
        "ISO 13920-BF", "ISO 13920-CG"]),

    # ── EN 1090 execution classes (steel structures) ────────────────────────
    _e("execution_class", "EN 1090 execution class (steel structures)",
       "EN 1090-2 — Execution of steel structures. Classes EXC1..EXC4 "
       "denote increasing inspection/quality requirements (EXC4 = strictest, "
       "e.g. bridges, dynamically loaded structures).",
       ["EXC1", "EXC 1", "EXC2", "EXC 2", "EXC3", "EXC 3", "EXC4", "EXC 4",
        "EN 1090"]),

    # ── Inspection / NDT callouts ───────────────────────────────────────────
    _e("inspection", "Visual testing (VT)",
       "ISO 17637 (visual inspection of welds).",
       ["VT", "VISUAL TESTING", "VISUAL INSPECTION"]),
    _e("inspection", "Penetrant testing (PT)",
       "ISO 3452.",
       ["PT", "PENETRANT TESTING", "DYE PENETRANT"]),
    _e("inspection", "Magnetic particle testing (MT)",
       "ISO 17638.",
       ["MT", "MAGNETIC PARTICLE", "MAGNAFLUX"]),
    _e("inspection", "Radiographic testing (RT)",
       "ISO 17636.",
       ["RT", "RADIOGRAPHIC TESTING", "X-RAY", "RADIOGRAPHY"]),
    _e("inspection", "Ultrasonic testing (UT)",
       "ISO 17640.",
       ["UT", "ULTRASONIC TESTING", "ULTRASONIC"]),
]


# ── Regex patterns ──────────────────────────────────────────────────────────

# ISO 5817 quality letter: "ISO 5817" + space/dash + B/C/D
ISO_5817_RE = re.compile(r"\b(?:ISO|EN)\s*(?:5817|25817)\s*[-\s]?\s*([BCD])\b")

# ISO 13920 tolerance letter(s): single dim letter (A-D) or paired (e.g. "BF")
ISO_13920_RE = re.compile(r"\bISO\s*13920\s*[-\s]?\s*([A-D])([E-H])?\b")

# EN 1090 execution class
EXC_RE = re.compile(r"\bEXC\s*([1-4])\b")


_BY_FORM: dict[str, dict] = {}
for _entry in sorted(
    WELD_QUALITY, key=lambda e: -max(len(s) for s in e["surface_forms"])
):
    for _sf in _entry["surface_forms"]:
        _BY_FORM.setdefault(_sf.upper(), _entry)


def lookup(token: str) -> dict | None:
    return _BY_FORM.get(token.upper().strip())


SURFACE_FORMS: frozenset[str] = frozenset(_BY_FORM.keys())
