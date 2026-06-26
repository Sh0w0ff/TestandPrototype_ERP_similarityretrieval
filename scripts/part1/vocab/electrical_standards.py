"""Electrical-machine standards — ABB Drives-relevant domain vocabulary.

ABB Drives drawings often carry IEC standard callouts for motor housings,
frame sizes, efficiency classes, and IP/cooling ratings. These are
high-value features for similarity (drawings of equivalent motor frames
across product variants cluster) and for BOM (frame size constrains
mounting hardware selection).

Sourcing (full citation context in thesis_direction.md §7):
    IEC 60034   — Rotating electrical machines (multi-part series).
                  Free preview pages enumerate the parts and IE-class table.
    IEC 60072   — Dimensions and output series for rotating electrical
                  machines (frame-number → mounting-flange dimensions).
    IEC 60204   — Electrical equipment of machines.
    IEC 60529   — IP (ingress protection) rating system.
    ABB technical guides (public) cross-reference these.

This module captures the *named codes*. Corpus presence to be confirmed
with inspect_vocab_hits.py.
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


ELECTRICAL: list[dict] = [
    # ── IEC 60034 — Rotating electrical machines ────────────────────────────
    _e("design_standard", "IEC 60034 — Rotating electrical machines",
       "Multi-part: -1 (rating + performance), -2 (efficiency), -5 (IP), "
       "-6 (cooling), -7 (construction/mounting), -8 (terminal marking), "
       "-30 (IE classes).",
       ["IEC 60034", "IEC 60034-1", "IEC 60034-2", "IEC 60034-5", "IEC 60034-6",
        "IEC 60034-7", "IEC 60034-8", "IEC 60034-30"]),

    # ── IE efficiency classes (IEC 60034-30) ────────────────────────────────
    _e("efficiency_class", "IEC 60034-30 IE efficiency class",
       "IE1=standard, IE2=high, IE3=premium, IE4=super-premium, IE5=ultra. "
       "Mandatory marking on industrial motors in EU since 2011 (ErP Directive).",
       ["IE1", "IE2", "IE3", "IE4", "IE5"]),

    # ── IEC 60072 — Frame sizes ─────────────────────────────────────────────
    _e("frame_size", "IEC 60072 frame size (motor shaft height)",
       "IEC 60072-1/-2. Frame number = shaft height in mm. Standard "
       "sizes 56, 63, 71, 80, 90 (S/L), 100 (L), 112 (M), 132 (S/M), "
       "160 (M/L), 180 (M/L), 200 (L), 225 (S/M), 250, 280, 315, 355, "
       "400, 450, 500, 560, 630, 710, 800.",
       ["IEC 56", "IEC 63", "IEC 71", "IEC 80", "IEC 90", "IEC 100", "IEC 112",
        "IEC 132", "IEC 160", "IEC 180", "IEC 200", "IEC 225", "IEC 250",
        "IEC 280", "IEC 315", "IEC 355", "IEC 400", "IEC 450", "IEC 500",
        # ABB internal often writes "M2BAX 132", "M3BP 160" — frame number
        # appears bare. The regex below catches these.
       ]),

    # ── Mounting designations (IEC 60034-7) ─────────────────────────────────
    _e("mounting", "IEC 60034-7 mounting designation (IM-code)",
       "Codes like IM B3 (foot mount), IM B5 (flange mount), IM B14 "
       "(face mount), IM V1 (vertical, shaft down).",
       ["IM B3", "IM B5", "IM B14", "IM B35", "IM V1", "IM V5", "IM V6",
        "IM 1001", "IM 3001", "IM 2001"]),

    # ── IEC 60529 — Ingress protection (IP rating) ──────────────────────────
    _e("ingress_protection", "IEC 60529 IP rating (ingress protection)",
       "Two digits: first = solid-particle protection (0-6), second = "
       "liquid protection (0-9). Common motor ratings IP55, IP56, IP65, IP66.",
       ["IP00", "IP20", "IP21", "IP22", "IP23", "IP44", "IP54", "IP55",
        "IP56", "IP65", "IP66", "IP67", "IP68"]),

    # ── Insulation class (IEC 60085) ────────────────────────────────────────
    _e("insulation_class", "IEC 60085 thermal insulation class",
       "Letters A/E/B/F/H/N/R/S = increasing max winding temperature. "
       "Most industrial motors use Class F (155 °C) with Class B (130 °C) "
       "temperature rise.",
       ["CLASS A", "CLASS E", "CLASS B", "CLASS F", "CLASS H", "CLASS N",
        "CLASS R", "CLASS S", "INSULATION F", "INSULATION B"]),

    # ── Duty cycles (IEC 60034-1) ───────────────────────────────────────────
    _e("duty_cycle", "IEC 60034-1 duty cycle (S-code)",
       "S1=continuous, S2=short-time, S3=intermittent periodic, S4..S10 "
       "= variants. Drives whole product family categorisation.",
       ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]),

    # ── Cooling designations (IEC 60034-6) ──────────────────────────────────
    _e("cooling", "IEC 60034-6 cooling code (IC-code)",
       "Codes like IC411 (TEFC — totally enclosed fan-cooled), IC01 "
       "(open self-ventilated), IC418 (forced air).",
       ["IC00", "IC01", "IC06", "IC410", "IC411", "IC416", "IC418", "IC511",
        "IC611", "TEFC", "ODP"]),

    # ── IEC 60204 — Electrical equipment of machines ────────────────────────
    _e("machine_electrical", "IEC 60204 — Electrical equipment of machines",
       "Safety / general requirements for industrial machine electrics. "
       "Often cited on cabinet drawings.",
       ["IEC 60204", "IEC 60204-1"]),

    # ── EN 61800 — Adjustable-speed electrical power drive systems ──────────
    _e("drive_system", "EN/IEC 61800 — Adjustable-speed drives",
       "EN 61800-3 (EMC requirements), -5-1 (safety), -9 (efficiency). "
       "Direct ABB Drives relevance.",
       ["IEC 61800", "EN 61800", "IEC 61800-3", "IEC 61800-5-1", "EN 61800-9"]),
]


# ── Regex patterns ──────────────────────────────────────────────────────────

# IE class alone
IE_CLASS_RE = re.compile(r"\bIE([1-5])\b")

# IP rating
IP_RATING_RE = re.compile(r"\bIP\s*([0-6])([0-9])\b")

# IM mounting code (letter form, e.g. "IM B3"; numeric form e.g. "IM 1001")
IM_CODE_RE = re.compile(r"\bIM\s*(?:([BV])\s*(\d{1,2})|(\d{4}))\b")

# Duty cycle S-code
DUTY_CYCLE_RE = re.compile(r"\bS([1-9]|10)\b")


_BY_FORM: dict[str, dict] = {}
for _entry in sorted(
    ELECTRICAL, key=lambda e: -max(len(s) for s in e["surface_forms"])
):
    for _sf in _entry["surface_forms"]:
        _BY_FORM.setdefault(_sf.upper(), _entry)


def lookup(token: str) -> dict | None:
    return _BY_FORM.get(token.upper().strip())


SURFACE_FORMS: frozenset[str] = frozenset(_BY_FORM.keys())
