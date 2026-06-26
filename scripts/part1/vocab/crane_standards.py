"""Crane design standards — Konecranes-relevant domain vocabulary.

Konecranes drawings characterise components by their crane-design class
(hoisting / load groups). If these callouts appear in our corpus, they're
high-value features both for similarity (matching crane components to
crane components) and for BOM (component selection often keyed on the
class label).

Sourcing (full citation context in thesis_direction.md §7):
    FEM 1.001 — European Materials Handling Federation 'Rules for the
                design of hoisting appliances'. Free 3rd-edition PDF is
                widely circulated in the crane industry.
    EN 13001  — Cranes — General Design. Free preview lists the part
                numbering (-1 General, -2 Load actions, -3-x Limit states).
    ISO 4301  — Cranes — Classification (older companion to FEM 1.001).
    Konecranes / Demag public technical guides reproduce the class tables.

This module captures the *named classes* — exact appearance on drawings
needs to be confirmed by corpus inspection (run inspect_vocab_hits.py).
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


CRANE: list[dict] = [
    # ── FEM 1.001 / ISO 4301 mechanism groups (M-classes) ───────────────────
    # NOTE: bare M1..M8 / A1..A8 / L1..L4 / Q1..Q4 / T0..T9 are NOT in
    # surface_forms because they collide with paper sizes (A4 = ISO 216 sheet
    # format) and other 2-char drawing notations. Use FEM_GROUP_RE / FEM_CLASS_RE
    # below with a required "FEM" / "ISO 4301" / "GROUP" / "CLASS" anchor
    # in surrounding text. Corpus-check 2026-05-23 caught 150/166 false
    # positives from bare-A-letter sweeps (all sheet-size callouts).
    _e("mechanism_group", "FEM 1.001 / ISO 4301 mechanism group",
       "FEM 1.001 Booklet 2 + ISO 4301-1. M1=lightest duty, M8=heaviest. "
       "Combines load spectrum class (L1..L4) with hours of use class (T0..T9). "
       "Match only via context-anchored regex (see CONTEXT_FEM_RE).",
       ["1Bm", "1Am", "2m", "3m", "4m", "5m"]),

    _e("structure_group", "FEM 1.001 / ISO 4301 structure (appliance) group",
       "FEM 1.001 Booklet 2 + ISO 4301-1. A1..A8 = whole-appliance "
       "classification by load spectrum × usage frequency. "
       "Match only via context-anchored regex.",
       []),

    # ── EN 13001 design-standard parts ──────────────────────────────────────
    _e("design_standard", "EN 13001 — Cranes, general design",
       "EN 13001-1 (General principles), -2 (Load actions), -3-x (Limit "
       "states for structures / wire ropes / wheels / hooks).",
       ["EN 13001", "EN 13001-1", "EN 13001-2", "EN 13001-3-1", "EN 13001-3-2",
        "EN 13001-3-3", "EN 13001-3-5"]),

    # ── EN 14492 specific subsystems ────────────────────────────────────────
    _e("subsystem_standard", "EN 14492 — Power-driven winches and hoists",
       "EN 14492-1 (winches), -2 (hoists).",
       ["EN 14492", "EN 14492-1", "EN 14492-2"]),

    # ── Hoisting / load classes (alternative letter forms) ──────────────────
    _e("load_class", "FEM 1.001 load spectrum class",
       "L1=light (rare full load), L2=medium, L3=heavy, L4=very heavy. "
       "Bare L1..L4 / Q1..Q4 omitted from surface_forms (collide with L-grade "
       "line-pipe steels and many other 2-char codes); match via context regex.",
       []),
    _e("usage_class", "FEM 1.001 usage / total operating time class",
       "T0..T9 = increasing total hours of use over service life. "
       "Bare T0..T9 omitted from surface_forms (high collision risk); "
       "match via context regex.",
       []),

    # ── Crane-component specific standards ──────────────────────────────────
    _e("component_standard", "DIN 15400 — Crane hooks",
       "Hook-shank dimensions, materials, capacities.",
       ["DIN 15400", "DIN 15401", "DIN 15402"]),
    _e("component_standard", "DIN 15020 — Wire rope drives, lifetime calculation",
       "Wire-rope design for crane hoists.",
       ["DIN 15020", "DIN 15020-1", "DIN 15020-2"]),
    _e("component_standard", "DIN 15018 — Steel structures of cranes",
       "Older German standard (largely superseded by EN 13001 but still cited).",
       ["DIN 15018", "DIN 15018-1"]),

    # ── ISO 4308 — Wire ropes for cranes ────────────────────────────────────
    _e("component_standard", "ISO 4308 — Cranes and lifting appliances: wire rope selection",
       "ISO 4308-1 (general), -2 (mobile cranes).",
       ["ISO 4308", "ISO 4308-1", "ISO 4308-2"]),
]


# ── Regex patterns ──────────────────────────────────────────────────────────

# Context-anchored patterns — require a FEM/ISO 4301/GROUP/CLASS anchor in the
# preceding ~30 chars. This avoids the paper-size collision (A4 = ISO 216
# sheet format) that produced 150/166 false positives in the 2026-05-23 corpus
# check. Use these patterns; do NOT match bare A4 / M5 / L3 etc. alone.

# Mechanism / structure group with FEM/ISO 4301/GROUP/CLASS anchor.
CONTEXT_FEM_RE = re.compile(
    r"(?:FEM(?:\s*1[.\-]?001)?|ISO\s*4301|GROUP|CLASS|HOIST\w*\s+CLASS)"
    r"[\s:=-]*([MA])([1-8])\b",
    re.IGNORECASE,
)

# Load / usage classes (L1..L4, T0..T9, Q1..Q4) with similar anchor.
CONTEXT_FEM_CLASS_RE = re.compile(
    r"(?:FEM|ISO\s*4301|LOAD\s+SPECTRUM|USAGE)[\s:=-]*([LTQ])([0-9])\b",
    re.IGNORECASE,
)


_BY_FORM: dict[str, dict] = {}
# Some entries deliberately have an empty surface_forms list (context-anchored
# regex-only matching, e.g. FEM group classes). Filter them out of the sort.
for _entry in sorted(
    (e for e in CRANE if e["surface_forms"]),
    key=lambda e: -max(len(s) for s in e["surface_forms"]),
):
    for _sf in _entry["surface_forms"]:
        _BY_FORM.setdefault(_sf.upper(), _entry)


def lookup(token: str) -> dict | None:
    return _BY_FORM.get(token.upper().strip())


SURFACE_FORMS: frozenset[str] = frozenset(_BY_FORM.keys())
