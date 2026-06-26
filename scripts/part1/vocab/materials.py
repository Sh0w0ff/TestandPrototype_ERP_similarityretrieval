"""Material-grade vocabulary — strongest similarity feature.

Extends the existing `S<yield>` steel-grade regex in extract_drawing.py
to the full material taxonomy: structural + tool + stainless steels,
cast iron, aluminium, plastics, plus AISI/SAE US designations.

Sourcing (full citation context in thesis_direction.md §7):
    Groover Ch.6 — Metals (steel, cast iron, aluminium, non-ferrous)
    Groover Ch.7 — Ceramics
    Groover Ch.8 — Polymers (lists ISO 1043 abbreviations)
    Klocke V1 §7.3–7.7 — Machinability sections enumerate steel/cast-iron/
                          non-ferrous/non-metal grades by designation
    EN/ISO free  — Standard scope pages list designation grammar (numbering
    previews        ranges, prefix letters)
"""

from __future__ import annotations

import re


def _e(standard: str, canonical: str, source: str,
       surface_forms: list[str], family: str = "metal") -> dict:
    return {
        "standard": standard,
        "canonical": canonical,
        "family": family,
        "source": source,
        "surface_forms": surface_forms,
    }


MATERIALS: list[dict] = [
    # ── EN 10027-1 structural steels (S-prefix, P-prefix, L-prefix) ─────────
    _e("EN 10027-1", "Structural steel (S-grade, yield-specified)",
       "Groover §6.2 (Ferrous Metals); Klocke V1 §7.4.1 (Machining Steels). "
       "S-prefix denotes structural steel, number = min yield in MPa.",
       ["S185", "S235JR", "S235J0", "S235J2", "S275JR", "S275J0", "S275J2",
        "S355JR", "S355J0", "S355J2", "S355K2", "S355J2+N", "S355NL", "S420N",
        "S460N", "S460NL", "S690QL"], family="steel_structural"),
    _e("EN 10027-1", "Pressure-vessel steel (P-grade)",
       "Groover §6.2; EN 10028. P-prefix = pressure-vessel steel.",
       ["P235GH", "P265GH", "P295GH", "P355GH", "P460NH"], family="steel_pressure"),
    _e("EN 10027-1", "Line-pipe steel (L-grade)",
       "EN 10208 / API 5L equivalents. L-prefix = line-pipe steel.",
       ["L245", "L290", "L360", "L415", "L450", "L485"], family="steel_pipe"),

    # ── EN 10027-2 material numbers (1.xxxx system) ─────────────────────────
    # These are numeric — covered by regex below — but a few common ones are
    # listed for direct surface-form lookup since they appear verbatim.
    _e("EN 10027-2", "Material number (1.xxxx system)",
       "EN 10027-2 numbering: 1.xxxx where the first 4 digits encode steel "
       "group; e.g. 1.0038=S235JR, 1.0570=S355J2, 1.4301=304 stainless.",
       ["1.0038", "1.0570", "1.0577", "1.4301", "1.4307", "1.4401", "1.4404",
        "1.4571", "1.7225", "1.2379", "1.8159"], family="steel_matnum"),

    # ── EN 10088 stainless steels (1.4xxx) ──────────────────────────────────
    # NOTE: bare 3-digit AISI numbers (304/316/430/420) and AISI tool-steel
    # codes (D2/D3/H11/H13) were REMOVED 2026-05-23 — corpus check showed
    # they collide with dimensions in mm (e.g. L420W142T8) and ISO 286
    # tolerance codes (H11 in dimension columns). Only the unambiguous
    # 1.xxxx EN material numbers and the X<spec> EN names are kept.
    _e("EN 10088", "Austenitic stainless steel",
       "Groover §6.2.3 (Stainless steels); Klocke V1 §7.4.7 (Non-rusting steels). "
       "Bare AISI 304/316 omitted (collides with 3-digit dimensions).",
       ["1.4301", "1.4307", "1.4401", "1.4404", "1.4571",
        "X5CrNi18-10", "X2CrNiMo17-12-2"],
       family="steel_stainless"),
    _e("EN 10088", "Ferritic / martensitic stainless steel",
       "Groover §6.2.3. Bare AISI 420/430/440C omitted (collides with "
       "3-digit dimensions; see 2026-05-23 corpus check).",
       ["1.4016", "1.4021", "1.4034", "1.4125"],
       family="steel_stainless"),

    # ── Tool steels ─────────────────────────────────────────────────────────
    _e("EN ISO 4957", "Tool steel (cold-work / hot-work / HSS)",
       "Klocke V1 §4.2 (Tool Steels) + §7.4.5. Bare AISI codes "
       "(D2/D3/H11/H13) omitted — H11/H13 collide with ISO 286 tolerance "
       "codes, D2/D3 with part-number substrings.",
       ["1.2379", "1.2210", "1.2842", "1.2510", "1.2767", "1.2343", "1.2344",
        "X155CrVMo12-1", "X40CrMoV5-1"],
       family="steel_tool"),

    # ── EN 1561 / 1563 / 1564 cast iron ─────────────────────────────────────
    _e("EN 1561", "Lamellar (grey) cast iron",
       "Groover §6.2.5 (Cast Irons); Klocke V1 §7.5.2 (Grey Cast Iron). "
       "Format: EN-GJL-<tensile MPa>.",
       ["EN-GJL-150", "EN-GJL-200", "EN-GJL-250", "EN-GJL-300", "EN-GJL-350",
        "GG-15", "GG-20", "GG-25", "GG-30"], family="cast_iron"),
    _e("EN 1563", "Spheroidal (nodular/ductile) cast iron",
       "Groover §6.2.5. Format: EN-GJS-<tensile>-<elongation>.",
       ["EN-GJS-400-15", "EN-GJS-400-18", "EN-GJS-500-7", "EN-GJS-600-3",
        "EN-GJS-700-2", "GGG-40", "GGG-50", "GGG-60"], family="cast_iron"),
    _e("EN 1564", "Austempered ductile iron (ADI)",
       "Groover §6.2.5. Higher-strength heat-treated nodular iron.",
       ["EN-GJS-800-8", "EN-GJS-1000-5", "EN-GJS-1200-2"], family="cast_iron"),

    # ── EN 573 / ISO 209 aluminium alloys ───────────────────────────────────
    _e("EN 573", "Aluminium wrought alloy",
       "Groover §6.3 (Nonferrous Metals — Aluminum); Klocke V1 §7.6.1.",
       ["EN AW-1050A", "EN AW-2024", "EN AW-3003", "EN AW-5052", "EN AW-5083",
        "EN AW-5754", "EN AW-6005A", "EN AW-6061", "EN AW-6063", "EN AW-6082",
        "EN AW-7020", "EN AW-7075",
        # Bare four-digit forms also common in US-style notation:
        "AL 6061", "AL 6082", "AL 7075"], family="aluminium"),

    # ── Brass / bronze / copper ─────────────────────────────────────────────
    _e("EN 12420", "Brass (CuZn alloys)",
       "Groover §6.3.4 (Copper and Its Alloys).",
       ["CuZn37", "CuZn40Pb2", "CuZn39Pb3", "CW508L", "CW617N"], family="brass"),
    _e("EN 1982", "Bronze (CuSn / CuAl alloys)",
       "Groover §6.3.4.",
       ["CuSn8", "CuSn12", "CuAl10Fe3", "CC480K", "CB495K"], family="bronze"),

    # ── ISO 1043-1 plastics ─────────────────────────────────────────────────
    # NOTE: bare 2-letter polymer abbreviations (PA, PE, PP, PC, PS, PET)
    # were REMOVED 2026-05-23 — they collide with title-block abbreviations,
    # part-number substrings, and engineering shorthand (PC ≠ polycarbonate
    # in many contexts). Only ≥3-char unambiguous abbreviations kept.
    _e("ISO 1043-1", "Plastic — polyamide family",
       "Groover §8.2 (Thermoplastic Polymers); ISO 1043 abbreviation system. "
       "Bare 'PA' omitted (high collision risk).",
       ["PA6", "PA66", "PA11", "PA12", "PA46"], family="plastic"),
    _e("ISO 1043-1", "Plastic — polyolefin family",
       "Groover §8.2. Bare 'PE'/'PP' omitted (collision risk); only "
       "qualified forms kept.",
       ["HDPE", "LDPE", "UHMWPE"], family="plastic"),
    _e("ISO 1043-1", "Plastic — engineering polymers",
       "Groover §8.2. 'PC' omitted (collides with 'piece count' / 'price code' "
       "/ many drawing abbreviations). 'PEI' kept — unambiguous.",
       ["POM", "PBT", "PEEK", "PEI", "PSU", "PPS"], family="plastic"),
    _e("ISO 1043-1", "Plastic — vinyl / styrenic",
       "Groover §8.2. 'PS' omitted (collides with parts-per / postscript / "
       "many other abbreviations).",
       ["PVC", "ABS", "SAN", "HIPS"], family="plastic"),
    _e("ISO 1043-1", "Plastic — fluoropolymer / acrylic",
       "Groover §8.2.",
       ["PTFE", "PVDF", "PMMA"], family="plastic"),
    _e("ISO 11469", "Glass-fibre / mineral reinforced plastic",
       "Reinforcement suffix system: -GF<n> = glass fibre %wt; -MD = mineral.",
       ["PA66-GF30", "PA6-GF15", "PA6-GF30", "PP-GF20", "PBT-GF30"],
       family="plastic_reinforced"),

    # ── AISI / SAE US designations ──────────────────────────────────────────
    # NOTE: bare 4-digit AISI codes (1018/1020/4140 etc.) collide heavily
    # with part numbers and dimensions on these drawings. Kept only the
    # qualified forms ("ASTM A36", "A572 GR50") and AISI prefix forms.
    _e("AISI/SAE", "Carbon / alloy steel (US designation)",
       "Groover §6.2. Bare 4-digit AISI codes (1018/4140/etc.) omitted — "
       "collide with part numbers and dimensions. Use qualified prefix forms.",
       ["AISI 1018", "AISI 1020", "AISI 1045", "AISI 4140", "AISI 4340",
        "AISI 8620", "SAE 4140", "SAE 4340"], family="steel_us"),
    _e("ASTM A36", "Structural carbon steel (US)",
       "ASTM A36 / A572 / A992 — US structural steel standards (≈ S235/S355). "
       "Bare 'A36' omitted (collides with paper-size-style codes); use "
       "'ASTM A36' qualified form.",
       ["ASTM A36", "A572 GR50", "ASTM A572", "ASTM A992"], family="steel_us"),
]


# ── Regex patterns for structured material codes ────────────────────────────

# S-grade steel (already partial in extract_drawing.py — replicated here for
# completeness): S<yield><charpy>[+<state>]
STEEL_S_GRADE_RE = re.compile(r"\bS(\d{3})([A-Z]{1,3}\d?)?(\+[A-Z]{1,2})?\b")

# EN 10027-2 material number: 1.xxxx (4-digit) optionally followed by digit.
EN_MATNUM_RE = re.compile(r"\b1\.[0-9]{4}(?:\.[0-9])?\b")

# EN AW aluminium: EN AW-<4 digits>[<letter>]
EN_AW_RE = re.compile(r"\bEN\s*AW-(\d{4})([A-Z])?\b")

# Cast iron: EN-GJ[LSM]-<digits>(-<digits>)?
CAST_IRON_RE = re.compile(r"\bEN-GJ[LSMV]-(\d{3,4})(?:-(\d+))?\b")

# Stainless number form: 1.4xxx
STAINLESS_NUM_RE = re.compile(r"\b1\.4\d{3}\b")

# Plastic reinforcement: <polymer>-GF<percent> or -MD<percent>
PLASTIC_REINF_RE = re.compile(
    r"\b(PA6|PA66|PA12|PP|PBT|PET|POM|PC|PEEK)-(GF|MD|CF)(\d{2})\b"
)


_BY_FORM: dict[str, dict] = {}
for _entry in sorted(MATERIALS, key=lambda e: -max(len(s) for s in e["surface_forms"])):
    for _sf in _entry["surface_forms"]:
        _BY_FORM.setdefault(_sf.upper(), _entry)


def lookup(token: str) -> dict | None:
    """Lookup a material designation by surface form."""
    return _BY_FORM.get(token.upper().strip())


SURFACE_FORMS: frozenset[str] = frozenset(_BY_FORM.keys())
