"""ISO 4063 welding and allied process numerical reference.

ISO 4063 assigns a number to each welding / cutting / brazing process. The
number is often written verbatim on drawings (e.g. "Welding process 135").
This module encodes the common numbers plus their abbreviations and English
names, so a regex hit on either the number or the abbreviation maps to the
same canonical entry.

Structure per entry:
    code:           ISO 4063 number (string; some are 2-digit families, some 3-digit)
    abbrev:         Common abbreviation (MMA, MAG, TIG, ...)
    canonical:      Standard English name
    surface_forms:  Variants found on drawings or notes

Coverage below is the practical subset for steel-fabrication drawings (the
corpus is ABB + Konecranes crane / structural components). Specialty
processes (electroslag, thermite, etc.) are omitted.
"""

from __future__ import annotations

WELDING: list[dict] = [
    # ── 1xx  Arc welding ────────────────────────────────────────────────────
    {"code": "111", "abbrev": "MMA", "canonical": "Manual metal arc welding (SMAW)",
     "surface_forms": ["MMA", "SMAW", "STICK WELDING", "MANUAL METAL ARC"]},
    {"code": "114", "abbrev": "FCAW-S", "canonical": "Self-shielded flux-cored arc welding",
     "surface_forms": ["FCAW-S", "SELF SHIELDED FLUX CORED"]},
    {"code": "121", "abbrev": "SAW", "canonical": "Submerged arc welding (single wire)",
     "surface_forms": ["SAW", "SUBMERGED ARC", "SUBMERGED ARC WELDING"]},
    {"code": "131", "abbrev": "MIG", "canonical": "Metal inert gas welding",
     "surface_forms": ["MIG", "METAL INERT GAS", "GMAW-I"]},
    {"code": "135", "abbrev": "MAG", "canonical": "Metal active gas welding",
     "surface_forms": ["MAG", "METAL ACTIVE GAS", "GMAW", "CO2 WELDING"]},
    {"code": "136", "abbrev": "FCAW", "canonical": "Flux-cored arc welding with active shielding gas",
     "surface_forms": ["FCAW", "FLUX CORED ARC", "FLUX CORED ARC WELDING"]},
    {"code": "138", "abbrev": "MCAW", "canonical": "Metal-cored arc welding with active shielding gas",
     "surface_forms": ["MCAW", "METAL CORED ARC"]},
    {"code": "141", "abbrev": "TIG", "canonical": "Tungsten inert gas welding",
     "surface_forms": ["TIG", "GTAW", "TUNGSTEN INERT GAS"]},
    {"code": "15",  "abbrev": "PAW", "canonical": "Plasma arc welding",
     "surface_forms": ["PAW", "PLASMA ARC", "PLASMA ARC WELDING"]},

    # ── 2xx  Resistance welding ─────────────────────────────────────────────
    {"code": "21", "abbrev": "RSW", "canonical": "Resistance spot welding",
     "surface_forms": ["RSW", "SPOT WELDED", "SPOT WELDING", "RESISTANCE SPOT"]},
    {"code": "22", "abbrev": "RSEW", "canonical": "Resistance seam welding",
     "surface_forms": ["RSEW", "SEAM WELDED", "SEAM WELDING"]},
    {"code": "23", "abbrev": "RPW", "canonical": "Projection welding",
     "surface_forms": ["RPW", "PROJECTION WELDED", "PROJECTION WELDING"]},
    {"code": "24", "abbrev": "FW",  "canonical": "Flash welding",
     "surface_forms": ["FLASH WELDED", "FLASH WELDING"]},  # NB: FW also = fillet weld; disambiguate by context.
    {"code": "25", "abbrev": "UW",  "canonical": "Resistance butt welding (upset)",
     "surface_forms": ["UPSET WELDING", "RESISTANCE BUTT WELDING"]},

    # ── 3xx  Gas welding ────────────────────────────────────────────────────
    {"code": "311", "abbrev": "OAW", "canonical": "Oxy-acetylene welding",
     "surface_forms": ["OAW", "OXY-ACETYLENE", "OXY ACETYLENE", "GAS WELDED"]},

    # ── 4xx  Solid-state welding ────────────────────────────────────────────
    {"code": "42", "abbrev": "FRW", "canonical": "Friction welding",
     "surface_forms": ["FRICTION WELDED", "FRICTION WELDING"]},
    {"code": "43", "abbrev": "FSW", "canonical": "Friction stir welding",
     "surface_forms": ["FSW", "FRICTION STIR WELDED", "FRICTION STIR WELDING"]},
    {"code": "44", "abbrev": "ULW", "canonical": "Ultrasonic welding",
     "surface_forms": ["ULTRASONIC WELDED", "ULTRASONIC WELDING"]},
    {"code": "47", "abbrev": "GPW", "canonical": "Gas pressure welding",
     "surface_forms": ["GAS PRESSURE WELDING"]},
    {"code": "48", "abbrev": "CW",  "canonical": "Cold pressure welding",
     "surface_forms": ["COLD WELDED", "COLD PRESSURE WELDING"]},

    # ── 5xx  Beam welding ───────────────────────────────────────────────────
    {"code": "51", "abbrev": "EBW", "canonical": "Electron beam welding",
     "surface_forms": ["EBW", "ELECTRON BEAM WELDING", "ELECTRON BEAM WELDED"]},
    {"code": "52", "abbrev": "LBW", "canonical": "Laser beam welding",
     "surface_forms": ["LBW", "LASER BEAM WELDING", "LASER WELDED", "LASER WELDING"]},

    # ── 7xx  Other welding ──────────────────────────────────────────────────
    {"code": "78", "abbrev": "SW", "canonical": "Stud welding",
     "surface_forms": ["STUD WELDED", "STUD WELDING"]},  # NB: SW also = square weld in some callouts.

    # ── 9xx  Brazing, soldering, braze-welding ──────────────────────────────
    {"code": "91", "abbrev": "B",  "canonical": "Brazing",
     "surface_forms": ["BRAZED", "BRAZING"]},
    {"code": "94", "abbrev": "S",  "canonical": "Soldering",
     "surface_forms": ["SOLDERED", "SOLDERING"]},
    {"code": "97", "abbrev": "BW", "canonical": "Braze welding",
     "surface_forms": ["BRAZE WELDED", "BRAZE WELDING"]},  # NB: BW also = butt weld in callouts.
]


# Disambiguation note:
# The single-letter / two-letter weld-symbol abbreviations FW, BW, SW collide
# with weld-joint-type callouts (Fillet Weld, Butt Weld, Square Weld) used in
# extract_drawing.py's WELD_CALLOUT_PAT. Consumers must disambiguate by
# context: an isolated "FW 5" near a weld symbol = fillet weld throat 5; a
# "PROCESS: FW" or "ISO 4063 24" = flash welding. Do NOT collapse these.

_BY_CODE: dict[str, dict] = {e["code"]: e for e in WELDING}
_BY_FORM: dict[str, dict] = {}
for _e in sorted(WELDING, key=lambda e: -max(len(s) for s in e["surface_forms"])):
    for _sf in _e["surface_forms"]:
        _BY_FORM.setdefault(_sf, _e)


def lookup(token: str) -> dict | None:
    """Return ISO 4063 entry for a code ('135') or surface form ('MAG'), or None."""
    t = token.strip().upper()
    return _BY_CODE.get(t) or _BY_FORM.get(t)


SURFACE_FORMS: frozenset[str] = frozenset(_BY_FORM.keys())
CODES: frozenset[str] = frozenset(_BY_CODE.keys())
