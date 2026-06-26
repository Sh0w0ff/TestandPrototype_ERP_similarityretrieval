"""DIN 8580 manufacturing process taxonomy.

DIN 8580 is the umbrella German standard that classifies all manufacturing
processes into 6 main groups. Each main group has its own *standalone*
sub-standard within the DIN 858x family — Klocke 2013 Vol 4 Ch 1 p.1 states:
"Forming processes are summarized in DIN 8582 and grouped in accordance
with their 'strain', i.e. the predominant stresses." The family is:

    Group  Sub-standard   Domain                              Klocke vol
    1      DIN 8581       Primary forming (Urformen)          Vol 5 (not held)
    2      DIN 8582       Forming         (Umformen)          Vol 4 ✓
    3      DIN 8583       Separating      (Trennen)           Vol 1 ✓ / Vol 2 / Vol 3
    4      DIN 8584       Joining         (Fügen)             —
    5      DIN 8585       Coating         (Beschichten)       —
    6      DIN 8587       Changing material properties        —

We have NOT verified the dotted internal subcodes inside each 858x standard
(e.g. whether "drilling" sits at DIN 8589-2 or some other number). Earlier
revisions of this module fabricated subcodes like "3.2.1" from memory; those
have been removed. Each entry now carries only the *verified* group number
plus the sub-standard reference, and a `source` field naming the secondary
literature we used to confirm the process belongs in that group.

Sources used (full citation context in thesis_direction.md §7):
    Klocke V1   = Klocke F. — Manufacturing Processes 1: Cutting (Springer/RWTH 2011)
    Klocke V4   = Klocke F. — Manufacturing Processes 4: Forming (Springer/RWTH 2013)
    Groover     = Groover M.P. — Fundamentals of Modern Manufacturing 4e (Wiley 2010)
    TWI         = TWI (twi-global.com) — for ISO 4063 numbers (see iso4063.py)

Per-entry structure:
    group:          DIN 8580 main group number (1..6) — VERIFIED
    sub_standard:   DIN 858x reference for the sub-standard — VERIFIED group→standard mapping
    group_name:     Canonical group label
    canonical:      Standard English name of the process
    source:         Citation tag for where we confirmed the process / its grouping
    surface_forms:  Text variants found on drawings (uppercase)
"""

from __future__ import annotations

# Group → (sub-standard reference, canonical group name).
GROUP_INFO: dict[int, tuple[str, str]] = {
    1: ("DIN 8581", "Primary forming (Urformen)"),
    2: ("DIN 8582", "Forming (Umformen)"),
    3: ("DIN 8583", "Separating (Trennen)"),
    4: ("DIN 8584", "Joining (Fügen)"),
    5: ("DIN 8585", "Coating (Beschichten)"),
    6: ("DIN 8587", "Changing material properties"),
}


def _e(group: int, canonical: str, source: str, surface_forms: list[str]) -> dict:
    sub, group_name = GROUP_INFO[group]
    return {
        "group": group,
        "sub_standard": sub,
        "group_name": group_name,
        "canonical": canonical,
        "source": source,
        "surface_forms": surface_forms,
    }


PROCESSES: list[dict] = [
    # ── Group 1: Primary forming (DIN 8581) ─────────────────────────────────
    _e(1, "Casting from liquid state", "Groover §11 (Metal Casting Processes)",
       ["CAST", "CASTING", "SAND CAST", "DIE CAST", "INVESTMENT CAST"]),
    _e(1, "Sintering / powder metallurgy", "Groover §16 (Powder Metallurgy)",
       ["SINTERED", "SINTERING"]),
    _e(1, "Additive manufacturing", "Groover §33 (Rapid Prototyping) — name confirmed",
       ["3D PRINTED", "ADDITIVE", "SLM", "SLS", "FDM", "DMLS"]),

    # ── Group 2: Forming / DIN 8582 (Klocke V4) ─────────────────────────────
    _e(2, "Rolling", "Klocke V4 + Groover §19.1",
       ["ROLLED", "ROLLING", "HOT ROLLED", "COLD ROLLED"]),
    _e(2, "Forging (open-die / closed-die)", "Klocke V4 + Groover §19.3",
       ["FORGED", "FORGING"]),
    _e(2, "Extrusion", "Klocke V4 + Groover §19.5",
       ["EXTRUDED", "EXTRUSION"]),
    _e(2, "Drawing (wire / deep drawing)", "Klocke V4 + Groover §19.6/§20.3",
       # NOTE: bare "DRAWING" is intentionally excluded — it collides with
       # title-block boilerplate ("TECHNICAL DRAWING", "DRAWING NUMBER") and
       # produced 151/166 false positives in the 2026-05-23 A/B comparison.
       # Only multi-word forms and the past-participle "DRAWN" are kept.
       ["DRAWN", "DEEP DRAWN", "COLD DRAWN", "WIRE DRAWING", "DEEP DRAWING"]),
    _e(2, "Bending", "Klocke V4 + Groover §20.2",
       ["BENT", "BENDING", "PRESS BRAKE", "FORMED"]),
    _e(2, "Pressing / stamping", "Klocke V4 + Groover §20.5",
       ["PRESSED", "PRESSING", "STAMPED", "STAMPING"]),

    # ── Group 3: Separating / DIN 8583 (Klocke V1 — cutting w/ defined edge) ─
    # Klocke V1 Ch 9 = rotational primary movement (turning, milling, drilling, sawing)
    # Klocke V1 Ch 10 = translatory primary movement (broaching, shaving, planing)
    _e(3, "Turning",   "Klocke V1 §9.1 + Groover §22",
       ["TURNED", "TURNING", "LATHE"]),
    _e(3, "Drilling, boring, reaming, tapping (hole-making family)",
       "Klocke V1 §9.3 + Groover §22 (hole-making operations)",
       ["DRILLED", "DRILLING", "BORED", "BORING", "REAMED", "REAMING",
        "COUNTERSUNK", "COUNTERSINKING", "COUNTERBORED", "COUNTERBORING",
        "TAPPED", "TAPPING", "THREADED", "THREADING"]),
    _e(3, "Milling",   "Klocke V1 §9.2 + Groover §22",
       ["MILLED", "MILLING", "CNC MILLED"]),
    _e(3, "Planing / shaping", "Klocke V1 §10.3",
       ["PLANED", "PLANING", "SHAPED"]),
    _e(3, "Broaching", "Klocke V1 §10.1",
       ["BROACHED", "BROACHING"]),
    _e(3, "Sawing",    "Klocke V1 §9.4",
       ["SAWED", "SAWING", "SAW CUT", "BAND SAW"]),
    _e(3, "Filing / deburring / chamfering", "Groover §22 (related finishing ops)",
       ["FILED", "FILING", "DEBURRED", "DEBURRING", "CHAMFERED", "CHAMFERING"]),
    # Group 3 — abrasive / undefined-edge machining (Klocke V2 not held, but
    # processes confirmed by Groover §25 "Grinding and Other Abrasive Processes")
    _e(3, "Grinding",  "Groover §25 (Grinding) — Klocke V2 not held",
       ["GROUND", "GRINDING", "SURFACE GROUND"]),
    _e(3, "Honing",    "Groover §25.5 — Klocke V2 not held",
       ["HONED", "HONING"]),
    _e(3, "Lapping",   "Groover §25.5 — Klocke V2 not held",
       ["LAPPED", "LAPPING"]),
    _e(3, "Polishing", "Groover §27 (Surface processing) — process belongs to grp 3",
       ["POLISHED", "POLISHING"]),
    _e(3, "Blasting (shot / sand / bead / grit)",
       "Groover §27.1 (surface cleaning/treatments)",
       ["SANDBLASTED", "SANDBLASTING", "SHOT BLASTED", "SHOT BLASTING",
        "BEAD BLASTED", "GRIT BLASTED", "BLASTED", "BLASTING"]),
    # Group 3 — thermal / non-traditional separating (Klocke V3 not held)
    _e(3, "Flame / oxy-fuel cutting", "Groover §26 (Non-traditional machining) — V3 not held",
       ["OXY-FUEL", "FLAME CUT", "FLAME CUTTING", "GAS CUT"]),
    _e(3, "Plasma arc cutting", "Groover §26 — V3 not held",
       ["PLASMA CUT", "PLASMA CUTTING"]),
    _e(3, "Laser beam cutting", "Groover §26 — V3 not held",
       ["LASER CUT", "LASER CUTTING"]),
    _e(3, "Water-jet / abrasive water-jet cutting", "Groover §26 — V3 not held",
       ["WATER JET", "WATERJET", "ABRASIVE WATER JET"]),
    _e(3, "Electrical discharge machining (EDM)", "Groover §26.3 — V3 not held",
       ["EDM", "WIRE EDM", "SPARK ERODED"]),
    _e(3, "Pickling / chemical etching", "Groover §26.5",
       ["PICKLED", "PICKLING", "ETCHED", "ETCHING"]),
    # Group 3 — generic / family-level operations (kept separate from specific
    # operations above so the model can fall back to the family when the
    # specific operation isn't named).
    _e(3, "Machining (generic — defined-edge cutting family)",
       "Klocke V1 (full volume) + Groover §21 (Theory of Metal Machining) + §22",
       ["MACHINED", "MACHINING", "CNC MACHINED", "CNC MACHINING"]),
    _e(3, "Cutting (generic — separating, type unspecified)",
       "Klocke V1 (defined edge) + DIN 8583 umbrella. Use only when the "
       "specific cutting method (laser/plasma/saw/etc.) is not named.",
       ["CUT", "CUTTING"]),
    _e(3, "Cleaning (industrial surface cleaning, pre-coating prep)",
       "Groover §28 (Industrial Cleaning and Coating Processes — consolidated "
       "in 4e per the preface). DIN 8583 surface-preparation territory.",
       ["CLEANED", "CLEANING"]),
    _e(3, "Degreasing", "Groover §28 (chemical cleaning, alkaline/solvent degreasing)",
       ["DEGREASED", "DEGREASING"]),

    # ── Group 4: Joining / DIN 8584 ─────────────────────────────────────────
    # Welding subprocess detail lives in iso4063.py (ISO 4063 numbering).
    _e(4, "Welding (generic — see ISO 4063 for subprocess)",
       "Groover §30 (Welding Processes) + ISO 4063",
       ["WELDED", "WELDING"]),
    _e(4, "Soldering", "Groover §31.1",
       ["SOLDERED", "SOLDERING"]),
    _e(4, "Brazing",   "Groover §31.2",
       ["BRAZED", "BRAZING"]),
    _e(4, "Mechanical fastening (bolted / screwed / riveted / pinned)",
       "Groover §32 (Mechanical Assembly)",
       ["BOLTED", "BOLTING", "SCREWED", "RIVETED", "RIVETING", "PINNED"]),
    _e(4, "Adhesive bonding", "Groover §31.3",
       ["BONDED", "BONDING", "GLUED", "ADHESIVE"]),

    # ── Group 5: Coating / DIN 8585 ─────────────────────────────────────────
    _e(5, "Painting", "Groover §28 (Coating and Deposition Processes)",
       ["PAINTED", "PAINTING"]),
    _e(5, "Powder coating", "Groover §28 (organic coatings)",
       ["POWDER COATED", "POWDER COATING"]),
    _e(5, "Hot-dip galvanizing", "Groover §28 (hot dipping) + ISO 1461 ref",
       ["GALVANIZED", "GALVANISED", "GALVANIZING", "HDG", "HOT DIP GALVANIZED"]),
    _e(5, "Electroplating", "Groover §28.2 (electroplating)",
       ["PLATED", "PLATING", "ELECTROPLATED", "ZINC PLATED", "NICKEL PLATED",
        "CHROME PLATED"]),
    _e(5, "Anodizing", "Groover §28 (conversion coatings)",
       ["ANODIZED", "ANODISED", "ANODIZING"]),
    _e(5, "Thermal spray coating", "Groover §28 (thermal spraying)",
       ["THERMAL SPRAYED", "FLAME SPRAYED", "HVOF"]),
    _e(5, "Phosphating", "Groover §28 (conversion coatings)",
       ["PHOSPHATED", "PHOSPHATING"]),

    # ── Group 6: Changing material properties / DIN 8587 ────────────────────
    _e(6, "Hardening / quenching", "Groover §27 (Heat Treatment of Metals)",
       ["HARDENED", "HARDENING", "QUENCHED", "QUENCHING"]),
    _e(6, "Tempering", "Groover §27",
       ["TEMPERED", "TEMPERING"]),
    _e(6, "Annealing / stress relieving", "Groover §27",
       ["ANNEALED", "ANNEALING", "STRESS RELIEVED", "STRESS RELIEVING"]),
    _e(6, "Normalising", "Groover §27",
       ["NORMALIZED", "NORMALISED", "NORMALIZING"]),
    _e(6, "Case hardening (carburising / nitriding)",
       "Groover §27 (surface hardening)",
       ["CASE HARDENED", "CARBURISED", "CARBURIZED", "NITRIDED", "NITRIDING"]),
    _e(6, "Induction hardening", "Groover §27 (selective surface hardening)",
       ["INDUCTION HARDENED", "INDUCTION HARDENING"]),
]


# Flat lookup: uppercase surface form → entry dict. Longest forms first so
# multi-word phrases match before single-word substrings in a token sweep.
_LOOKUP: dict[str, dict] = {}
for _entry in sorted(PROCESSES, key=lambda e: -max(len(s) for s in e["surface_forms"])):
    for _sf in _entry["surface_forms"]:
        _LOOKUP.setdefault(_sf, _entry)


def lookup(token: str) -> dict | None:
    """Return DIN 858x entry for an uppercase surface-form token, or None."""
    return _LOOKUP.get(token.upper())


SURFACE_FORMS: frozenset[str] = frozenset(_LOOKUP.keys())
