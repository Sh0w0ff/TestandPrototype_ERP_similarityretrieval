"""Legacy corpus-discovered process vocabulary — snapshot for A/B comparison.

Verbatim copy of the PROCESS_KEYWORDS dict and the PROC_CANDIDATE_STOPLIST
that lived in extract_drawing.py through the 166-PDF baseline (see
session_resume_2026_05_22_evening). Preserved here so we can:

    1. Re-run the schema extractor with old vs new vocab and diff coverage.
    2. Trace any regression in a specific field back to a vocabulary change.
    3. Cite the original heuristic in the thesis if needed.

DO NOT extend this file. Add new processes to din8580.py / iso4063.py instead.
"""

from __future__ import annotations

# Original curated seed list. Categories were the author's labels; they do not
# map cleanly to DIN 8580 groups (e.g. "cutting" mixes 3.4 thermal cutting with
# 3.2.x machining; "removal" mixes 3.2 (defined edge) with 3.3 (undefined)).
PROCESS_KEYWORDS: dict[str, list[str]] = {
    "joining":      ["WELDED", "WELDING", "BRAZED", "BOLTED", "RIVETED"],
    "cutting":      ["LASER CUT", "PLASMA CUT", "OXY-FUEL", "WATER JET", "SAW CUT",
                     "LASER", "CNC", "EDM", "CUTTING"],
    "removal":      ["MACHINED", "MACHINING", "MILLED", "DRILLED", "TURNED",
                     "BORED", "REAMED", "GROUND", "GRINDING", "TAPPED", "THREADED",
                     "COUNTERSUNK", "COUNTERBORED", "CHAMFERED", "DEBURRED"],
    "forming":      ["STAMPED", "BENT", "BENDING", "FORMED", "ROLLED", "PRESSED", "FORGED", "DRAWN"],
    "heat_treat":   ["HARDENED", "ANNEALED", "NORMALIZED", "TEMPERED", "QUENCHED"],
    "surface_prep": ["DEGREASING", "CLEANING", "SANDBLASTING", "SHOT BLASTING",
                     "PICKLING", "POLISHING"],
    "assembly":     ["INSTALLED", "FIXING", "ASSEMBLED", "MOUNTED", "MOUNTING"],
}

PROC_CANDIDATE_STOPLIST: set[str] = {
    "USED", "MADE", "ADDED", "REMOVED", "REVISED", "APPROVED", "MARKED", "UNMARKED",
    "CHANGED", "DESIGN", "DRAWING", "STRING", "MISSING", "FOLLOWING", "OPERATING",
    "ENGINEERING", "PACKAGING", "SHIPPING", "WARNING", "CHARGED", "PROVIDED",
    "CONTAINED", "REFERENCED", "DUPLICATED", "INCLUDED", "EXCLUDED",
    "PROTECTING", "REPLACING", "TRACKING", "LOADING", "PROCESSING",
    "BUILDING", "PUBLISHING", "PRINTING",
    "REPRODUCED", "DISCLOSED", "ALTERED", "EMPLOYED", "RESERVED", "BASED",
    "PREPARED", "ADDING", "ACCORDING", "MANUFACTURED",
    "UPDATED", "MODIFIED", "MOVED", "CHECKED", "CORRECTED",
    "DROPPING", "INSULATING", "PAINTING",
    "MANUFACTURING",
    "COATING",
    "TAPERED", "INCLINED", "BEARING", "UNFOLDED", "COATED", "INSULATING",
    "LIFTING", "CLOSING", "LOWERED", "ALLOWED", "DECREASING", "STARTING",
    "MEASURING",
    "REEVING",
}

_LEGACY_LOOKUP: dict[str, str] = {
    kw: cat for cat, kws in PROCESS_KEYWORDS.items() for kw in kws
}


def legacy_lookup(token: str) -> str | None:
    """Return the legacy category label for a surface form, or None."""
    return _LEGACY_LOOKUP.get(token.upper())
