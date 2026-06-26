"""ISO 1302 / ISO 21920 surface texture — Ra/Rz/Rmax values and N-grades.

Surface-finish callouts encode manufacturing intent. `Ra 3.2` typically
means machined; `Ra 1.6` ground; `Ra 0.8` finely ground or honed;
`Ra 0.4` polished. Two drawings with identical surface specs are likely
similar parts; the values also constrain which DIN 858x process must
appear in the workflow.

Sourcing (full citation context in thesis_direction.md §7):
    Groover §5.3 — Surfaces (Roughness, Ra/Rz, lay symbols, N-grades).
    Klocke V1 §3.5 (Kinematic Surface Roughness) + §2.4.1 (Surface
                  Parameters) — defines Ra, Rz, Rt with mathematical formulas.
    ISO 1302 / ISO 21920 free preview — symbol grammar (Ra under triangle).
    ISO 1302 Table 2 — N1..N12 roughness grades mapped to Ra values.
"""

from __future__ import annotations

import re


# N-grade ↔ Ra (µm) mapping per ISO 1302 (older but still cited).
N_GRADE_TO_RA: dict[str, float] = {
    "N1": 0.025, "N2": 0.05, "N3": 0.1, "N4": 0.2, "N5": 0.4,
    "N6": 0.8,   "N7": 1.6,  "N8": 3.2, "N9": 6.3, "N10": 12.5,
    "N11": 25.0, "N12": 50.0,
}


def _e(category: str, canonical: str, source: str, surface_forms: list[str]) -> dict:
    return {
        "category": category,
        "canonical": canonical,
        "source": source,
        "surface_forms": surface_forms,
    }


SURFACE_TEXTURE: list[dict] = [
    # Roughness parameters (the parameter names themselves).
    _e("parameter", "Arithmetical mean roughness (Ra)",
       "Klocke V1 §2.4.1 / §3.5; Groover §5.3.2.",
       ["Ra", "RA"]),
    _e("parameter", "Maximum height of profile (Rz)",
       "Klocke V1 §2.4.1.",
       ["Rz", "RZ"]),
    _e("parameter", "Maximum profile height (Rt)",
       "Klocke V1 §2.4.1.",
       ["Rt", "RT"]),
    _e("parameter", "Maximum peak height (Rp)",
       "Klocke V1 §2.4.1.",
       ["Rp", "RP"]),
    _e("parameter", "Mean spacing of profile elements (Rsm)",
       "ISO 21920-2.",
       ["Rsm", "RSM"]),

    # Lay symbols (text forms — actual symbols are visual).
    _e("lay", "Parallel lay (∥)",
       "ISO 1302; Groover §5.3.",
       ["LAY PARALLEL", "PARALLEL LAY"]),
    _e("lay", "Perpendicular lay (⊥)",
       "ISO 1302.",
       ["LAY PERPENDICULAR", "PERPENDICULAR LAY"]),
    _e("lay", "Crossed lay (X)",
       "ISO 1302.",
       ["LAY X", "CROSSED LAY"]),
    _e("lay", "Multi-directional lay (M)",
       "ISO 1302.",
       ["LAY M", "MULTI-DIRECTIONAL"]),
    _e("lay", "Circular lay (C)",
       "ISO 1302.",
       ["LAY C", "CIRCULAR LAY"]),
    _e("lay", "Radial lay (R)",
       "ISO 1302.",
       ["LAY R", "RADIAL LAY"]),
    _e("lay", "Pitted / particulate lay (P)",
       "ISO 1302.",
       ["LAY P", "PITTED LAY"]),

    # N-grades — encoded as surface forms; mapping to Ra in N_GRADE_TO_RA.
    _e("n_grade", "ISO 1302 roughness N-grade",
       "ISO 1302 Table 2; Groover §5.3 (older roughness-grade system).",
       list(N_GRADE_TO_RA.keys())),
]


# ── Regex patterns ──────────────────────────────────────────────────────────

# Ra value: "Ra 3.2", "Ra3.2", "Ra=3.2", "Ra 0.8" (µm assumed).
RA_VALUE_RE = re.compile(r"\bR[aAzZtTpP]\s*=?\s*(\d+(?:\.\d+)?)\b")

# N-grade alone (must be word-boundary to avoid catching e.g. N7-prefixed bearings).
N_GRADE_RE = re.compile(r"\b(N(?:1[0-2]|[1-9]))\b")


_BY_FORM: dict[str, dict] = {}
for _entry in sorted(
    SURFACE_TEXTURE, key=lambda e: -max(len(s) for s in e["surface_forms"])
):
    for _sf in _entry["surface_forms"]:
        _BY_FORM.setdefault(_sf.upper(), _entry)


def lookup(token: str) -> dict | None:
    return _BY_FORM.get(token.upper().strip())


def n_to_ra(n_grade: str) -> float | None:
    """Convert N-grade ('N7') to Ra value in µm, or None."""
    return N_GRADE_TO_RA.get(n_grade.upper())


SURFACE_FORMS: frozenset[str] = frozenset(_BY_FORM.keys())
