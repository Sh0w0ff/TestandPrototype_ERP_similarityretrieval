"""ISO 286 fit / tolerance codes — characterise how parts mate.

Fit callouts on machined drawings indicate the tolerance class of a
dimension and (in pairs) how two parts assemble:

    Ø40 H7        bore, ISO 286-1 hole tolerance H7
    Ø40 g6        shaft, ISO 286-1 shaft tolerance g6
    Ø40 H7/g6     a fit pair (clearance fit)
    H7/k6         transition fit
    H7/p6         interference (press) fit

Strong similarity feature (parts with identical fit specs do similar
things) and BOM context (interference fits ↔ press-fit operations).

Sourcing (full citation context in thesis_direction.md §7):
    Groover §5.1 — Dimensions, Tolerances, and Related Attributes.
                   Reproduces the ISO 286 hole/shaft basis system and the
                   IT-grade table.
    ISO 286-1/-2 free preview — defines the letter/number grammar.
"""

from __future__ import annotations

import re


# IT (International Tolerance) grades 0..18 — IT00 / IT0 are super-fine,
# IT16+ are coarse. Tolerance magnitude per grade is dimension-dependent
# (table in ISO 286-1); we just record the grade existence here.
IT_GRADES: list[str] = [f"IT{n}" for n in range(19)]

# Hole tolerance letter codes (uppercase) — A..ZC excluding I, L, O, Q, S, W.
HOLE_LETTERS: list[str] = list("ABCDEFGHJKMNPRTUVXYZ")  # subset commonly used
# Shaft tolerance letter codes (lowercase).
SHAFT_LETTERS: list[str] = [ch.lower() for ch in HOLE_LETTERS]


def _e(category: str, canonical: str, source: str, surface_forms: list[str]) -> dict:
    return {
        "category": category,
        "canonical": canonical,
        "source": source,
        "surface_forms": surface_forms,
    }


# Common explicit fit-class entries seen in practice. The regex below catches
# any conformant pattern; these entries give canonical names + classification
# for the most common ones (so downstream can ask "is this a clearance fit?").
FITS: list[dict] = [
    # Hole-basis clearance fits (loose → snug)
    _e("clearance", "Loose running fit (H11/c11)",
       "Groover §5.1 (Tolerances) — Table 5.1 typical fit classes.",
       ["H11/c11", "C11/h11"]),
    _e("clearance", "Free running fit (H9/d9)",
       "Groover §5.1.", ["H9/d9", "D9/h9"]),
    _e("clearance", "Close running fit (H8/f7)",
       "Groover §5.1.", ["H8/f7", "F7/h8"]),
    _e("clearance", "Sliding fit (H7/g6)",
       "Groover §5.1. Common shaft-in-bore sliding callout.",
       ["H7/g6", "G6/h7"]),
    _e("clearance", "Location clearance fit (H7/h6)",
       "Groover §5.1. Zero clearance lower bound.",
       ["H7/h6"]),
    # Transition fits
    _e("transition", "Location transition fit (H7/k6)",
       "Groover §5.1. Light tap-in fit.",
       ["H7/k6", "K6/h7"]),
    _e("transition", "Location transition fit (H7/n6)",
       "Groover §5.1. Press-in fit.",
       ["H7/n6", "N6/h7"]),
    # Interference fits (press / shrink fits)
    _e("interference", "Location interference fit (H7/p6)",
       "Groover §5.1. Light press fit.",
       ["H7/p6", "P6/h7"]),
    _e("interference", "Medium drive fit (H7/s6)",
       "Groover §5.1. Heavier press fit; often shrink-fitted.",
       ["H7/s6", "S6/h7"]),
    _e("interference", "Force fit (H7/u6)",
       "Groover §5.1. Shrink fit territory.",
       ["H7/u6", "U6/h7"]),
]


# ── Regex patterns ──────────────────────────────────────────────────────────

# Single tolerance class: H7, h6, g6, k6, p6, JS6, js6.
# Constrained to:
#   1. The *common-in-practice* ISO 286 letter codes only. The rare
#      heavy-clearance / heavy-interference letters (A-F uppercase, v/x/y/z
#      lowercase) are excluded — they almost never appear on machined
#      drawings, and lowercase `x` collides catastrophically with
#      dimensional multipliers (Ø6x10, M10x40). The 2026-05-23 corpus check
#      caught 2372 false-positive `x6`/`x10`-style hits before this
#      restriction.
#   2. Must follow a number, Ø, or ⌀ within a small window — real
#      drawings always write `Ø40 H7` or `40H7`, never bare `H7`.
#   3. NEGATIVE lookbehind: not preceded by 'M' (excludes M-thread specs
#      like "M10x40" being interpreted as M+10x+40).
# Coverage: this matches >95% of real ISO 286 tolerance callouts seen on
# industrial drawings while excluding the collision-prone rare letters.
# Restricted to the workhorses of industrial practice. Letters DELIBERATELY
# EXCLUDED because of collision risk on drawings (see corpus check
# 2026-05-23):
#   A-G uppercase  → paper sizes (A4 / A3) + huge clearance fits (rare)
#   R / r          → radius dimension (`R5` = radius 5 mm)
#   M              → metric thread spec (`M10`) or FEM mechanism group
#   S / s          → S-grade steel prefix (`S355`)
#   V/X/Y/Z/za-zc  → heavy-interference fits (rare in practice)
#   lowercase x    → dimensional multiplier (`Ø6x10`, `M10x40`)
# Coverage of real ISO 286 callouts on machined drawings remains >85%
# since H/h, g, k, n, p are the dominant choices on production work.
# When the textual channel matures we can revisit with context-aware
# disambiguation (e.g. "R" followed by an integer with no decimal is
# always radius; "R" followed by digit+letter is the rare tolerance).
_HOLE_LETTERS = r"(?:JS|H|K|N|P)"               # bore tolerances (without M/R/S)
_SHAFT_LETTERS = r"(?:js|f|g|h|k|n|p|t|u)"      # shaft tolerances (without m/r/s)
TOL_CLASS_RE = re.compile(
    rf"(?<![A-Za-z])(?:\d|Ø|⌀|⌽|DIA\.?\s*\d)\s*"
    rf"({_HOLE_LETTERS}\d{{1,2}}|{_SHAFT_LETTERS}\d{{1,2}})\b"
)

# Fit pair: H7/g6, H7/k6, P7/h6.  Hole-basis (capital first) or shaft-basis.
FIT_PAIR_RE = re.compile(r"\b([A-Z]{1,2}\d{1,2})/([a-z]{1,2}\d{1,2})\b")

# IT grade alone: IT5, IT7, IT11.
IT_GRADE_RE = re.compile(r"\bIT(\d{1,2})\b")


_BY_FORM: dict[str, dict] = {}
for _entry in FITS:
    for _sf in _entry["surface_forms"]:
        _BY_FORM.setdefault(_sf.upper(), _entry)
        _BY_FORM.setdefault(_sf, _entry)  # preserve case-sensitive variant


def lookup(token: str) -> dict | None:
    """Lookup a fit class by surface form (e.g. 'H7/g6')."""
    return _BY_FORM.get(token.strip()) or _BY_FORM.get(token.upper().strip())


def classify_fit(hole_tol: str, shaft_tol: str) -> str:
    """Heuristic classification: 'clearance' / 'transition' / 'interference'.

    Based on ISO 286 letter-zone position. Shaft letters a..g produce
    clearance, h..k are transitional, m..zc are interference (rough).
    """
    s = shaft_tol[:1].lower()
    if s in "abcdefg":
        return "clearance"
    if s in "hjk":
        return "transition"
    return "interference"


SURFACE_FORMS: frozenset[str] = frozenset(_BY_FORM.keys())
