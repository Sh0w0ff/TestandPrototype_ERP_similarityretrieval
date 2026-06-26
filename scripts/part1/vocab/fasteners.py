"""Fastener vocabulary — direct BOM-child detection.

Bolts/screws/nuts/washers are the most common BOM child components on
assembly drawings. A typical callout reads:

    M10x40 ISO 4014 8.8 - A2C    (metric hex bolt, partial thread, grade 8.8)
    DIN 933 M8x20                 (older designation, same thing)
    M6x16 ISO 4762 12.9           (socket head cap screw)
    G1/4 BSP                      (pipe thread)

This module captures the families (ISO 4014 etc.) as structured entries
with canonical name and source, plus three regex helpers for the most
common structural patterns: thread spec (M/G sizes), strength-grade
notation, and full bolt callouts.

Sourcing (full citation context in thesis_direction.md §7):
    Groover §32   — Mechanical Assembly (Wiley 4e). Covers threaded
                    fasteners, nuts, washers, strength grades.
    ISO/DIN free  — Standard scope pages (iso.org/obp + din.de) enumerate
    previews        the fastener-standard numbering ranges.
    Wikipedia     — "List of ISO standards" and "ISO metric screw thread"
                    cross-checks for fastener-standard mapping.
"""

from __future__ import annotations

import re


def _e(standard: str, canonical: str, source: str, surface_forms: list[str],
       family: str = "fastener") -> dict:
    return {
        "standard": standard,
        "canonical": canonical,
        "family": family,
        "source": source,
        "surface_forms": surface_forms,
    }


# Each entry's surface_forms list the literal *standard reference* strings
# you'd see in a drawing callout. They're matched alongside the regex
# patterns below; the entry is selected by whichever standard number the
# regex captures.
FASTENERS: list[dict] = [
    # ── Hex bolts / screws (ISO + DIN equivalents) ──────────────────────────
    _e("ISO 4014", "Hex head bolt (partial thread)",
       "Groover §32.1 — hex-head bolt family. ISO 4014 is the metric "
       "partial-thread variant; DIN 931 is the older equivalent (still in use).",
       ["ISO 4014", "DIN 931"], family="hex_bolt"),
    _e("ISO 4017", "Hex head bolt (full thread)",
       "Groover §32.1. ISO 4017 is the full-thread variant; DIN 933 is the "
       "older designation seen most often on European drawings.",
       ["ISO 4017", "DIN 933"], family="hex_bolt"),
    _e("ISO 8765", "Hex head bolt with fine pitch",
       "Fine-pitch variant of ISO 4014.",
       ["ISO 8765"], family="hex_bolt"),

    # ── Socket head cap screws ──────────────────────────────────────────────
    _e("ISO 4762", "Socket head cap screw (hex socket)",
       "Groover §32.1 — internal-drive cap screw. ISO 4762 / DIN 912 are "
       "equivalent; DIN 912 is the older designation.",
       ["ISO 4762", "DIN 912"], family="socket_head_cap_screw"),
    _e("ISO 10642", "Socket countersunk head screw",
       "Countersunk equivalent. DIN 7991 is the older designation.",
       ["ISO 10642", "DIN 7991"], family="csk_socket_screw"),
    _e("ISO 7380", "Button head socket screw",
       "Low-profile internal-drive screw.",
       ["ISO 7380"], family="button_head_socket_screw"),

    # ── Slotted / cross-recessed machine screws ─────────────────────────────
    _e("ISO 1207", "Slotted cheese head screw",
       "DIN 84 is the older designation.",
       ["ISO 1207", "DIN 84"], family="machine_screw"),
    _e("ISO 7045", "Cross-recessed (Phillips/Pozi) pan head screw",
       "DIN 7985 is the older designation.",
       ["ISO 7045", "DIN 7985"], family="machine_screw"),
    _e("ISO 7046", "Cross-recessed countersunk head screw",
       "DIN 965 is the older designation.",
       ["ISO 7046", "DIN 965"], family="csk_machine_screw"),

    # ── Nuts ────────────────────────────────────────────────────────────────
    _e("ISO 4032", "Hex nut (style 1)",
       "Groover §32.1. DIN 934 is the older designation.",
       ["ISO 4032", "DIN 934"], family="hex_nut"),
    _e("ISO 4035", "Thin hex nut",
       "Low-profile hex nut. DIN 439 older designation.",
       ["ISO 4035", "DIN 439"], family="hex_nut"),
    _e("ISO 7040", "Prevailing-torque hex nut (nylon insert)",
       "Self-locking nut family. DIN 985 older designation.",
       ["ISO 7040", "DIN 985"], family="locknut"),
    _e("ISO 7042", "All-metal prevailing-torque hex nut",
       "All-metal self-locking nut. DIN 980 older designation.",
       ["ISO 7042", "DIN 980"], family="locknut"),

    # ── Washers ─────────────────────────────────────────────────────────────
    _e("ISO 7089", "Plain washer, normal",
       "Groover §32.1. DIN 125 is the older designation.",
       ["ISO 7089", "DIN 125"], family="washer"),
    _e("ISO 7090", "Plain washer, chamfered",
       "Chamfered variant of ISO 7089. DIN 125 covers both.",
       ["ISO 7090"], family="washer"),
    _e("ISO 7091", "Plain washer, normal — product grade C",
       "Coarse-grade plain washer.",
       ["ISO 7091"], family="washer"),
    _e("ISO 7093", "Plain washer, large diameter",
       "Larger OD than ISO 7089. DIN 9021 older designation.",
       ["ISO 7093", "DIN 9021"], family="washer"),
    _e("DIN 127", "Split spring lock washer",
       "Spring lock washer (single coil). No direct ISO equivalent in "
       "common use; DIN 127 still the standard reference.",
       ["DIN 127"], family="washer"),

    # ── Threaded rod / studs ────────────────────────────────────────────────
    _e("DIN 975", "Threaded rod",
       "Long threaded rod stock. ISO 898-1 covers material/strength.",
       ["DIN 975"], family="threaded_rod"),
    _e("DIN 976", "Stud bolt (threaded rod, cut length)",
       "Stud bolt for through-bolting.",
       ["DIN 976"], family="stud"),

    # ── Pins / parallel keys (BOM-relevant joining hardware) ────────────────
    _e("ISO 2338", "Parallel pin (dowel pin)",
       "Cylindrical dowel pin. DIN 7 older designation.",
       ["ISO 2338", "DIN 7"], family="pin"),
    _e("ISO 8734", "Parallel pin, hardened (precision dowel)",
       "Hardened-steel dowel pin. DIN 6325 older designation.",
       ["ISO 8734", "DIN 6325"], family="pin"),
    _e("ISO 2339", "Taper pin",
       "Tapered dowel pin. DIN 1 older designation.",
       ["ISO 2339", "DIN 1"], family="pin"),
    _e("ISO 8752", "Spring-type straight pin (slotted)",
       "Spring (split) pin. DIN 1481 older designation.",
       ["ISO 8752", "DIN 1481"], family="pin"),
    _e("DIN 6885", "Parallel key (machine key for shafts)",
       "Standard parallel key for shaft-hub connections.",
       ["DIN 6885"], family="key"),

    # ── Retaining rings (circlips) ──────────────────────────────────────────
    _e("DIN 471", "External retaining ring (shaft circlip)",
       "Groover §32.5 (Retaining rings).",
       ["DIN 471"], family="retaining_ring"),
    _e("DIN 472", "Internal retaining ring (bore circlip)",
       "Groover §32.5.",
       ["DIN 472"], family="retaining_ring"),

    # ── Rivets ──────────────────────────────────────────────────────────────
    _e("ISO 15983", "Blind rivet, open end",
       "Groover §32.2. DIN 7337 older designation.",
       ["ISO 15983", "DIN 7337"], family="rivet"),
    _e("DIN 124", "Solid rivet, round head",
       "Solid (structural) rivet.",
       ["DIN 124"], family="rivet"),
]


# ── Strength grades (ISO 898-1 / ISO 3506) ──────────────────────────────────
# These are not standalone fasteners — they're modifiers (e.g. "M10 ISO 4014 8.8").
# Kept as a flat list because they're often the only structural info captured.
STRENGTH_GRADES: list[dict] = [
    {"code": "4.6",  "standard": "ISO 898-1", "material": "low/medium-carbon steel"},
    {"code": "4.8",  "standard": "ISO 898-1", "material": "low/medium-carbon steel"},
    {"code": "5.6",  "standard": "ISO 898-1", "material": "low/medium-carbon steel"},
    {"code": "5.8",  "standard": "ISO 898-1", "material": "low/medium-carbon steel"},
    {"code": "6.8",  "standard": "ISO 898-1", "material": "low/medium-carbon steel"},
    {"code": "8.8",  "standard": "ISO 898-1", "material": "medium-carbon steel, Q&T"},
    {"code": "9.8",  "standard": "ISO 898-1", "material": "medium-carbon steel, Q&T"},
    {"code": "10.9", "standard": "ISO 898-1", "material": "alloy steel, Q&T"},
    {"code": "12.9", "standard": "ISO 898-1", "material": "alloy steel, Q&T"},
    # ISO 3506 stainless grades
    {"code": "A2-50", "standard": "ISO 3506-1", "material": "stainless A2 (304-equivalent)"},
    {"code": "A2-70", "standard": "ISO 3506-1", "material": "stainless A2, cold-worked"},
    {"code": "A2-80", "standard": "ISO 3506-1", "material": "stainless A2, high-strength"},
    {"code": "A4-50", "standard": "ISO 3506-1", "material": "stainless A4 (316-equivalent)"},
    {"code": "A4-70", "standard": "ISO 3506-1", "material": "stainless A4, cold-worked"},
    {"code": "A4-80", "standard": "ISO 3506-1", "material": "stainless A4, high-strength"},
]
_GRADE_LOOKUP = {g["code"]: g for g in STRENGTH_GRADES}


# ── Regex patterns ──────────────────────────────────────────────────────────

# Metric thread spec: M3 … M64, optional pitch (M10x1.5), optional length (M10x40).
# Catches: "M10", "M10x40", "M10x1.5", "M10x1.5x40".
METRIC_THREAD_RE = re.compile(
    r"\bM(\d{1,3}(?:\.\d)?)(?:x(\d+(?:\.\d+)?))?(?:x(\d+(?:\.\d+)?))?\b"
)

# Pipe thread: G1/8, G1/4, G3/8, G1/2, G3/4, G1, G1-1/4, G1-1/2, G2, G3, G4, G6.
PIPE_THREAD_RE = re.compile(r"\bG\d(?:-?\d/\d)?(?:/\d)?\b")

# Strength grade: enumerated ISO 898 / 3506 grades only, AND require a
# fastener-context anchor in a 30-char window before/after. The earlier
# liberal pattern `\d{1,2}\.\d` matched 1339 random decimals on the
# 166-PDF corpus (2026-05-23 regex sweep). Combined enumeration + context
# anchor reduces this to actual grade callouts.
_GRADE_VALUES = r"(?:4\.[68]|5\.[68]|6\.8|8\.8|9\.8|10\.9|12\.9)"
_FASTENER_CTX = (
    r"(?:M\d{1,3}(?:x\d+(?:\.\d+)?)*\s*|GRADE\s+|ISO\s*898[-\s]?\d?\s*|"
    r"DIN\s*(?:931|933|912|934)\s*|BOLT\s+|SCREW\s+|NUT\s+|STUD\s+)"
)
STRENGTH_GRADE_RE = re.compile(
    rf"(?:{_FASTENER_CTX}{_GRADE_VALUES}|{_GRADE_VALUES}\s+"
    rf"(?:GRADE|BOLT|SCREW|NUT|STUD|ISO\s*898))",
    re.IGNORECASE,
)
STAINLESS_GRADE_RE = re.compile(r"\b(A[24])-(50|70|80)\b")

# Standard reference: "ISO <num>" or "DIN <num>".
STANDARD_REF_RE = re.compile(r"\b(ISO|DIN)\s*(\d{1,5})\b")


# ── Lookups ────────────────────────────────────────────────────────────────

_BY_STANDARD: dict[str, dict] = {}
for e in FASTENERS:
    for sf in e["surface_forms"]:
        _BY_STANDARD.setdefault(sf.upper().replace(" ", ""), e)
        _BY_STANDARD.setdefault(sf.upper(), e)


def lookup(token: str) -> dict | None:
    """Lookup by standard reference ('ISO 4014', 'DIN 933', etc.)."""
    t = token.upper().strip()
    return _BY_STANDARD.get(t) or _BY_STANDARD.get(t.replace(" ", ""))


def lookup_strength_grade(code: str) -> dict | None:
    """Lookup an ISO 898-1 or ISO 3506-1 strength-grade code."""
    return _GRADE_LOOKUP.get(code.upper())


SURFACE_FORMS: frozenset[str] = frozenset(
    sf for e in FASTENERS for sf in e["surface_forms"]
)
