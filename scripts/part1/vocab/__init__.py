"""Standards-anchored vocabulary library for technical-drawing extraction.

Every submodule encodes a published international standard (ISO / DIN /
EN / IEC / FEM / ASME / AISI-SAE). None of the entries are tailored to
our ABB / Konecranes corpus — these are general libraries that any
mechanical-engineering drawing analyser could reuse.

Modules:
    din8580             — DIN 8580 / DIN 858x manufacturing process taxonomy
    iso4063             — ISO 4063 welding-process numbers
    assembly            — Mechanical-assembly action verbs (Groover §32)
    fasteners           — ISO/DIN bolts/nuts/washers + ISO 898 strength grades
    materials           — EN 10027 / EN 10088 / EN 573 / ISO 1043 / AISI-SAE
    bearings            — ISO 15 / ISO 355 / ISO 104 bearing designations
    fits                — ISO 286 fit / tolerance codes
    gdt                 — ISO 1101 / ASME Y14.5 GD&T characteristic labels
    surface_texture     — ISO 1302 / ISO 21920 Ra/Rz/N-grades
    weld_quality        — ISO 5817 / ISO 13920 / EN 1090 weld quality + EXC
    crane_standards     — FEM 1.001 / EN 13001 / DIN 15xxx (KC-relevant)
    electrical_standards — IEC 60034 / 60072 / 60204 / 60529 (ABB-relevant)
    corrosion           — ISO 12944 / ISO 1461 / EN 10346 coatings/protection
    legacy_corpus_discovered — verbatim snapshot of the prior corpus-discovered
                               vocab, retained for A/B coverage comparison.

Each submodule exposes:
    ENTRIES list[dict] — canonical entries with standard, canonical name,
                         source citation, and surface_forms
    lookup(token)      — uppercase surface-form → entry dict (or None)
    SURFACE_FORMS      — frozenset of all uppercase surface forms

Several modules also expose compiled regex patterns for structured codes
(e.g. M-thread, 1.xxxx material numbers, 6xxx bearings, IP ratings, fit
pairs like H7/g6). See each module's docstring.
"""

from .din8580 import PROCESSES as DIN8580_PROCESSES, lookup as din8580_lookup
from .iso4063 import WELDING as ISO4063_WELDING, lookup as iso4063_lookup
from .assembly import ASSEMBLY_ACTIONS, lookup as assembly_lookup
from .fasteners import (
    FASTENERS, STRENGTH_GRADES,
    lookup as fastener_lookup,
    lookup_strength_grade,
)
from .materials import MATERIALS, lookup as material_lookup
from .bearings import BEARING_FAMILIES, lookup as bearing_lookup, parse_bore_code
from .fits import FITS, lookup as fit_lookup, classify_fit
from .gdt import GDT, lookup as gdt_lookup
from .surface_texture import SURFACE_TEXTURE, lookup as surface_texture_lookup, n_to_ra
from .weld_quality import WELD_QUALITY, lookup as weld_quality_lookup
from .crane_standards import CRANE, lookup as crane_lookup
from .electrical_standards import ELECTRICAL, lookup as electrical_lookup
from .corrosion import CORROSION, lookup as corrosion_lookup
from .production_lines import PRODUCTION_LINES, lookup as production_line_lookup
from .legacy_corpus_discovered import (
    PROCESS_KEYWORDS as LEGACY_PROCESS_KEYWORDS,
    PROC_CANDIDATE_STOPLIST as LEGACY_PROC_STOPLIST,
    legacy_lookup,
)

# Convenience: every "standards-anchored" module in one tuple — useful when
# you want to sweep a token against all libraries simultaneously without
# binding to specific names.
ALL_STANDARDS_LIBS = (
    ("din8580",             DIN8580_PROCESSES,    din8580_lookup),
    ("iso4063",             ISO4063_WELDING,      iso4063_lookup),
    ("assembly",            ASSEMBLY_ACTIONS,     assembly_lookup),
    ("fasteners",           FASTENERS,            fastener_lookup),
    ("materials",           MATERIALS,            material_lookup),
    ("bearings",            BEARING_FAMILIES,     bearing_lookup),
    ("fits",                FITS,                 fit_lookup),
    ("gdt",                 GDT,                  gdt_lookup),
    ("surface_texture",     SURFACE_TEXTURE,      surface_texture_lookup),
    ("weld_quality",        WELD_QUALITY,         weld_quality_lookup),
    ("crane_standards",     CRANE,                crane_lookup),
    ("electrical_standards", ELECTRICAL,          electrical_lookup),
    ("corrosion",           CORROSION,            corrosion_lookup),
    ("production_lines",    PRODUCTION_LINES,     production_line_lookup),
)


__all__ = [
    # Data
    "DIN8580_PROCESSES", "ISO4063_WELDING", "ASSEMBLY_ACTIONS",
    "FASTENERS", "STRENGTH_GRADES",
    "MATERIALS", "BEARING_FAMILIES", "FITS", "GDT", "SURFACE_TEXTURE",
    "WELD_QUALITY", "CRANE", "ELECTRICAL", "CORROSION",
    "PRODUCTION_LINES",
    # Lookups
    "din8580_lookup", "iso4063_lookup", "assembly_lookup",
    "fastener_lookup", "lookup_strength_grade",
    "material_lookup", "bearing_lookup", "parse_bore_code",
    "fit_lookup", "classify_fit",
    "gdt_lookup", "surface_texture_lookup", "n_to_ra",
    "weld_quality_lookup", "crane_lookup", "electrical_lookup",
    "corrosion_lookup", "production_line_lookup",
    # Legacy
    "LEGACY_PROCESS_KEYWORDS", "LEGACY_PROC_STOPLIST", "legacy_lookup",
    # Aggregate
    "ALL_STANDARDS_LIBS",
]
