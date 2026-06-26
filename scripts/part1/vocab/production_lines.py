"""ERP-grounded production-line vocabulary (Stera ERP, bilingual EN/FI).

Sourced empirically from the Stera-provided ERP CSVs (35 production-line
categories in Work_Center_Basic_Data.csv, 310 distinct work-phase
descriptions in Work_Phases.csv — see Data set documentation.docx). This
is the customer's *own* taxonomy of how parts actually move through their
factory floor, expressed in the bilingual Finnish/English shop-floor
vocabulary used by Konecranes and ABB Drives operators.

Methodology rationale:
    DIN 858x is the published international taxonomy of *what kind of
    process* a manufacturing step is (Forming, Separating, etc.). The
    Stera ERP taxonomy is the same map drawn from the *production-line*
    angle — what shop-floor station the part visits. Both are
    defensible: DIN is the literature standard, ERP is the customer-
    validated empirical taxonomy. They overlap (BEND→DIN 8582,
    WELD→DIN 8584, …) and complement (ERP adds non-process operational
    categories like INSPECT/PACK/PROGRAM, plus Finnish-language surface
    forms that the English-only DIN vocab misses on KC drawings).

Per-entry structure:
    erp_code       Production-line code from Work_Center_Basic_Data.csv
    canonical      English canonical name
    din_group      DIN 8580 main group (1..6) or None
    din_sub_std    DIN 858x sub-standard or None
    category       'din_process' | 'qc' | 'logistics' | 'prep' | 'handling'
    source         Citation tag
    surface_forms  English + Finnish text variants (uppercase)

Finnish vocabulary notes:
    Finnish action nouns end in -us / -minen / -ys (HITSAUS = welding).
    Past participles end in -ttu / -tty / -tu (HITSATTU = welded).
    Both forms appear on KC drawings; both included in surface_forms.
    Finnish is bilingual in the KC corpus (legacy drawings, see §7.1
    Bilingual content finding).

Provenance for the citation backbone:
    Stera Technologies, "Data set documentation.docx" (2025) — describes
    the four ERP CSVs and the Work_Center_Basic_Data "Production Line"
    column as "High-level production process category (such as bending,
    welding, laser)." This is the customer's own definition of the
    taxonomy used here.
"""

from __future__ import annotations

_SRC_WC = ("Stera ERP Work_Center_Basic_Data.csv "
           "(production-line column; 2025-provided dataset)")
_SRC_WP = ("Stera ERP Work_Phases.csv "
           "(work-phase descriptions; 2025-provided dataset)")


def _e(erp_code: str, canonical: str, source: str, surface_forms: list[str],
       din_group: int | None = None, din_sub_std: str | None = None,
       category: str = "din_process") -> dict:
    return {
        "erp_code": erp_code,
        "canonical": canonical,
        "din_group": din_group,
        "din_sub_std": din_sub_std,
        "category": category,
        "source": source,
        "surface_forms": surface_forms,
    }


PRODUCTION_LINES: list[dict] = [
    # ── DIN 8582 — Forming ──────────────────────────────────────────────────
    _e("BEND", "Bending (press brake / robotic bending)",
       _SRC_WC + "; ERP work-phase 'BENDING BYSTRONIC' (351), 'SÄRMÄYS' (447)",
       ["BENT", "BENDING", "PRESS BRAKE",
        "SÄRMÄYS", "SÄRMÄTTY", "SÄRMÄTÄÄN", "SÄRMÄYS PIENET", "SÄRMÄYS HFP"],
       din_group=2, din_sub_std="DIN 8582"),
    _e("ROLLING", "Rolling",
       _SRC_WC + " (ROLLING work-center 210A)",
       ["ROLLED", "ROLLING", "HOT ROLLED", "COLD ROLLED",
        "PYÖRÖTYS", "PYÖRITYS"],
       din_group=2, din_sub_std="DIN 8582"),
    _e("TPP", "Turret punch press (sheet metal punching/stamping)",
       _SRC_WC + " (TPP = Turret Punch Press; AMADA-family machines; 11 phases)",
       ["TPP", "PUNCH", "PUNCHING", "PUNCHED", "STAMPING", "STAMPED",
        "MEISTÖ", "MEISTÄYS", "PERFOROINTI"],
       din_group=2, din_sub_std="DIN 8582"),
    _e("COMBI", "Combination machine (fiber laser + punch)",
       _SRC_WC + " ('KUITUKOMBI' work-center 205K)",
       ["COMBI", "KOMBI", "KUITUKOMBI"],
       din_group=2, din_sub_std="DIN 8582"),

    # ── DIN 8583 — Separating ───────────────────────────────────────────────
    _e("LASER", "Laser cutting (fiber / CO2)",
       _SRC_WC + " (9 LASER work-centers); ERP 'LASER 6kW' (454), 'KUITULASER' (127)",
       ["LASER CUT", "LASER CUTTING", "LASER",
        "LASERLEIKKAUS", "KUITULASER", "FIBER LASER"],
       din_group=3, din_sub_std="DIN 8583"),
    _e("PLASMA", "Plasma cutting",
       _SRC_WC + " (PLASMA work-center 204; 'Plasma' 1198 phases)",
       ["PLASMA CUT", "PLASMA CUTTING", "PLASMA",
        "PLASMALEIKKAUS"],
       din_group=3, din_sub_std="DIN 8583"),
    _e("SAWING", "Sawing",
       _SRC_WC + " (SAWING work-center 210)",
       ["SAWED", "SAWING", "SAW CUT", "BAND SAW",
        "SAHAUS", "KATKAISU"],
       din_group=3, din_sub_std="DIN 8583"),
    # CUT entry removed 2026-05-24 after audit (audit_extractor.py).
    # ERP never assigns the CUT production line on this corpus (0 ERP support,
    # 133 false positives). Real cutting operations are routed via LASER /
    # PLASMA / SAWING work centers, each with their own entry. Bare CUT /
    # CUTTING / LEIKKAUS are too generic to associate with a specific PL —
    # they would be best routed via din8580.py "Separating" group, not here.
    _e("MACHINING", "Machining (defined-edge cutting family)",
       _SRC_WC + " (MACHINING category, 9 work-centers); 'Koneistus' 65 phases. "
       "Surface-form list pruned 2026-05-24 (audit): bare 'MACHINED' / "
       "'MACHINING' caused 233 FPs (likely 'MACHINED SURFACE' in tolerance "
       "notes); kept CNC-qualified English variants + unambiguous Finnish.",
       ["CNC MACHINED", "CNC MACHINING",
        "KONEISTUS", "KONEISTETTU", "KONEISTUSKESKUS"],
       din_group=3, din_sub_std="DIN 8583"),
    _e("DRILL", "Drilling",
       _SRC_WC + " (DRILL category, 3 work-centers)",
       ["DRILL", "DRILLED", "DRILLING",
        "PORAUS", "PORATTU"],
       din_group=3, din_sub_std="DIN 8583"),
    _e("THREAD", "Threading / tapping",
       _SRC_WC + " (THREAD + THREAD.A); ERP 'THREADING' 672 phases, 'KIERTEYTYS' 332. "
       "Surface-form list pruned 2026-05-24 (audit): English 'THREADED' / "
       "'THREADING' / 'TAPPED' / 'TAPPING' caused 348 FPs because they "
       "describe hole *features* (e.g. 'M8 THREADED HOLES'), not the THREAD "
       "production line. Real thread-feature signal lives in fasteners.py "
       "METRIC_THREAD_RE. Kept only unambiguous Finnish shop-floor forms.",
       ["KIERTEYTYS", "KIERTEYTETTY", "KIERREITYS"],
       din_group=3, din_sub_std="DIN 8583"),
    _e("GRIND", "Grinding",
       _SRC_WC + " (GRIND category, 5 work-centers); ERP 'GRINDING' 69 phases",
       ["GROUND", "GRINDING", "SURFACE GROUND",
        "HIONTA", "HIOTTU"],
       din_group=3, din_sub_std="DIN 8583"),
    _e("BLAST", "Blasting (shot / sand / bead)",
       _SRC_WC + " (BLAST category); ERP 'BLASTING' 383, 'SINKOUS' 101, 'PUHALLUS' 124",
       ["BLASTED", "BLASTING", "SANDBLASTED", "SANDBLASTING",
        "SHOT BLASTED", "SHOT BLASTING", "BEAD BLASTED",
        "SINKOUS", "SINGOTTU", "PUHALLUS", "PUHALLETTU", "RAEPUHALLUS"],
       din_group=3, din_sub_std="DIN 8583"),
    _e("DEBURR", "Deburring",
       _SRC_WC + " (DEBURR category); ERP 'DEBURRING' 1096 phases",
       ["DEBURRED", "DEBURRING",
        "JÄYSTEEN POISTO", "JÄYSTEENPOISTO"],
       din_group=3, din_sub_std="DIN 8583"),
    _e("WASH", "Washing / industrial cleaning (pre-coating prep)",
       _SRC_WC + " (WASH, WASH.A); ERP 'PAINEPESU' 150, 'Ultraäänipesu' 226. "
       "Surface-form list pruned 2026-05-24 (audit): English 'CLEANED' / "
       "'CLEANING' caused 631 FPs because they appear on every ABB drawing "
       "as part of 'Degreasing cleaning' surface-prep text (paint preparation, "
       "not the WASH industrial-cleaning production line). Generic Cleaning is "
       "still caught by din8580.py at DIN 8583 group level. Kept Finnish "
       "shop-floor forms which are unambiguous.",
       ["WASHED", "WASHING",
        "PESU", "PESTY", "PAINEPESU", "ULTRAÄÄNIPESU",
        "SUOJAMUOVIN POISTO"],   # protective film removal — adjacent to cleaning
       din_group=3, din_sub_std="DIN 8583"),

    # ── DIN 8584 — Joining ──────────────────────────────────────────────────
    _e("WELD", "Welding (manual + robotic)",
       _SRC_WC + " (WELD, WELD.A); ERP 'KONECRANES WELDING' 377, "
       "'PREPARATION WELDING' 378, 'HITSAUS' + 'MUU HITSAUS' 150",
       ["WELDED", "WELDING", "WELD",
        "HITSAUS", "HITSATTU", "HITSATAAN", "MUU HITSAUS",
        "PREPARATION WELDING", "KONECRANES WELDING"],
       din_group=4, din_sub_std="DIN 8584"),
    _e("SPOT.WELD", "Resistance spot welding",
       _SRC_WC + " (SPOT.WELD, SPOT.WELD.A); ISO 4063 process 21",
       ["SPOT WELDED", "SPOT WELDING", "RESISTANCE SPOT", "RSW",
        "PISTEHITSAUS"],
       din_group=4, din_sub_std="DIN 8584"),
    _e("SOLDER", "Soldering",
       _SRC_WC + " (SOLDER work-center 241P)",
       ["SOLDERED", "SOLDERING",
        "JUOTOS", "JUOTETTU"],
       din_group=4, din_sub_std="DIN 8584"),
    _e("PEMM", "PEM nut / clinch fastener insertion",
       _SRC_WC + " (PEMM category); ERP 'PEMMAUS' 182 phases. "
       "Clinch fastener (PEM brand) press-in operation on sheet metal — "
       "between fastening (DIN 8584) and pressing (DIN 8582).",
       ["PEMM", "PEM NUT", "PEMMAUS", "PEMMATTU",
        "CLINCH NUT", "CLINCHED"],
       din_group=4, din_sub_std="DIN 8584"),

    # ── DIN 8585 — Coating ──────────────────────────────────────────────────
    _e("PAINT", "Painting (top coat + primer)",
       _SRC_WC + " (PAINT category, 12 work-centers); ERP 'PAINTING' 361, "
       "'MAALAUS' 121+225, 'POHJAMAALAUS' (primer) 53",
       ["PAINTED", "PAINTING", "PRIMER", "PRIMED", "TOP COAT", "BASE COAT",
        "MAALAUS", "MAALATTU", "POHJAMAALAUS", "POHJAMAALATTU"],
       din_group=5, din_sub_std="DIN 8585"),

    # ── Non-DIN ERP operational categories ──────────────────────────────────
    # These are not in DIN 8580 scope but are real, high-frequency phases
    # in the customer's actual production flow. Useful for filtering /
    # contextual signal (e.g. "this drawing belongs to an item that went
    # through inspection").
    _e("INSPECT", "Inspection / QC",
       _SRC_WC + " (INSPECT category); ERP 'INSPECTION' 805, "
       "'INSPECTION Welding' 401, 'KATSELMUS' 119",
       ["INSPECTED", "INSPECTION", "QC", "QA",
        "TARKASTUS", "TARKASTETTU", "KATSELMUS", "KATSELMOITU"],
       category="qc"),
    _e("PACK", "Packing",
       _SRC_WC + " (PACK category); ERP 'PAKKAUS' 1345, 'PACKING' 286",
       ["PACK", "PACKED", "PACKING",
        "PAKKAUS", "PAKATTU", "PAKKAUS LÄHETTÄMÖ"],
       category="logistics"),
    _e("COLLECT", "Material collection / kitting",
       _SRC_WC + " (COLLECT category); ERP 'KERÄILY' 'SETITYS' 132",
       ["COLLECT", "COLLECTION", "KITTING",
        "KERÄILY", "KERÄYS", "SETITYS"],
       category="handling"),
    _e("PROGRAM", "(CNC) programming",
       _SRC_WC + " (PROGRAM category); ERP 'OHJELMOINTI' 951 phases. "
       "Programming the CNC machine for the part — preparatory work-phase.",
       ["PROGRAM", "PROGRAMMING", "CNC PROGRAMMING", "CAM",
        "OHJELMOINTI"],
       category="prep"),
    _e("IRROITUS", "Removal / separation from raw stock",
       _SRC_WP + " ('IRROITUS' 707, 'IRROTUS' 791). Removing the cut part "
       "from the sheet/blank after laser/plasma/punch processing.",
       ["REMOVAL", "SEPARATION",
        "IRROITUS", "IRROTUS", "IRROTETTU"],
       category="handling"),
    _e("WORK", "Generic factory work / assembly",
       _SRC_WC + " (WORK category — most-populated, 37 work-centers); "
       "ERP 'Tehdastyö' (factory work) variants, 'Assembly' phases",
       ["FACTORY WORK", "ASSEMBLY WORK",
        "TEHDASTYÖ", "KOOSTAMINEN", "ASENNUS", "ASENNETTU"],
       category="handling"),
]


# Flat lookup. Longest first.
_LOOKUP: dict[str, dict] = {}
for _entry in sorted(
    PRODUCTION_LINES, key=lambda e: -max(len(s) for s in e["surface_forms"])
):
    for _sf in _entry["surface_forms"]:
        _LOOKUP.setdefault(_sf.upper(), _entry)


def lookup(token: str) -> dict | None:
    return _LOOKUP.get(token.upper().strip())


SURFACE_FORMS: frozenset[str] = frozenset(_LOOKUP.keys())


# Subset: Finnish-language surface forms. Built by inspection of the
# entries above. Used by `scripts/part1/compare_vocab.py` to quantify what
# the Finnish vocab catches that the English-only DIN 8580 path misses
# on KC bilingual title blocks.
FINNISH_FORMS: frozenset[str] = frozenset({
    # -us / -ys action nouns
    "SÄRMÄYS", "SÄRMÄTTY", "SÄRMÄTÄÄN", "SÄRMÄYS PIENET", "SÄRMÄYS HFP",
    "PYÖRÖTYS", "PYÖRITYS",
    "MEISTÖ", "MEISTÄYS", "PERFOROINTI",
    "KOMBI", "KUITUKOMBI",
    "LASERLEIKKAUS", "KUITULASER",
    "PLASMALEIKKAUS",
    "SAHAUS", "KATKAISU",
    "LEIKKAUS", "LEIKATTU",
    "KONEISTUS", "KONEISTETTU", "KONEISTUSKESKUS",
    "PORAUS", "PORATTU",
    "KIERTEYTYS", "KIERTEYTETTY", "KIERREITYS",
    "HIONTA", "HIOTTU",
    "SINKOUS", "SINGOTTU", "PUHALLUS", "PUHALLETTU", "RAEPUHALLUS",
    "JÄYSTEEN POISTO", "JÄYSTEENPOISTO",
    "PESU", "PESTY", "PAINEPESU", "ULTRAÄÄNIPESU",
    "SUOJAMUOVIN POISTO",
    "HITSAUS", "HITSATTU", "HITSATAAN", "MUU HITSAUS",
    "PISTEHITSAUS",
    "JUOTOS", "JUOTETTU",
    "PEMMAUS", "PEMMATTU",
    "MAALAUS", "MAALATTU", "POHJAMAALAUS", "POHJAMAALATTU",
    "TARKASTUS", "TARKASTETTU", "KATSELMUS", "KATSELMOITU",
    "PAKKAUS", "PAKATTU", "PAKKAUS LÄHETTÄMÖ",
    "KERÄILY", "KERÄYS", "SETITYS",
    "OHJELMOINTI",
    "IRROITUS", "IRROTUS", "IRROTETTU",
    "TEHDASTYÖ", "KOOSTAMINEN", "ASENNUS", "ASENNETTU",
})

ENGLISH_FORMS: frozenset[str] = frozenset(SURFACE_FORMS - FINNISH_FORMS)
