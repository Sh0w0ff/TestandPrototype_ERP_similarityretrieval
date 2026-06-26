"""Assembly-action vocabulary — BOM-relevant verbs that are NOT manufacturing processes.

Why this is a separate module from `din8580.py`:
    DIN 8580 governs how an individual part is MADE (casting, forging,
    machining, coating, ...). Assembly is excluded from DIN 8580's scope.
    But for the thesis endgoal — BOM and product-structure generation —
    assembly verbs are highly relevant because they describe the relation
    between a parent assembly and its child components ("installed on",
    "mounted to", "fitted into").

    Keeping these separate from DIN 858x means:
      - The DIN-anchored process vocabulary stays methodologically clean
        (only processes the standard actually covers).
      - The assembly vocabulary can be cited against a different source
        (mechanical-assembly textbook chapters, not DIN 8580).
      - Downstream consumers can ask for "processes" vs "assembly actions"
        independently when building BOM features.

Sourcing (full citation context in thesis_direction.md §7):
    Groover §32  — "Mechanical Assembly" (Fundamentals of Modern Manufacturing 4e,
                   Wiley 2010). Covers threaded fasteners, rivets, eyelets,
                   press / shrink fits, snap fits, retaining rings, sewing,
                   adhesive bonding (also in DIN 8584).
    Note         — DIN 8593 exists ("Manufacturing processes joining") but is
                   joining-focused (already covered by DIN 8584 in din8580.py).
                   Assembly *actions* like installation/mounting are operational
                   verbs that appear on shop-floor drawings and assembly
                   instructions; no single standard codifies them, so the
                   citation backbone here is the textbook chapter.

Categories used here:
    installation   — placing the part into / onto the parent assembly
    fastening      — securing it once placed (separate from the joining
                     *process*, which is in DIN 8584)
    fitting        — geometric mating (press fit, slide fit, etc.)
    positioning    — spatial alignment verbs
    assembly_meta  — generic "assembled / assembly" labels

Per-entry structure (mirrors din8580.py for downstream uniformity):
    category:       Sub-category label (above)
    canonical:      English action name
    source:         Citation tag
    surface_forms:  Text variants found on drawings (uppercase)
"""

from __future__ import annotations


def _e(category: str, canonical: str, source: str, surface_forms: list[str]) -> dict:
    return {
        "category": category,
        "canonical": canonical,
        "source": source,
        "surface_forms": surface_forms,
    }


ASSEMBLY_ACTIONS: list[dict] = [
    # ── Installation / mounting (BOM-critical: child→parent placement) ─────
    _e("installation", "Install (place into parent assembly)",
       "Groover §32 (Mechanical Assembly) — installation is the umbrella for "
       "assembling a component into its parent unit.",
       ["INSTALL", "INSTALLED", "INSTALLING", "INSTALLATION"]),
    _e("installation", "Mount (secure onto parent assembly)",
       "Groover §32 — mounting overlaps with installation but typically "
       "implies external attachment (e.g. motor mounted to frame).",
       ["MOUNT", "MOUNTED", "MOUNTING"]),

    # ── Fastening (the verb of using a joining process, not the process) ──
    _e("fastening", "Fix / secure in place",
       "Generic shop-floor verb; not in DIN 8580. Common in assembly notes "
       "('fix with M10 bolts'). Logged for completeness — coarse signal.",
       ["FIX", "FIXED", "FIXING", "SECURED", "SECURING"]),
    _e("fastening", "Tighten (to torque)",
       "Groover §32.1.5 (Tightening of Threaded Fasteners). Often paired "
       "with a torque spec.",
       ["TIGHTEN", "TIGHTENED", "TIGHTENING", "TORQUE"]),

    # ── Fitting (geometric mating; precedes/replaces fastening) ────────────
    _e("fitting", "Fit (geometric mating, type unspecified)",
       "Groover §32.4 (Press fits and Shrink fits) for press/shrink; "
       "general 'FITTING' callout on drawings often means dimensional pairing.",
       ["FIT", "FITTED", "FITTING"]),
    _e("fitting", "Press fit",
       "Groover §32.4.1 (Interference fits).",
       ["PRESS FIT", "PRESS FITTED", "PRESS FITTING"]),
    _e("fitting", "Shrink fit",
       "Groover §32.4.2 (Thermal interference fits).",
       ["SHRINK FIT", "SHRINK FITTED", "SHRINK FITTING"]),
    _e("fitting", "Slide fit",
       "Common dimensional callout (clearance fit family); ISO 286 tolerance "
       "system territory (not held — surface form only).",
       ["SLIDE FIT", "SLIDING FIT", "SLIP FIT"]),
    _e("fitting", "Snap fit",
       "Groover §32.5 (Snap fits and retaining rings).",
       ["SNAP FIT", "SNAP FITTED"]),

    # ── Positioning / alignment (often appears alongside installation) ─────
    _e("positioning", "Position / locate",
       "Generic; not in any specific standard. Useful for BOM as a "
       "context verb around installation callouts.",
       ["POSITION", "POSITIONED", "POSITIONING", "LOCATE", "LOCATED", "LOCATING"]),
    _e("positioning", "Align",
       "Generic shop-floor verb.",
       ["ALIGN", "ALIGNED", "ALIGNING", "ALIGNMENT"]),
    _e("positioning", "Center / centre",
       "Generic positioning callout (US + UK spelling).",
       ["CENTERED", "CENTRED", "CENTERING", "CENTRING"]),

    # ── Assembly meta (the umbrella term itself) ───────────────────────────
    _e("assembly_meta", "Assemble (generic assembly action)",
       "Groover §32 (Mechanical Assembly) umbrella.",
       ["ASSEMBLE", "ASSEMBLED", "ASSEMBLING", "ASSEMBLY"]),
    _e("assembly_meta", "Sub-assemble (intermediate assembly level)",
       "BOM-specific term; not in DIN/Groover but standard in production-"
       "management literature (e.g. multi-level BOM).",
       ["SUBASSEMBLY", "SUB-ASSEMBLY", "SUB ASSEMBLY"]),
]


# Flat lookup: uppercase surface form → entry dict. Longest first.
_LOOKUP: dict[str, dict] = {}
for _entry in sorted(
    ASSEMBLY_ACTIONS, key=lambda e: -max(len(s) for s in e["surface_forms"])
):
    for _sf in _entry["surface_forms"]:
        _LOOKUP.setdefault(_sf, _entry)


def lookup(token: str) -> dict | None:
    """Return assembly-action entry for an uppercase surface-form token, or None."""
    return _LOOKUP.get(token.upper())


SURFACE_FORMS: frozenset[str] = frozenset(_LOOKUP.keys())
