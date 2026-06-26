"""_util — shared helpers for the text pipeline (copied, not imported, from legacy).

light_clean: the LIGHT body-text cleaner (copied verbatim from scripts/part1/body_text_filter.py).
Policy: KEEP every real note word; ROUTE bare dims/tolerances to a separate list (not deleted);
DROP only unambiguous garbage (non-ASCII/CJK, O-runs, symbol-only, <2-char fragments). The vocab
is a downstream normalizer, NOT a gate here.
"""
import re
from collections import defaultdict

RE_ONLY_O = re.compile(r"^[OoＯ0oº]+$")
RE_TOL    = re.compile(r"^[+\-]\d+(?:[.,]\d+)?$")
RE_DIM    = re.compile(r"^[R⌀ØøΦϕ(]?\s?\d+(?:[.,]\d+)?(?:\s?[x×]\s?\d+)?\)?(?:mm|MM|pcs|PCS|°)?$")
RE_SYMBOL = re.compile(r"^[^\w]+$")


def light_clean(text):
    """text -> (kept_notes[list], routed_dims[list], dropped[dict reason->tokens])."""
    kept, dims = [], []
    dropped = defaultdict(list)
    for tok in text.split():
        t = tok.strip()
        if not t:
            continue
        if not t.isascii():
            dropped["non_ascii_symbol"].append(t); continue
        if RE_ONLY_O.match(t):
            dropped["o_run_holes"].append(t); continue
        if RE_SYMBOL.match(t):
            dropped["symbol_only"].append(t); continue
        if RE_TOL.match(t) or RE_DIM.match(t):
            dims.append(t); continue
        if len(t.strip(".,:;()-")) < 2:
            dropped["short_fragment"].append(t); continue
        kept.append(t)
    return kept, dims, dict(dropped)


def is_zone_marker(text):
    """A box cell carrying no title-block signal: (a) ONLY single-char tokens (Z, M, 'L A',
    'F 1', B/U/D/C/E); or (b) a lone 1-2 digit number (border column refs: 7, 12)."""
    toks = text.split()
    if not toks:
        return False
    if all(len(t) == 1 and t.isalnum() for t in toks):
        return True
    return len(toks) == 1 and toks[0].isdigit() and len(toks[0]) <= 2
