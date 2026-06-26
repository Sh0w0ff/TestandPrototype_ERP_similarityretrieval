"""SAM view-selection filter v5 -- background-removal + joined-blocks, 2 layers.

Mode-A approach (user, 2026-05-31). SAM's masks are fine; this is pure selection.

OBSERVATION that drives it: in the raw SAM output there is always a BACKGROUND
mask (the beige blob) that runs AROUND the views -- it hugs the image border and
is in none of the views. Remove it and the views fall out as the connected
("joined") blobs of what remains.

LAYER 1 -- background removal + joined blocks:
  - A mask is BACKGROUND if it hugs the image border: it touches >=3 of the 4
    edges (the page-wide everything-mask), OR it fully covers any single edge
    (>=0.9 of one side -- a full-width/height band). Views may *graze* an edge a
    little but never cover a whole side, so they survive.
    NOTE (SQE): the second background blob is a full-width band along the BOTTOM;
    it also clipped a bit of the lower LEFT/RIGHT edges -- that's why the test is
    "fully covers ONE edge", not "touches one edge a little". Caught it as bg #2.
  - Union every non-background mask -> connected components (8-conn) = joined
    blocks. Each block is a candidate VIEW (its full extent, hollow interior and
    all, because it's defined by what isn't background).

LAYER 2 -- split a joined block that holds >1 view (PAIRWISE disjoint + ink-gap):
  Per block (everything RELATIVE TO THE BLOCK, since a block << page):
  1. candidates = blobs that are big for the block (cov in [SUBVIEW_OF_BLK,
     FILL_BLOCK)); the FILL_BLOCK ceiling drops the block's own background/spanner.
  2. group candidates into views with union-find. Two candidates are the SAME view
     UNLESS clearly DIFFERENT, which needs ALL of:
       - both PAIRWISE-unique areas (A\B, B\A) >= MIN_DISJ_FRAC of the block, and
       - those unique areas do NOT touch, and
       - NO INK BRIDGE between them: the drawing ink in A\B and in B\A are in
         different connected components -> a real whitespace gutter separates two
         views. If ink runs unbroken between them it's ONE continuous part that SAM
         merely cut (e.g. a side bar split into halves) -> keep as one view.
  3. >=2 groups -> split. Each view's bbox is taken on its DISJOINT pixels (group
     minus the others) so the resulting boxes don't overlap.
  KEY FIXES vs earlier attempts: uniqueness is PAIRWISE (measuring vs ALL
  candidates let 3 near-duplicate masks cancel each other to ~0 and missed 38918);
  and the splitter is gated on an INK GAP, not on mask overlap (all real splits had
  mask-overlap 0, so an overlap test would have blocked them).

Output: sam_overlay_filtered_v5.png + sam_masks_filtered_v5.json + review copy.
"""
import base64, json, sys, zlib
import sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]          # repo root (portable: works wherever the repo lives)
sys.path.insert(0, str(ROOT / "scripts")); import paths
SAMPLES = paths.VISUAL_PIPE

# --- knobs -----------------------------------------------------------------
MIN_AREA_FRAC = 0.004   # ignore masks < 0.4% of page (noise / specks)
# layer 1 -- background
BORDER_FRAC   = 0.010   # border-ring thickness = this * min(H,W)
SIDE_TOUCH    = 0.40    # a side is "touched" if the mask covers >= this of that edge
BG_MIN_SIDES  = 3       # background if it touches >= this many sides (everything-mask)
BG_FULL_EDGE  = 0.90    # ...or fully covers any one edge >= this (full-width/height band)
MIN_COMP_FRAC = 0.006   # keep a joined block only if >= this of page
# layer 2 -- split a block holding multiple views (all sizes RELATIVE TO BLOCK)
FILL_BLOCK    = 0.85    # a blob covering >= this of the BLOCK = spanner/background -> drop
SUBVIEW_OF_BLK= 0.13    # a candidate sub-blob covers >= this of the BLOCK area
MIN_DISJ_FRAC = 0.10    # a candidate's PAIRWISE-unique area must be >= this of the BLOCK
TOUCH_DIL_FRAC= 0.010   # two unique areas "touch" if within this * min(H,W) of each other
INK_DIL_FRAC  = 0.004   # dilate drawing ink by this * min(H,W) before connectivity (bridge dashes)
# layer 3 -- reject text/annotation blocks (Mode B). A view is TEXT iff BOTH:
TEXT_SPAN_MAX = 0.30    # no ink component spans >= this of the view in BOTH dims (no 2D geometry)
TEXT_DENS_MIN = 350     # AND glyph density >= this (connected comps per megapixel).
                        # The AND is deliberate: a busy real view is dense but HIGH-span
                        # (saved); a skeletal real view is low-span but SPARSE (saved); only
                        # text is low-span AND dense. Vendor-neutral, no text layer (KC-safe).
MAX_KEPT      = 10
# ---------------------------------------------------------------------------

def dec(s, H, W):
    b = zlib.decompress(base64.b64decode(s.encode()))
    return np.unpackbits(np.frombuffer(b, np.uint8), bitorder="big")[:H*W].reshape(H, W).astype(bool)

def run(stem):
    sd = SAMPLES / stem
    meta = json.loads((sd / "sam_views/sam_masks.json").read_text())
    H, W = meta["image_hw"]; page = H*W
    t = max(2, int(BORDER_FRAC*min(H, W)))
    log = {"input": len(meta["masks"])}

    # ---- LAYER 1: classify masks, build foreground -------------------------
    fg = np.zeros((H, W), bool); fgblobs = []; nbg = 0
    for m in meta["masks"]:
        x, y, w, h = m["bbox_xywh"]
        if w == 0 or h == 0 or m["area"]/page < MIN_AREA_FRAC: continue
        seg = dec(m["rle"], H, W)
        # per-side border coverage
        top = seg[:t, :].any(0).mean(); bot = seg[-t:, :].any(0).mean()
        left = seg[:, :t].any(1).mean(); right = seg[:, -t:].any(1).mean()
        sides = sum(v >= SIDE_TOUCH for v in (top, bot, left, right))
        if sides >= BG_MIN_SIDES or max(top, bot, left, right) >= BG_FULL_EDGE:
            nbg += 1; continue                       # BACKGROUND -> drop
        fg |= seg
        fgblobs.append({"seg": seg, "pix": int(seg.sum()), "bbox": (x, y, w, h)})
    log["bg_removed"] = nbg

    # joined blocks = connected components of the foreground
    ncc, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    blocks = [{"id": i, "mask": lab == i,
               "bbox": tuple(int(v) for v in stats[i, :4])}
              for i in range(1, ncc) if stats[i, cv2.CC_STAT_AREA]/page >= MIN_COMP_FRAC]
    log["joined_blocks"] = len(blocks)

    # drawing-ink connected components (dilated to bridge dashes) -- for ink-gap test
    sup = cv2.imread(str(sd / "polygon_removed_super.png"))
    if sup.shape[:2] != (H, W): sup = cv2.resize(sup, (W, H))
    ink_raw = (cv2.cvtColor(sup, cv2.COLOR_BGR2GRAY) < 250).astype(np.uint8)
    inkd = cv2.dilate(ink_raw, np.ones((2*max(1, int(INK_DIL_FRAC*min(H, W)))+1,)*2, np.uint8))
    _, ink_lab = cv2.connectedComponents(inkd, 8)

    # ---- LAYER 2: per block, PAIRWISE-disjoint grouping (see header) --------
    def bbox_of(mask):
        ys_, xs_ = np.where(mask)
        return (int(xs_.min()), int(ys_.min()), int(xs_.max()-xs_.min()+1), int(ys_.max()-ys_.min()+1))
    def ink_bridge(dA, dB):                          # ink in dA and dB share a component?
        la = set(np.unique(ink_lab[np.logical_and(dA, inkd > 0)])) - {0}
        lb = set(np.unique(ink_lab[np.logical_and(dB, inkd > 0)])) - {0}
        return bool(la & lb)
    dil = np.ones((2*max(1, int(TOUCH_DIL_FRAC*min(H, W)))+1,)*2, np.uint8)
    views = []; splits = 0
    for blk in blocks:
        bmask = blk["mask"]; barea = int(bmask.sum())
        clips = []                                   # big-for-block, non-filler, clipped
        for b in fgblobs:
            c = np.logical_and(b["seg"], bmask); cov = int(c.sum())/barea
            if SUBVIEW_OF_BLK <= cov < FILL_BLOCK: clips.append(c)
        if len(clips) < 2:
            views.append(blk["bbox"]); continue
        # union-find: same view unless clearly different (unique big, non-touching, ink-gapped)
        par = list(range(len(clips)))
        def find(i):
            while par[i] != i: par[i] = par[par[i]]; i = par[i]
            return i
        for i in range(len(clips)):
            for j in range(i+1, len(clips)):
                dA = np.logical_and(clips[i], ~clips[j]); dB = np.logical_and(clips[j], ~clips[i])
                fa, fb = dA.sum()/barea, dB.sum()/barea
                touch = np.logical_and(cv2.dilate(dA.astype(np.uint8), dil).astype(bool), dB).any()
                different = (fa >= MIN_DISJ_FRAC and fb >= MIN_DISJ_FRAC
                            and not touch and not ink_bridge(dA, dB))
                if not different: par[find(i)] = find(j)
        groups = {}
        for i in range(len(clips)): groups.setdefault(find(i), []).append(i)
        if len(groups) < 2:
            views.append(blk["bbox"]); continue
        splits += 1
        gmasks = []
        for idxs in groups.values():
            m = np.zeros((H, W), bool)
            for i in idxs: m |= clips[i]
            gmasks.append(m)
        for gi, m in enumerate(gmasks):              # box each view on its DISJOINT pixels
            others = np.zeros((H, W), bool)
            for gj, mm in enumerate(gmasks):
                if gj != gi: others |= mm
            d = np.logical_and(m, ~others)
            views.append(bbox_of(d if d.any() else m))
    log["layer2_splits"] = splits

    # ---- LAYER 3: reject text/annotation blocks (Mode B; see knobs) ---------
    def is_text(v):
        x, y, w, h = v; sub = ink_raw[y:y+h, x:x+w]
        n, _, stats, _ = cv2.connectedComponentsWithStats(sub, 8)
        if n <= 1: return False
        span = max(min(stats[k, cv2.CC_STAT_WIDTH]/w, stats[k, cv2.CC_STAT_HEIGHT]/h) for k in range(1, n))
        dens = (n-1)/(w*h/1e6)
        return span < TEXT_SPAN_MAX and dens >= TEXT_DENS_MIN
    kept = [v for v in views if not is_text(v)]
    log["text_rejected"] = len(views) - len(kept)
    views = sorted(kept, key=lambda v: -v[2]*v[3])[:MAX_KEPT]
    log["final"] = len(views)

    # ---- output ------------------------------------------------------------
    base = cv2.imread(str(sd / "polygon_removed_super.png"))
    if base.shape[:2] != (H, W): base = cv2.resize(base, (W, H))
    ov = (base.astype(np.float32)*0.55 + 60).astype(np.uint8)
    rng = np.random.default_rng(0)
    for blk in blocks:
        c = rng.integers(40, 230, 3); ov[blk["mask"]] = (0.6*ov[blk["mask"]] + 0.4*c).astype(np.uint8)
    for i, (x, y, w, h) in enumerate(views):
        cv2.rectangle(ov, (x, y), (x+w, y+h), (0, 0, 0), 3)
        cv2.putText(ov, f"{i}", (x+5, y+34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 4)
        cv2.putText(ov, f"{i}", (x+5, y+34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 2)
    cv2.imwrite(str(sd/"sam_views/sam_overlay_filtered_v5.png"), ov)
    (ROOT/"review").mkdir(exist_ok=True)
    scl = 1300/max(H, W)
    cv2.imwrite(str(ROOT/"review"/f"{stem.split('_')[0][:20]}_v5.png"), cv2.resize(ov, (int(W*scl), int(H*scl))))
    out = [{"bbox_xywh": [int(a) for a in v], "bbox_area_frac": round(v[2]*v[3]/page, 3)} for v in views]
    (sd/"sam_views/sam_masks_filtered_v5.json").write_text(json.dumps({"counts": log, "n": len(out), "views": out}, indent=2))
    print(f"{stem[:28]:30} {log}")

if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(s)
