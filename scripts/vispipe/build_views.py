"""vispipe/build_views — UNIFIED visual-channel orchestrator (parallel to scripts/textpipe).

One entry point that runs the per-drawing VIEW pipeline end-to-end and collects browsable overlays:

  page-1 raster (200 DPI)  ->  polygon_removed_super  (vector geometry, glyph-tight text removal,
  title-block white-fill; render_super_vector = canonical)  ->  SAM ViT-H "everything" masks
  (view_split_sam)  ->  v5 view selector (sam_postfilter_v5)  ->  per-stem view bboxes.

Resolves the canonical-super ambiguity in ONE place: render_super_vector writes `_vec`, SAM reads
`polygon_removed_super.png` -> this orchestrator copies `_vec` -> the SAM-input name. The heavy legacy
steps (SAM, v5) are PROVEN and are invoked as-is via subprocess (SAM model loads once for all stems);
this module is the glue + the review-image collection, not a rewrite.

Outputs per stem under cache/samples/text_bbox_only/<stem>/ (sam_views/), AND browsable copies under
review/vispipe/ (raw SAM overlay + v5-filtered overlay per stem + a contact sheet).

PAGE 1 ONLY for now (matches the page-1 text baseline; multipage view-set union is a later step).

Run:  python scripts/vispipe/build_views.py --sample /tmp/visual_sample.txt [--prep-only] [--seg-only]
"""
import sys, shutil, argparse, subprocess
from pathlib import Path
import numpy as np
import cv2
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[2]          # repo root (portable: works wherever the repo lives)
sys.path.insert(0, str(ROOT / "scripts")); import paths
PART1 = ROOT / "scripts" / "part1"
SAMPLES = paths.VISUAL_PIPE                  # <stem>/ per-drawing visual working dir
REVIEW = paths.visual_images()              # browsable overlays for this run
PY = sys.executable
DPI = 200


def prep(stem):
    """Ensure page1.png + polygon_removed_super.png exist for a stem (the SAM input)."""
    d = SAMPLES / stem
    d.mkdir(parents=True, exist_ok=True)
    super_png = d / "polygon_removed_super.png"
    if super_png.exists():
        return "cached"
    # page-1 raster (render_super_vector uses it for output size)
    p1 = d / "page1.png"
    if not p1.exists():
        pdf = paths.PDF_DIR / f"{stem}.pdf"
        if not pdf.exists():
            pdf = paths.PDF_DIR / f"{stem}.PDF"
        pix = fitz.open(pdf)[0].get_pixmap(dpi=DPI)
        pix.save(str(p1))
    # vector super (canonical); writes polygon_removed_super_vec.png
    r = subprocess.run([PY, str(PART1 / "render_super_vector.py"), stem],
                       capture_output=True, text=True)
    vec = d / "polygon_removed_super_vec.png"
    if not vec.exists():
        return f"FAIL super ({r.stderr.strip().splitlines()[-1] if r.stderr else '?'})"
    shutil.copyfile(vec, super_png)        # pin canonical-super as the SAM input
    return "ok"


def seg(stems):
    """SAM 'everything' masks (loads model ONCE for all stems) then v5 view selection."""
    print(f"[seg] SAM on {len(stems)} stems (model loads once; CPU, slow)...", flush=True)
    subprocess.run([PY, str(PART1 / "view_split_sam.py"), *stems], check=False)
    print(f"[seg] v5 view selection...", flush=True)
    subprocess.run([PY, str(PART1 / "sam_postfilter_v5.py"), *stems], check=False)


def collect(stems):
    """Copy raw + v5 overlays into review/vispipe/ and build a contact sheet."""
    REVIEW.mkdir(parents=True, exist_ok=True)
    tiles = []
    for s in stems:
        sv = SAMPLES / s / "sam_views"
        raw, v5 = sv / "sam_overlay.png", sv / "sam_overlay_filtered_v5.png"
        if v5.exists():
            shutil.copyfile(v5, REVIEW / f"{s[:40]}__v5.png")
        if raw.exists():
            shutil.copyfile(raw, REVIEW / f"{s[:40]}__raw.png")
        src = v5 if v5.exists() else raw
        if src.exists():
            im = cv2.imread(str(src))
            if im is not None:
                t = cv2.resize(im, (320, int(320 * im.shape[0] / im.shape[1])))
                cv2.putText(t, s[:30], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
                tiles.append(t)
    if tiles:
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
        cols = 5
        rows = [np.hstack(tiles[i:i + cols] + [np.full((h, 320, 3), 255, np.uint8)] * (cols - len(tiles[i:i + cols])))
                for i in range(0, len(tiles), cols)]
        cv2.imwrite(str(REVIEW / "_contact_sheet.png"), np.vstack(rows))
    print(f"[collect] review images -> {REVIEW}/  (+ _contact_sheet.png)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--prep-only", action="store_true")
    ap.add_argument("--seg-only", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()
    stems = [l.strip() for l in open(args.sample) if l.strip()]
    print(f"{len(stems)} stems")

    if args.collect_only:
        collect(stems); return
    if not args.seg_only:
        ok = 0
        for i, s in enumerate(stems):
            r = prep(s); ok += r in ("ok", "cached")
            print(f"  [prep {i+1}/{len(stems)}] {r:<8} {s[:46]}", flush=True)
        print(f"[prep] {ok}/{len(stems)} have super raster")
    if args.prep_only:
        return
    seg(stems)
    collect(stems)


if __name__ == "__main__":
    main()
