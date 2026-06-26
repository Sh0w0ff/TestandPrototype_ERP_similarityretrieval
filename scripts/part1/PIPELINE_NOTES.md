# Part-1 pipeline — engineering findings

Running log of non-obvious findings that shape the preprocessing/view-seg pipeline.

---

## 2026-06-03 — OCR scale-check, finder-parity proof, vector_removed fix

**Scale check (7-stem sample, both vendors).** Box+body OCR validated beyond the 2 initial stems.
- **ABB is also fully path-rendered** (not KC-only): `3AUA0000091190`, `3AXD50000000762` = 0 pymupdf
  words; same pipeline recovers their title block (`Material: HDG STEEL SHEET 2,5MM EN10327…`,
  `Gen Tol: ISO 2768-M`, `Weight kg`). The ≤50-word band = **162 KC + 8 ABB**.
- **Box OCR** (title block + BOM) = strong, vendor-agnostic — the high-value recovery.
- **Body OCR** good for NOTES (`Surface treatment: KC1, RAL7021`, `See drawing QINST-1-A…`,
  `All corners R10`) but unreliable for DIMS; section hatching / centre-lines / bolt-holes →
  garbage tokens (CJK runs, `O`-strings). → use body OCR as a note recoverer, not a dim source;
  add a token garbage filter (drop non-ASCII / pure-`O`) at integration.
- Corpus pymupdf word counts cached → `cache/pymupdf_word_counts.csv` (no recompute for Fact-30 band).

**Finder parity proven.** `compare_finders.py` runs BOTH the real `vector_lines_poly.py` and the OCR
`detect_cells` on random fresh stems → cell counts IDENTICAL (60/60, 686/686, 59/59, 590/590; 2 ABB +
2 KC). Montages in `cache/finder_compare/<stem>/` (orig_/ocr_ textboxes + vector_removed + montage.png).

**vector_removed BUG fixed (user-spotted).** `ocr_body_experiment.py` was white-filling only the
ink/containment-FILTERED cell subset when building `body_vector_removed_300.png`, so empty grid/
title-block boxes SURVIVED as ruling lines → body OCR garbage + looked like the finder "missed
polygons." FIX: fill the FULL detected set (`cells_deg` = all `detect_cells` minus the 1 page-spanning
frame poly). Filters now choose only WHAT TO OCR, never what to remove from the body image.
- `detect_cells` is byte-identical to `vector_lines_poly` polys_kept; the divergence was purely the
  downstream filters. Production (`vector_lines_poly.py` `polygon_removed.png`) was already correct
  (fills all polys_kept) → integration inherits the right behaviour. `ocr_unified.py` routes a single
  full-page pass (no vector_removed) so the bug never applied there.
- Also ported the original's **outer-frame mask + tight content-crop** into the OCR body path
  (shared `border_mask_crop` in `ocr_box_experiment.py`) so the body raster zooms in identically and
  drawing-frame zone markers (A–E / 1–12) fall outside the crop. Used by ocr_body + compare_finders.

**New scripts:** `compare_finders.py` (parity harness), `ocr_table_parse.py` (structured-BOM
anchor-parse prototype — written, validated by inspection on SQCH multi-row BOM, not yet wired).

---

## 2026-06-02 — TEXT-channel OCR fallback for path-rendered text (box + body)

**Problem.** ~167/1829 PDFs (9.1%, ≤50 pymupdf words; 135 have *zero*) render their text as
vector outlines, so `get_text` returns nothing — the title block, embedded BOM table, notes and
dimensions are all visible in the raster but invisible to the text layer (see
`project_kc_path_rendered_text`). OCR-on-the-raster recovers them. Built SEPARATELY from the
production extractor first (outputs in `cache/ocr_experiment/<stem>/`) to prove it out.

**Two targets, both eventually feeding the existing text+location files** (`polygon_text.txt` for
boxes, `polygon_removed_text.txt` for body), used only as a FALLBACK where pymupdf is empty:
- **(A) Box OCR** — title-block / BOM-table CELLS. `scripts/part1/ocr_box_experiment.py`.
- **(B) Body OCR** — the drawing body (`vector_removed`): free-floating notes + dimension values.
  `scripts/part1/ocr_body_experiment.py` (+ `ocr_unified.py` = one-pass route-to-cell-or-body variant).

**Box OCR method (A).** Reuse the `vector_lines_poly.py` cell detector (segments → shapely
`polygonize` → area filter → edge-connectivity) → render page @ **300 DPI** (≈35–40px text on A2;
cells are tiny so high DPI is cheap) → for each cell crop its bbox, **mask the crop to the true
polygon** (white outside, so L/T-shaped cells don't OCR neighbour text in the notch) → OCR. Two
pre-OCR filters cut ~577→56 cells: **ink** (drop cells with <0.3% dark interior = empty grid cells)
and **containment dedup** (drop aggregate cells wholly containing ≥2 smaller ones = the duplicate
title-block "blob" reads). Recovers the full embedded KC BOM (`6 PLATE EN10029-PL12-S355J2+N 194 207
52253059 1.9` …) + all title-block fields, incl. on the 0-word SQE side plate.

**OCR engine = PaddleOCR** (3-way bake-off vs Tesseract + EasyOCR, `ocr_engine_bakeoff.py`):
- **Decisive: only PaddleOCR reads rotated/vertical dims** (90°-rotated `R=5000`); Tesseract + EasyOCR
  both fail — and engineering drawings are full of vertical dimensions.
- EasyOCR runs on Apple **MPS (GPU)** and is ~2.5× faster/crop, but has systematic **digit errors**
  (`R5o00`, `PL2O`, `DZOH12`) → disqualifying for part numbers/dims.
- Tesseract is fast and ties on clean horizontal text with `--psm 6` (keeps decimals — the earlier
  "drops decimals" was a psm artifact), but fails rotation. Kept only as a cheap fallback.
- **GPU note:** PaddlePaddle on macOS-ARM is **CPU-only** (`compiled_with_cuda=False`, no Metal/MPS
  backend). Run the production pass over 1829 PDFs on the CUDA box (`paddlepaddle-gpu`, server models).

**Body OCR method (B), with the load-bearing speed/quality fix.** Build `vector_removed` @ 300 DPI
(white-fill the detected cells on the page render) → **tiled** OCR (2×2 overlapping tiles, map boxes
back, IoU-dedupe seams) → cluster text lines into GROUPS (union-find on margin-expanded boxes) →
order each group by **row-band then x** (`order_members`, fixes the y-first scramble). Engine config
that matters:
- `text_detection_model_name="PP-OCRv5_mobile_det"` (whole-page body det **6 min → ~14 s** vs the
  heavy server det); disable doc-orientation / unwarp / textline-orientation (clean upright crops).
- **`text_det_limit_side_len=960, text_det_limit_type="max"` — LOAD-BEARING.** At the default high
  resolution the mobile det over/under-segments small multi-line PROSE (`Machining and drilling before
  painting` → `Machinnll`) AND is slower. **Diagnosis: detection RESOLUTION, not the model nor the
  rec** — same mobile det reads the note perfectly on an 820px crop but fails on a 3719px tile;
  mobile-det+server-REC gives the identical bad result. Behaviour is non-monotonic in tile size (3×3
  read the note, 2×2 and 4×4 didn't), so tuning tile-size is fragile — **capping det resolution at the
  model's native ~960 is the robust knob.** Result: **2×2 + det_limit=960 → note correct + 8/8 dims +
  14 s** (was 32 s and broken). Both faster AND more accurate (det processes smaller images).

**Coordinate / engine gotchas (will bite if forgotten):** PaddleOCR auto-resizes any input >4000px
(`max_side_limit`); Tesseract via a cv2 **ndarray** throws `UnicodeDecodeError` (Leptonica PNG-read
quirk emits binary on stderr) — write a temp PNG and OCR the **file path**; PaddleOCR 3.5 result API =
`ocr.predict(input=img)[0]` → `rec_texts` / `rec_scores` / `rec_boxes`.

**Still open:** route table-cell tokens to COLUMNS by x (structured BOM, not flat concat); wire box+body
OCR into the real extractor (`poly_words` / `outside_words`); port `render_super_vector`'s direct-vector
text removal to the normal `polygon_removed.png` (still old white-box). Preserved pre-geometry-fix box
script at `ocr_box_experiment_v1_bboxsquares.py`.

---

## 2026-06-01 — STAGE 2 per-view embedding go/no-go (`embed_views.py`)

Built `embed_views.py` and ran the visual-channel go/no-go on the 10-stem v5 crops.

**Coordinate gotcha (load-bearing):** v5 `bbox_xywh` live in the **SAM working space**
`sam_masks.json["image_hw"]`, NOT the native `polygon_removed_super.png` size. Must
`cv2.resize(super, (W, H))` to `image_hw` before cropping (this is what `sam_postfilter_v5.py`
does internally). For 038918 they happen to coincide (1955×1285); for other stems they differ.

**Line-art preprocess that worked:** grayscale → ink mask (`g < 200`) → dilate by
`k=round(max(w,h)/224)` (thin strokes survive the 224 downscale) → render clean white-bg/black-ink →
pad to square → resize 224 (INTER_AREA) → 3ch → ImageNet norm. Frozen DINOv2-small CLS, L2-norm.

**Result (qualitative, n=10):** pipeline works end-to-end; crops clean. Frozen DINOv2 *absolute*
cross-drawing cosine is compressed high (pooled 0.79–0.94) → no fixed threshold separates part-types
(the predicted line-art domain gap). BUT per-view **ranking / mutual-NN is meaningful**
(thin-sliver↔thin-sliver 0.93; big front-panels cluster 0.87–0.90; lone isometric isolated at 0.58).
→ **rank-based assignment matcher is right; absolute-cosine thresholding is not.** Soft NO-GO for
frozen-only, GO for pipeline + adaptation. Eyeball artifacts: `review/embed_views/per_view_nn.png/.tsv`,
`all_crops_contact.png`, `match_*.png`. Full detail: `thesis_direction.md` §9.8.16.

**NEXT:** labelled same- vs different-part-type eval; strip dimension lines from crops first (they
inflate similarity); then SSL contrastive adaptation if frozen confirmed weak.

---

## 2026-05-30 — Lineweight separation: ABB-only, RECORDED BUT NOT USED

**Finding.** Vector stroke-width can cleanly separate object geometry from
text + dimension lines — but **only on ABB drawings**, not Konecranes (KC).

- These CAD PDFs render *text itself as vector strokes*, so an all-paths vector
  render reproduces the text too (not just geometry).
- **ABB** (`3AUA*`): text + dimensions are the thin cluster (~0.142 pt); object
  outlines are thick (~0.567 / 0.85 pt). Dropping the thinnest cluster yields a
  clean geometry-only image (verified on 3AUA0000038918: all 5 views intact,
  text/dims/title-block gone).
- **KC fails**: text and object geometry share the same width.
  - `52919976`: object lines = 0.2 pt = same as text → dropping thinnest deletes
    the part geometry, leaves only frame/title-block.
  - `SQE side plate`: text = 0.72 pt = same as geometry → dropping thinnest
    leaves the full drawing *with* text.
  - KC text layer is also unreliable: `52919976` has 88 `get_text` words;
    `SQE` has **zero** (all path-rendered, invisible to pymupdf).

**Decision (2026-05-30): DO NOT use lineweight filtering in the main pipeline.**
Rationale: it would clean ABB inputs but not KC, biasing similarity/BOM results
between vendors. If we ever want to exploit it, do it as a **clearly-scoped
ABB-only ablation/experiment**, reported as such — never silently in the shared
pipeline.

Probe: `scripts/part1/probe_lineweights.py`.
Experimental renderer (keep for ABB-only tests): `scripts/part1/render_vector_only.py --thick-only`.

---

## 2026-05-30 — Vector-render super (replaces white-box wipe)

**Problem.** The old `polygon_removed_super.png` was built by rasterizing the
whole page (text + geometry) then **painting white rectangles over text word
bboxes**. Wherever a dimension number / annotation balloon overlapped a drawing
line, that white box erased the geometry underneath → ragged "eaten" view edges
that broke SAM view detection.

**Fix (validated on ABB 38918 + KC 52919976 + KC SQE).** A faithful port of the
original `vector_lines_poly.py` super logic — same polygon title-block
detection, same crop/zoom — changing ONLY the geometry-rendering medium:

1. Render geometry from vector paths (`get_drawings` replayed via the pymupdf
   Shape API), `closePath=False` (closing open polylines drew a corner-to-corner
   diagonal — bug, fixed).
2. **Text removal** = glyph-tight suppression of `get_text` word boxes (drop a
   path only if >60% of its bbox sits in a word box). Thin axis-aligned lines get
   a minimum bbox thickness (`LINE_FLOOR`) first, else a long view edge whose
   midpoint sits by a dimension number was wrongly suppressed (ate the 38918
   square-view right edge — bug, fixed).
3. **Title-block / frame removal** = WHITE-FILL the detected polygons on the
   rendered raster. Area-fill is geometry-safe. (An earlier attempt suppressed
   *paths* whose bbox was inside a polygon — that ate view slots/arcs/lines
   because the polygonize mesh links view interiors to the frame via dimension
   lines. Path-suppression rejected; area white-fill kept.)
4. Crop to non-white → matches the original crop to within ~1px on all 3 tested
   stems.

Net: no white boxes over text → **no collateral geometry erasure**, and the
crop/zoom-to-views behaviour of the original super is preserved.

**Residuals (none eat geometry):**
- Dimension lines/arrows survive (no lineweight → not removed).
- KC text largely survives: 52919976's text layer is misaligned with its
  path-rendered glyphs; SQE has no text layer at all ([[project_kc_path_rendered_text]]).
- ABB keeps a few faint path-rendered stray chars (no text-layer entry).

Script: `scripts/part1/render_super_vector.py` (`--debug` = suppressed text paths
in red + skips the white-fill). Original supers backed up as
`polygon_removed_super_raster.png`; swap is reversible via `vector_lines_poly.py`.

**ADOPTED 2026-05-30 (user sign-off):** `render_super_vector.py` is now the
canonical `polygon_removed_super.png` generator. Legacy raster super kept as
`polygon_removed_super_raster.png` backup (regenerable via `vector_lines_poly.py`).
All 10 sample stems swapped over and SAM re-run on them.

**Still open:** surviving dimension lines (no lineweight) and KC text (no text
layer) don't break SAM but aren't removed; thin open-profile / skeletal views
(52919976 bottom H-frame, 38918 channel section) are still under-detected by SAM
— a region model can't propose low-fill views. Being worked drawing-by-drawing.

---

## 2026-06-01 — View selection from SAM masks: `sam_postfilter_v5.py` (CANONICAL)

**Context.** SAM (`view_split_sam.py`) over-produces masks per drawing: a
page-spanning "everything" mask, loose multi-view blobs, the real per-view blobs,
and many sub-facets (triangles tracing chamfers, circles on holes). Selecting the
true views from this pile is the whole problem. SAM itself is fine — earlier
filters (v1 biggest-wins, v2 density-leaf-picker, v3 dominant-blob, v4 seg-gutter)
each failed on either hollow outline views, stacked views, or text blocks. v5 is
the version that holds on all 10 sample stems. **It is tuned on 10 stems — the
thresholds below need validation on a larger corpus slice before production.**

Input per stem: `sam_views/sam_masks.json` (bbox + area + RLE seg) + the vector
super `polygon_removed_super.png`. Output: `sam_masks_filtered_v5.json` (view
bboxes) + `sam_overlay_filtered_v5.png`.

**Three layers:**

1. **Background removal + joined blocks.** A SAM mask is BACKGROUND if it hugs the
   image border — touches >=3 of the 4 edges (the page everything-mask) OR fully
   covers any single edge (>=0.9 of one side; a full-width/height band). Real
   views never cover a whole edge (they may graze one). Drop background, OR the
   rest into a foreground, take 8-connected components = "joined blocks". Each
   block is a candidate view at full extent (hollow interior included, because it
   is defined by what is NOT background). NB: some drawings have a multi-piece
   background (SQE = 2 blobs: page + a bottom band).

2. **Split a block that holds >1 view.** Two stacked views can fuse into one block
   via a spanner mask. Per block (sizes RELATIVE TO THE BLOCK): candidates = blobs
   covering [0.13, 0.85) of the block (0.85 ceiling drops the block's own
   spanner/background). Union-find groups candidates as the SAME view UNLESS
   clearly DIFFERENT, needing ALL of: (a) both pairwise-unique areas A\B and B\A
   >= 0.10 of block, (b) those unique areas don't touch, (c) NO INK BRIDGE — the
   drawing ink in A\B and B\A lie in different dilated-ink connected components (a
   real whitespace gutter, not one continuous part SAM merely cut, e.g. a side bar
   split into halves). >=2 groups -> split; each view's bbox is taken on its
   DISJOINT pixels so boxes don't overlap.
   - Uniqueness MUST be pairwise (vs-all-candidates lets near-duplicate masks
     cancel each other to ~0 and misses real splits).
   - The split gate is the INK GAP, not mask overlap — genuine splits had mask
     overlap 0, so an overlap test would block them.

3. **Reject text/annotation blocks (Mode B).** KC notes are path-rendered vector
   text with no text layer, so a text-layer filter can't see them; use a visual
   rule. A view is TEXT iff BOTH: (a) 2D-span < 0.30 — no ink component spans
   >=30% of the view in BOTH width and height (no real 2D geometry, just glyph
   rows) AND (b) glyph density >= 350 connected-components / megapixel. The AND is
   complementary: a busy real view is dense but HIGH-span (kept); a skeletal real
   view is low-span but SPARSE (kept); only text is low-span AND dense. (Density
   alone fails — a busy real view e.g. 52919976's H-frame hits 1108 CC/Mpx,
   higher than any text block. First metric tried, biggest-CC DIAGONAL, was fooled
   by a wide text underline -> use the min(w,h) 2D-span instead.)

**Validation (all 10 stems).** Every real view captured incl. 38918's stacked
plan+lower split (5 views, non-overlapping boxes) and 52919976's bottom H-frame;
side bars stay whole (ink bridges the halves); all 3 annotation blocks rejected
(52262640 thermal-cut note, SQCH surface-treatment, SQE standard-side-plate) with
no real view dropped. Side-by-side review imgs: `review/compare_v5/`, montage
`review/v5_montage.png`.

**Superseded (kept for reference):** `sam_postfilter_v4.py` (everything-drop +
seg-gutter "crossing" reject + greedy), `_v3.py`/`_v3_seggutter.py`,
`_v2.py` (density leaf-picker), `_v1` = `sam_postfilter.py`.
