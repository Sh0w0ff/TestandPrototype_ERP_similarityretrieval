# textpipe — self-contained text-extraction pipeline (checkpoint 2026-06-03)

A from-scratch, **parity-tested** rebuild of the unified per-drawing SIGNAL builder. Replaces the
runtime hard-dependency on the legacy "mixed" scripts (`scripts/part1/*`) with **copies** of their
pure functions in focused modules. The legacy scripts stay **frozen as the reference**; a parity test
per module guards each copy against drift.

## Layout
```
scripts/textpipe/
  build_signal.py            orchestrator: route vector/ocr -> assemble signal.json + 3-panel dump png
  text_scripts/
    region.py    geometry: find cells + build super-removed body raster   (single geometry source)
    box_text.py  VECTOR text per box cell        |  box_ocr.py  OCR text per box cell
    body_text.py VECTOR body text (canonical)    |  body_ocr.py OCR body text (tiled)
    bom.py       BOM finder + row parser (both layouts)
    vocab_tags.py  phrase-level tagging over spatial units (NEW logic, not a legacy copy)
    _util.py     light_clean + is_zone_marker (copied from body_text_filter)
  tests/         test_<module>_parity.py  — all PASS
```
Output: `cache/signal_v2/<stem>/signal.json` + `signal_dump.png` (DRAWING | RAW PATHWAYS | FILTERED FINAL).
Routing: `<=50` pymupdf words -> OCR, else vector (`cache/pymupdf_word_counts.csv`).

## signal.json schema (current)
```
title_block: { cells:[{bbox_pt,text}], zone_markers:[...], fields:{}, standards:[...] }
body:        [ {bbox, text, notes:[...], dims:[...], dropped:{...}, tags:[{clause,negated,tags:{cat:[anchors]}}]} ]
             # BODY IS A LIST OF SPATIAL BLOCKS — never flattened, parallel to title_block.cells
body_dropped_blocks: [ "a4", "B", "22", ... ]           # low-char (<=2) noise blocks filtered out, recorded
bom_header:  [ "pos","qty","description", ... ]          # detected BOM column headers (grid layout)
bom_rows:    [ KC-schema row dicts ]                     # the prediction TARGET / KC baseline
vocab_tags:  { category: [ {phrase, anchors, negated} ] }  # phrase-level, aggregated over box cells
             # + body blocks; surface-domain (corrosion+surface_texture+RAL+KC) FOLDED into 'surface'
```

## Function provenance (copied verbatim from frozen legacy)
| new module | copies from | functions |
|---|---|---|
| region | ocr_box_experiment + ocr_body_experiment | find_pdf, render, detect_cells, axis_segments, border_mask_crop, ink_frac, filter_cells; composed: cells_deg/cells_kept/vector_removed |
| box_text | ocr_box_experiment | pymupdf_words_in |
| box_ocr | ocr_box_experiment | paddle_ocr + poly-masked crop |
| body_text | vector_lines_poly | outside-cell words + dedup + block grouping (canonical) |
| body_ocr | ocr_body_experiment | paddle_lines, group_lines, tiled_lines, order_members |
| bom | ocr_table_parse | cell_text(adapted), looks_like_bom_row, parse_row, parse_grid_columns, HEADER_VOCAB |
| _util | body_text_filter (+ build_signal) | light_clean, is_zone_marker |
| vocab_tags | NEW (uses `vocab.ALL_STANDARDS_LIBS`) | clause split + 14-DB lookup + curated surface anchors + FOLD(corrosion,surface_texture→surface) + clause negation. Tags run AFTER light_clean (dims out → no numeric FP). |

## Parity results (run `tests/test_*_parity.py`)
- region: strict bounds-set match + pixel-diff 0 (cells overlay AND super-removed raster).
- box_text: 13 stems, per-cell text identical to `pymupdf_words_in`.
- box_ocr: 52/52, 36/36, 55/55 vs cached `ocr_boxes.json`; **same OCR time** as legacy.
- body_text: 98/98, 101/101, 105/105 blocks vs the REAL `vector_lines_poly` (subprocess).
- body_ocr: 12/12, 24/24 groups vs live `ocr_body_experiment.run`; **same OCR time**.
- bom: 12/12 (per-column grid), 1/1 (single-row), 6/6 (whole-row) vs `ocr_table_parse`.

## Divergences vs old `dwgsignal`/legacy signal (intentional)
1. **body_text CANONICAL**: outside-set = ALL `detect_cells` + word de-dup + block grouping (== vector_lines_poly).
   Old `dwgsignal/body.py` used filtered `cells_kept` + no de-dup. SQEM: 191 raw -> 102 notes (new) vs 153 (old).
2. **Body is a LIST of spatial blocks, NEVER flattened** (parallel to `title_block.cells`). Each block keeps its
   bbox + raw text + its own `light_clean` (notes/dims/dropped) + its own phrase `tags`. Old `dwgsignal` joined the
   whole body into one `body_notes` string. (SQEM = 98 blocks, 53034084 = 12.)
3. **vocab_tags is phrase-level + unified** (`vocab_tags.py`): clause-tag each spatial unit (box cell / body block)
   AFTER light_clean, aggregate to `{cat:[{phrase,anchors,negated}]}`. Surface domain (corrosion + surface_texture +
   RAL + KC paint) is **FOLDED into one `surface` category** — no separate `surface_treatment` block. Negation is
   per-clause inline (`negated:true`), not a separate top-level dict. Legacy used only 4 inline `extract_text` regexes.
4. **Tagging runs AFTER `light_clean`** so bare dimensions are already routed out → no numeric false-matches
   (e.g. dim `22` never hits ISO 4063 code 22); no special numeric guard needed.
5. standards scan box + body unit text; KEYVAL `fields` box-only (interpret layer). RAL now flows through the
   `surface` vocab anchor, not a standalone `ral_codes` field.
6. **zone_markers** filtered out of `title_block.cells` (single-char + lone 1–2 digit border refs), recorded.
7. Output `cache/signal_v2/` (old `cache/signal/`). Both coexist. `extract_text` retained only as interpret layer
   (its `pair_blocks_spatial` + `raw_tokens` still unused).

## Known issues / next
- **Whole-BOM-sheet pages are THROWN for now (future parser, not now — user 2026-06-05):** a KC drawing
  whose page-1 is a standalone *multi-level BILL OF MATERIAL* sheet has that sheet's prose header block
  shelved to `debug.admin` (matched by `is_admin` on `BILL OF MATERIAL`) to keep its component
  Description/Specification text OUT of the input signal (leak-safe). The structured BOM *table* on that
  sheet is still parsed into `bom.rows` (the TARGET) where a table grid is detected, but any BOM data that
  only appears in that prose header is currently discarded. FUTURE IMPROVEMENT (not now): a dedicated
  parser that converts the whole-BOM-sheet content into proper `bom`-field rows, so we capture that
  target data fully (better TARGET coverage / KC baseline). Tracked in thesis_direction §9.8.27.
- **Leakage to fix (priority):** embedded-BOM table cells appear in BOTH `title_block.cells` (input feature) AND
  `bom_rows` (target). Separate them: mark `bom_cells`, exclude from the input-text feature. (Do NOT strip
  vocab-tagged words from body — a tag is an additive normalized projection, no leakage.)
- vocab tagger refinements: negation is clause-scoped (over-negates `machined/threaded` in "No paint in threaded
  and machined holes"); weld-size forms (`a4`,`s8a6`) + fit codes (`H11`) not yet caught; KC paint `KC1` only via
  surface phrase anchor, not a typed code (left for downstream per user).
- Comparison images in `cache/region_review/`: CMP_boxes, CMP_vectorremoved, CMP_boxocr, CMP_bodytext, CMP_bodyocr,
  CMP_full_A/B, CMP_blocktags.
