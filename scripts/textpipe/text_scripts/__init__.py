"""text_scripts — self-contained text-extraction modules for the unified SIGNAL pipeline.

Each module owns one responsibility and carries its own copy of the necessary functionality
(copied — NOT imported — from the proven legacy scripts in scripts/part1, which stay frozen as
the reference). A parity test (tests/test_region_parity.py) guards each copy against drift.

  region.py    — find title-block/table CELLS  +  build the vector-removed BODY raster (geometry)
  box_text.py  — vector: text per box cell           |  box_ocr.py   — OCR fallback per box cell
  body_text.py — vector: body (outside-cell) text     |  body_ocr.py  — OCR fallback on body raster
  bom.py       — BOM table finder + row parse

Routing (vector vs OCR by pymupdf word-count band) lives in the orchestrator, not here.
"""
