"""Central path config — the SINGLE source of truth for where every artifact lives.

Top-level layout (siblings under the repo root):

  PDF drawings/        source PDFs (untouched)
  preprocess/<stem>/   page1.png  page2.png  text.txt  meta.json   (pages + required text only)
  text_pipe/<stem>/    signal.json                                 textual-channel outputs
  text_pipe/images/<stem>/                                         viewable text renders (signal_dump)
  visual_pipe/<stem>/  sam_masks*.json  polygon_removed_super*.png  crops/   per-drawing visual outputs
  visual_pipe/index/   embeddings.npy  manifest.json               corpus-level visual index
  visual_pipe/images/<stem>/                                       viewable view-seg/embed renders
  total_pipe/          sim matrices, retrieval results             cross-channel (eval/fusion) on top
  total_pipe/images/                                               viewable retrieval sheets
  testing/<test>/files/   testing/<test>/images/                   anything a test run produces

Every script imports this instead of hardcoding `cache/...`. Change a root here -> everything follows.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # scripts/ is one level under the repo root

# DATA_ROOT = where the prebuilt corpus artifacts live (the existing FYP data pool).
# The prototype reuses the already-built index/signals/CSVs instead of recomputing them,
# so by default this points at the original FYP working dir. Override with $FYP_DATA_ROOT
# to relocate the data pool (e.g. on another machine).  CODE lives under ROOT (this repo);
# DATA lives under DATA_ROOT (may be the same dir or a sibling).
DATA_ROOT = Path(os.environ["FYP_DATA_ROOT"]).expanduser() if os.environ.get("FYP_DATA_ROOT") else Path("/Users/sh0w0ff/FYP")

# PDFs default to DATA_ROOT/"PDF drawings" but can live anywhere via $FYP_PDF_DIR.
PDF_DIR     = Path(os.environ["FYP_PDF_DIR"]).expanduser() if os.environ.get("FYP_PDF_DIR") else DATA_ROOT / "PDF drawings"
PREPROCESS  = DATA_ROOT / "preprocess"
TEXT_PIPE   = DATA_ROOT / "text_pipe"
VISUAL_PIPE = DATA_ROOT / "visual_pipe"
TOTAL_PIPE  = DATA_ROOT / "total_pipe"
TESTING     = ROOT / "testing"

VISUAL_INDEX = VISUAL_PIPE / "index"            # one embeddings.npy + manifest.json for the whole corpus


# ---- per-stem data dirs (pipe -> stem -> files) ----
def preprocess_dir(stem): return PREPROCESS / stem
def text_dir(stem):       return TEXT_PIPE / stem
def visual_dir(stem):     return VISUAL_PIPE / stem


# ---- viewable images: <pipe>/images[/<stem>] ----
def _images(root, stem=None):
    d = root / "images"
    return d / stem if stem else d

def text_images(stem=None):    return _images(TEXT_PIPE, stem)
def visual_images(stem=None):  return _images(VISUAL_PIPE, stem)
def total_images(stem=None):   return _images(TOTAL_PIPE, stem)


# ---- testing sandbox: testing/<test>/{files,images} ----
def testing_files(test):  return TESTING / test / "files"
def testing_images(test): return TESTING / test / "images"


def ensure(*dirs):
    """mkdir -p for any number of dirs; returns the first (convenience for `p = ensure(d)`)."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    return Path(dirs[0]) if dirs else None
