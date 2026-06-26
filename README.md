# Drawing to ERP prototype

Takes a 2D technical drawing (PDF) and does three things:

- extracts its content (title block fields and the internal BOM table)
- retrieves visually similar drawings from the archive
- proposes ERP data: work phases (production routing) and a BOM candidate pool

It works on drawings from any vendor with no per vendor tuning. Built on the existing ABB and
Konecranes corpus. This is a proof of concept, not a production system.

## How it works

The text channel produces `signal.json` (fields, internal BOM, and a search fingerprint).

The visual channel runs SAM view segmentation, embeds each view with a frozen DINOv2 encoder, and
pools the views into one vector (a mean pool, plus an optional self attention pool).

Given a drawing, the prototype then:

- finds similar drawings by combining the top results from both visual encoders into one pooled set,
  tagged by which encoder found each one
- predicts work phases with a trained classifier (word and character TF-IDF into a ResMLP-BN). It
  reaches about 0.91 macro F1 and 0.60 exact match on this corpus.
- builds a BOM candidate pool by voting over the ERP BOMs of the text similar neighbours. This is a
  shortlist for a human to pick from, not a full prediction.

The corpus index and signals are read only. New artifacts for a new PDF go to `scratch/`.

## Install

```bash
pip install -r requirements.txt
```

Packages: numpy, scipy, scikit-learn, torch, torchvision, transformers, segment-anything, pymupdf,
opencv-python, Pillow, shapely, streamlit. DINOv2-small downloads from Hugging Face on first run.
The new PDF path needs the SAM checkpoint at `models/sam_vit_h_4b8939.pth`.

Data is read from this repo by default. Point it elsewhere with `FYP_DATA_ROOT=/path/to/data`.

## Use

```bash
python proto.py make-database            # 1. build the search index
python proto.py train                    # 2. train both work phase models (a few min)
python proto.py infer 3AUA0000038918 --mode test     # 3a. evaluate a held out corpus drawing
python proto.py infer /path/new.pdf --mode deploy    # 3b. run on a genuinely new PDF
```

### Run the app

Activate the environment where you installed the requirements, then:

```bash
streamlit run app.py
```

Streamlit prints a local URL and opens your browser at it (by default `http://localhost:8501`).
In the sidebar, pick a corpus drawing or upload a new PDF, choose Test or Deploy, and press Run.
Stop the app with Ctrl-C in the terminal.

### Test vs Deploy

The `mode` setting picks a whole configuration at once: which trained model, which drawings are
searched, and whether the correct answer is shown.

| mode | model used | retrieval pool | ground truth |
|---|---|---|---|
| test | `eval` (trained on the train split only) | train split only | shown and scored |
| deploy | `deploy` (trained on all data) | whole corpus | hidden |

Use **test** to evaluate a held out drawing honestly. The model never saw it and the neighbours come
only from training data, so there is no leakage. This is what reproduces the reported accuracy.

Use **deploy** for a real new drawing, where every drawing is fair to use for both the model and the
retrieval pool.

`train` builds both models (`--scope eval|deploy|both`, default both). The retrieval pool can be set
on its own with `--pool all|train` (default `auto`, which follows the mode).

As a library:

```python
from prototype.pipeline import make_database, train, infer
make_database(); train()
r = infer("3AUA0000038918", mode="test")
```

## Layout

```
proto.py            CLI (make-database, train, infer)
app.py              Streamlit UI
prototype/          pipeline.py (API), engine.py (retrieval), workphase.py (classifier)
scripts/            research scripts reused as is (about 22 on the runtime path, rest for reproducibility)
```

## Required data (not in the repo)

The drawing data and ERP tables are proprietary and are not shipped with the code. The folders below
are committed empty (placeholders only). Before running anything, supply the data locally (copy the
data pool in, or point `FYP_DATA_ROOT` at it), then run `make-database` and `train`.

You must provide:

- The four ERP tables at the repo root: `Item_Basic_Data.csv`, `Bill_of_Materials.csv`,
  `Work_Phases.csv`, `Work_Center_Basic_Data.csv`.
- The SAM checkpoint `models/sam_vit_h_4b8939.pth` (only needed for the new PDF path).
- The drawing data pool (`PDF drawings/`, `text_pipe/`, `visual_pipe/`), or run the pipeline to
  build it from the PDFs.

Folder map:

```
*.csv                ERP tables (Item, BOM, Work_Phases, Work_Center) — required, supply locally
PDF drawings/        source PDFs
text_pipe/<stem>/    signal.json (extracted text per drawing)
visual_pipe/index/   embeddings.npy and manifest.json (the retrieval index)
visual_pipe/<stem>/  per drawing view segmentation and renders
models/              sam_vit_h_4b8939.pth (new PDF view segmentation)
cache/               built database and trained models (generated)
scratch/             new PDF artifacts (generated)
```

## Limits

- The visual channel reads page 1 only.
- The BOM is long tailed (about 79% one offs), so the pool is a shortlist and is scored only on the
  recurring components.
- Looking alike does not mean a shared product structure. A match means "look here first", not
  "same part".
- The new PDF path runs SAM and DINOv2 live and is slower. Corpus drawings use cached artifacts and
  are instant.
