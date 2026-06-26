DRAWING TO ERP PROTOTYPE
========================

Takes a 2D technical drawing (PDF) and does three things:

  - extracts its content (title block fields and the internal BOM table)
  - retrieves visually similar drawings from the archive
  - proposes ERP data: work phases (production routing) and a BOM candidate pool

It works on drawings from any vendor with no per vendor tuning. Built on the existing ABB and
Konecranes corpus. This is a proof of concept, not a production system.


RESULTS
-------

  Work phase prediction        macro F1 0.91, exact 0.60, micro 0.93. The strongest result and the
                               easiest to put into use. Works the same on both vendors (ABB 0.93, KC 0.90).
  Similar drawing retrieval    Success@5 0.84, P@1 0.69. Frozen DINOv2 mean pool, judged by
                               people. The simple model beat the adapted one.
  BOM candidate pool           recall 0.74 to 0.91 on recurring components. A shortlist, not a
                               full BOM (see below).
  Content extraction           title block fields and on drawing BOM read reliably. Useful alone.

What is fundamentally limited:

  - The BOM cannot be generated, only retrieved. About 79% of components are one offs that never
    repeat, so the tool shows a candidate pool for a human to finish.
  - BOM reuse is vendor specific. Konecranes scores 0.93, ABB scores 0.46. Report per vendor.
  - Looking alike does not mean the same structure. A visual match means search here first, not
    same part.
  - Text leads the prediction. Visual leads the similarity search. Combining them adds very little.

Dataset facts: about 99.95% of drawings are linked to their ERP record, about 94.2% are top level
assemblies, about 52.1% are multi page.


HOW IT WORKS
------------

The text channel produces signal.json (fields, internal BOM, and a text feature vector for search).

The visual channel runs SAM view segmentation, embeds each view with a frozen DINOv2 encoder, and
pools the views into one vector (a mean pool, plus an optional self attention pool).

Given a drawing, the prototype then:

  - finds similar drawings by combining the top results from both visual encoders into one pooled
    set, tagged by which encoder found each one
  - predicts work phases with a trained classifier (word and character TF-IDF into a ResMLP-BN)
  - builds a BOM candidate pool by voting over the ERP BOMs of the text similar neighbours

The corpus index and signals are read only. New files for a new PDF go to scratch/.


INSTALL
-------

  pip install -r requirements.txt

Packages: numpy, scipy, scikit-learn, torch, torchvision, transformers, segment-anything, pymupdf,
opencv-python, Pillow, shapely, streamlit. DINOv2-small downloads from Hugging Face on first run.
The new PDF path needs the SAM checkpoint at models/sam_vit_h_4b8939.pth.

Data is read from this repo by default. Point it elsewhere with FYP_DATA_ROOT=/path/to/data.


USE
---

  python proto.py make-database          1. build the search index
  python proto.py train                  2. train both work phase models (a few min)
  python proto.py infer 3AUA0000038918 --mode test     3a. evaluate a held out corpus drawing
  python proto.py infer /path/new.pdf --mode deploy    3b. run on a genuinely new PDF

Test vs Deploy. The mode setting picks a whole configuration at once: which trained model, which
drawings are searched, and whether the correct answer is shown.

  test     model trained on the train split only, retrieval from the train split only, ground
           truth shown and scored. Use this to evaluate a held out drawing fairly: the model
           never saw it and the neighbours come only from training data, so there is no leakage.
           This is what reproduces the reported accuracy.
  deploy   model trained on all data, retrieval from the whole corpus, no ground truth. Use this
           for a real new drawing, where every drawing is fair to use.

train builds both models (--scope eval|deploy|both, default both). The retrieval pool can be set on
its own with --pool all|train (default auto, which follows the mode).

As a library:

  from prototype.pipeline import make_database, train, infer
  make_database(); train()
  r = infer("3AUA0000038918", mode="test")


RUN THE APP
-----------

Two things must be right, or the app will not start:

  1. Run it from inside the repo folder, where app.py lives. Otherwise streamlit reports
     "File does not exist: app.py".
  2. Use the same environment that has the requirements installed (the one with torch and
     streamlit together). Otherwise you get "ModuleNotFoundError: No module named 'torch'"
     because streamlit was launched from a different environment.

  cd /path/to/this/repo            # the folder that contains app.py
  conda activate <your-env>        # the environment where you ran pip install -r requirements.txt
  streamlit run app.py

If you are unsure which streamlit you are using, run "which streamlit". It must point inside that
same environment, not your base install.

Streamlit prints a local URL and opens your browser at it (by default http://localhost:8501).
In the sidebar, pick a corpus drawing or upload a new PDF, choose Test or Deploy, and press Run.
Stop the app with Ctrl-C in the terminal.


LAYOUT
------

  proto.py            CLI (make-database, train, infer)
  app.py              Streamlit UI
  prototype/          pipeline.py (API), engine.py (retrieval), workphase.py (classifier)
  scripts/            research scripts reused as is (about 22 on the runtime path, rest for
                      reproducibility)


REQUIRED DATA (not in the repo)
-------------------------------

The drawing data and ERP tables are proprietary and are not shipped with the code. The folders below
are committed empty (placeholders only). Before running anything, supply the data locally (copy the
data pool in, or point FYP_DATA_ROOT at it), then run make-database and train.

You must provide:

  - The four ERP tables at the repo root: Item_Basic_Data.csv, Bill_of_Materials.csv,
    Work_Phases.csv, Work_Center_Basic_Data.csv.
  - The SAM checkpoint models/sam_vit_h_4b8939.pth (only needed for the new PDF path).
  - The drawing data pool (PDF drawings/, text_pipe/, visual_pipe/), or run the pipeline to build
    it from the PDFs.

Folder map:

  *.csv                ERP tables (required, supply locally)
  PDF drawings/        source PDFs
  text_pipe/<stem>/    signal.json (extracted text per drawing)
  visual_pipe/index/   embeddings.npy and manifest.json (the retrieval index)
  visual_pipe/<stem>/  per drawing view segmentation and renders
  models/              sam_vit_h_4b8939.pth (new PDF view segmentation)
  cache/               built database and trained models (generated)
  scratch/             new PDF files (generated)


LIMITS
------

  - The visual channel reads page 1 only.
  - The BOM is long tailed (about 79% one offs), so the pool is a shortlist and is scored only on
    the recurring components.
  - Looking alike does not mean a shared product structure. A match means look here first, not
    same part.
  - The new PDF path runs SAM and DINOv2 live and is slower. Corpus drawings use cached files
    and are instant.
