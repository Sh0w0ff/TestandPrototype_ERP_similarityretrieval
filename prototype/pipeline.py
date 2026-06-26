"""pipeline.py — the prototype's three-function public API.

    1. make_database()      assemble + cache the corpus retrieval database from the existing data pool
                            (frozen DINOv2 mean-pool vectors + text fingerprints + ERP truth)
    2. train()              train + cache the work-phase classifier on that database (ResMLP-BN)
    3. infer(drawing, mode) run on a drawing: extract content + retrieve k=5 similar + propose ERP
                            - corpus stem  -> fast path, reuse cached artifacts (test-set workflow)
                            - new PDF      -> full extract->embed->retrieve, written to SCRATCH only

The corpus index/signals are READ, never mutated. New per-drawing artifacts (for genuinely new PDFs)
go to additive scratch dirs; the shared embeddings.npy is never rewritten.

Modes:  "test"  -> include ground-truth + correctness (you have an answer key)
        "deploy"-> proposal only (a real new drawing has no answer key)
"""
from __future__ import annotations
import os, sys, json, subprocess, tempfile
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
for p in (SCRIPTS, SCRIPTS / "eval", SCRIPTS / "part1", SCRIPTS / "textpipe", SCRIPTS / "vispipe"):
    sys.path.insert(0, str(p))

import paths
from . import engine as E
from . import workphase as WP

CACHE = REPO / "cache"; CACHE.mkdir(exist_ok=True)
MEANPOOL = CACHE / "meanpool_index.npz"
MANIFEST = CACHE / "database.json"
SCRATCH = REPO / "scratch"; SCRATCH.mkdir(exist_ok=True)
PY = sys.executable


# ============================================================ 1. MAKE DATABASE
def make_database(rebuild=False, verbose=True):
    """Assemble the retrieval database from the existing corpus pool and cache it.
    Builds: frozen DINOv2 MEAN-POOL vector per drawing + a record of
    which drawings have text signals / ERP truth. Does NOT recompute the heavy per-drawing pipeline —
    it reuses visual_pipe/index/embeddings.npy + text_pipe/*/signal.json already on disk."""
    if MEANPOOL.exists() and not rebuild:
        z = np.load(MEANPOOL, allow_pickle=True)
        if verbose:
            print(f"database: cached mean-pool index ({len(z['stems'])} drawings) — use rebuild=True to refresh")
        return _db_summary()

    # required corpus artifacts
    if not paths.VISUAL_INDEX.joinpath("embeddings.npy").exists():
        raise FileNotFoundError(f"corpus visual index missing at {paths.VISUAL_INDEX} — "
                                "point FYP_DATA_ROOT at the data pool.")
    stems, M = E._meanpool_index()                        # frozen mean-pool
    np.savez(MEANPOOL, stems=np.array(stems), M=M)

    n_text = sum((paths.TEXT_PIPE / s / "signal.json").exists() for s in stems)
    manifest = dict(n_drawings=len(stems), n_with_text=n_text,
                    visual_index=str(paths.VISUAL_INDEX), data_root=str(paths.DATA_ROOT))
    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    if verbose:
        print(f"database built: {len(stems)} drawings indexed (mean-pool), {n_text} with text signals")
        print(f"  -> {MEANPOOL}")
    return _db_summary()


def _db_summary():
    z = np.load(MEANPOOL, allow_pickle=True)
    return dict(stems=list(z["stems"]), n=len(z["stems"]),
                trained=WP.is_trained("eval"), trained_deploy=WP.is_trained("deploy"),
                manifest=json.load(open(MANIFEST)) if MANIFEST.exists() else {})


def _load_meanpool():
    z = np.load(MEANPOOL, allow_pickle=True)
    return list(z["stems"]), z["M"]


# ============================================================ 2. TRAIN
def train(epochs=WP.EPOCHS, scope="both", verbose=True):
    """Train + cache the work-phase classifier(s) on the database.
      scope='eval'   -> train-split model, reports honest held-out accuracy (used in test mode)
      scope='deploy' -> all-data model, for genuinely new drawings (used in deploy mode)
      scope='both'   -> train both (default)
    BOM proposal needs no training (it is aggregate voting over retrieved neighbours)."""
    if not MEANPOOL.exists():
        make_database(verbose=verbose)
    scopes = ["eval", "deploy"] if scope == "both" else [scope]
    out = {}
    for s in scopes:
        out[s] = WP.train_and_cache(scope=s, epochs=epochs, verbose=verbose)
    return out


# ============================================================ 3. INFER
def infer(drawing, k=5, mode="test", vote=1, pool=None):
    """Run the prototype on one drawing. `drawing` = a corpus stem OR a path to a PDF.

    `mode` is the master switch — it picks a coherent configuration:
      "test"   -> eval model (train-split) + train-split retrieval pool + show ground truth/scoring.
                  Use this to evaluate a held-out test drawing honestly (no leakage).
      "deploy" -> deploy model (all-data) + whole-corpus retrieval pool + proposal only.
                  Use this for a genuinely new drawing.
    `pool` ("all"/"train") optionally OVERRIDES the mode default if you want to mix them.
    Returns a result dict consumed by the UI / renderer."""
    pdf = Path(drawing)
    is_pdf = pdf.suffix.lower() == ".pdf" and pdf.exists()
    stem = pdf.stem if is_pdf else str(drawing)

    # mode sets the model scope + the default retrieval pool; `pool` can override the pool.
    scope = "eval" if mode == "test" else "deploy"
    if not WP.is_trained(scope):
        scope = "eval" if WP.is_trained("eval") else scope        # fall back if deploy not built
    if pool is None:
        pool = "train" if mode == "test" else "all"
    allowed = E.split_sets()["train"] if pool == "train" else None

    stems, M = _load_meanpool()
    in_corpus = stem in stems

    if in_corpus:
        qvec = M[stems.index(stem)]
        sig = _load_signal(stem)
    elif is_pdf:
        qvec, sig = _process_new_pdf(pdf)
    else:
        raise ValueError(f"'{drawing}' is neither a corpus stem nor an existing PDF path.")

    # ---- visual neighbours: POOLED union, top-k from EACH encoder (frozen mean-pool + self-attn) ----
    if in_corpus:
        visual = E.union_neighbours(stem, k_each=k, allowed=allowed)   # up to 2k, source-tagged
    else:
        # new PDF: self-attn pool weights aren't available at inference -> frozen mean-pool only
        sims = E._mask(M @ qvec, stems, allowed)
        vtop = np.argsort(-sims)[:k]
        visual = [dict(stem=stems[j], source="mean", mean=float(sims[j]),
                       selfattn=None, score=float(sims[j])) for j in vtop if sims[j] > -2.0]

    # ---- text-similar neighbours (drive the ERP proposal; text is primary) ----
    text = E.text_neighbours(stem, k, allowed=allowed) if in_corpus else _text_neighbours_newdoc(sig, k, allowed)

    # ---- ERP proposal: work-phase CLASSIFIER  + BOM aggregate voting  ----
    doc = E.e1.fingerprint(sig) if sig else ""
    phases_clf = WP.predict(doc, scope=scope) if (sig and WP.is_trained(scope)) else []
    bom_pool = E.propose_erp(stem, text, vote=vote)["bom"] if text else []

    result = dict(
        stem=stem, mode=mode, pool=pool, scope=scope, in_corpus=in_corpus,
        visual=visual,
        content=E.extract_content(stem) if in_corpus else _content_from_sig(sig),
        workphases=phases_clf,                  # [(label, prob)] from the trained classifier
        bom_pool=bom_pool,                      # [(component_id, n_votes)]
        text_neighbours=text,
    )
    if mode == "test" and in_corpus:
        result["truth"] = E.ground_truth(stem)
        result["scoring"] = _score(result)
    return result


# ---------------------------------------------------------------- helpers
def _load_signal(stem):
    f = paths.TEXT_PIPE / stem / "signal.json"
    return json.load(open(f)) if f.exists() else None


def _content_from_sig(sig):
    if not sig:
        return None
    cl = sig.get("classified", {})
    return dict(fields=cl.get("fields", {}) or {},
                internal_bom=[c.get("text", "") for c in sig.get("bom", {}).get("cells", [])])


def _score(result):
    """Test-mode correctness vs the query's own ERP truth."""
    truth = result["truth"]
    pred_ph = {p for p, _ in result["workphases"]}
    true_ph = set(truth["phases"])
    inter = pred_ph & true_ph
    p = len(inter) / len(pred_ph) if pred_ph else 0.0
    r = len(inter) / len(true_ph) if true_ph else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    bom_pool = {c for c, _ in result["bom_pool"]}
    true_bom = set(truth["components"])
    bom_r = len(bom_pool & true_bom) / len(true_bom) if true_bom else None
    return dict(phase_precision=p, phase_recall=r, phase_f1=f1,
                phase_exact=(pred_ph == true_ph), bom_pool_recall=bom_r)


# ---------------------------------------------------------------- new-PDF path (best-effort, scratch)
def _process_new_pdf(pdf: Path):
    """Run the full per-drawing pipeline on a NEW pdf into scratch, return (mean-pool vec, signal).
    Heavy: SAM view-seg + DINOv2 embed + text extraction. Corpus index is never modified."""
    stem = pdf.stem
    # 1) TEXT — build_signal writes text_pipe/<stem>/signal.json (page-1, additive)
    env = dict(os.environ, FYP_PDF_DIR=str(pdf.parent))
    subprocess.run([PY, str(SCRIPTS / "textpipe" / "build_signal.py"), stem], env=env, check=False)
    sig = _load_signal(stem)

    # 2) VISUAL — render super + SAM + v5 view selection (build_views orchestrator, page-1)
    sample = SCRATCH / f"{stem}.sample"; sample.write_text(stem + "\n")
    subprocess.run([PY, str(SCRIPTS / "vispipe" / "build_views.py"), "--sample", str(sample)],
                   env=env, check=False)

    # 3) EMBED the new stem's views IN MEMORY (do NOT write to the shared index)
    qvec = _embed_stem_meanpool(stem)
    return qvec, sig


def _embed_stem_meanpool(stem):
    """Frozen DINOv2 mean-pool vector for one stem's v5 views — in memory, index untouched."""
    import embed_all as EA
    import cv2
    sd = paths.VISUAL_PIPE / stem / "sam_views"
    vf = sd / "sam_masks_filtered_v5.json"; mf = sd / "sam_masks.json"
    sup_png = paths.VISUAL_PIPE / stem / "polygon_removed_super.png"
    if not (vf.exists() and mf.exists() and sup_png.exists()):
        raise RuntimeError(f"visual views missing for {stem} — view-seg may have failed.")
    H, W = json.loads(mf.read_text())["image_hw"]
    views = json.loads(vf.read_text()).get("views", [])
    sup = cv2.imread(str(sup_png))
    if sup.shape[:2] != (H, W):
        sup = cv2.resize(sup, (W, H))
    sup_rgb = cv2.cvtColor(sup, cv2.COLOR_BGR2RGB)
    model, tf = EA.load_dino()
    crops = []
    for v in views:
        x, y, w, h = v["bbox_xywh"]
        crop = sup_rgb[max(0, y):y + h, max(0, x):x + w]
        if crop.size:
            crops.append(EA.preprocess(crop))
    if not crops:
        raise RuntimeError(f"no view crops for {stem}.")
    vecs = EA.embed_batch(model, tf, crops)      # (n,384) L2-normalised
    v = vecs.mean(0); return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


def _text_neighbours_newdoc(sig, k, allowed=None):
    """Top-k text-similar corpus drawings for a NEW drawing's signal (TF-IDF cosine)."""
    if not sig:
        return []
    T = E._text_index()
    q = T["vec"].transform([E.e1.fingerprint(sig)])
    sims = E._mask((T["X"] @ q.T).toarray().ravel(), T["stems"], allowed)
    top = np.argsort(-sims)[:k]
    return [(T["stems"][j], float(sims[j])) for j in top if sims[j] > -2.0]
