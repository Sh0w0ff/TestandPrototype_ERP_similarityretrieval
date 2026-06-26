"""engine.py — low-level inference helpers used by the pipeline.

Given a drawing stem that is part of the corpus pool, the helpers produce:

  1. EXTRACTED CONTENT     classified title-block fields + internal (on-drawing) BOM table
  2. SIMILAR DRAWINGS      nearest neighbours under frozen DINOv2 mean-pool, plus an optional
                           self-attention pool encoder (pooled union)
  3. ERP PROPOSAL          work-phases (production lines) + BOM-component candidate pool, by
                           case-based retrieval over text-similar neighbours (text drives this)

It reuses the existing scripts rather than re-implementing them:
  - visual index + mean pool  : scripts/part1/iam_pool.load_index   (embeddings.npy + manifest.json)
  - text fingerprint          : scripts/eval/e1.fingerprint         (input signal only)
  - ERP labels                : scripts/eval/erp_truth.ErpTruth     (stem -> phases / components)
  - work-phase label space    : scripts/eval/phase_classify.load_prod_lines (production lines)

The corpus index/signals are read-only.
"""
from __future__ import annotations
import os, sys, json, collections
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
for p in (SCRIPTS, SCRIPTS / "eval", SCRIPTS / "part1", SCRIPTS / "textpipe"):
    sys.path.insert(0, str(p))

import paths
import iam_pool as IP
import e1
from erp_truth import ErpTruth


# ----------------------------------------------------------------------------- retrieval-pool split
_SPLIT = {}
def split_sets():
    """Canonical train/val/test stem partition (from the multimodal split file). Used by the pool
    lever: restrict retrieval to TRAIN for honest held-out evaluation, or use ALL for a new file."""
    if not _SPLIT:
        f = paths.VISUAL_PIPE / "adapted" / "split.json"
        s = json.load(open(f)) if f.exists() else {}
        _SPLIT.update(train=set(s.get("train", [])), val=set(s.get("val", [])),
                      test=set(s.get("test", [])))
    return _SPLIT


def _mask(sims, stems, allowed):
    """Set similarity to -inf for stems outside `allowed` (None = no restriction)."""
    if allowed is not None:
        for j, s in enumerate(stems):
            if s not in allowed:
                sims[j] = -2.0
    return sims


# ----------------------------------------------------------------------------- visual retrieval
def _meanpool_index():
    """Build {stem: L2-normalised mean-pooled DINOv2 vector} from the prebuilt corpus index.
    This is the frozen mean-pool vector — no training, no adaptation."""
    E, by = IP.load_index()
    stems, vecs = [], []
    for s, lst in by.items():
        v = E[[i for i, *_ in lst]].mean(0)
        v = v / (np.linalg.norm(v) + 1e-9)
        stems.append(s); vecs.append(v.astype(np.float32))
    return stems, np.stack(vecs)


def visual_neighbours(query_stem, k=5, allowed=None):
    """Top-k visually-similar drawings (frozen mean-pool cosine kNN), excluding self.
    `allowed` restricts the candidate pool (e.g. the train split). Frozen mean-pool is the default."""
    stems, M = _meanpool_index()
    if query_stem not in stems:
        return None
    qi = stems.index(query_stem)
    sims = M @ M[qi]; sims[qi] = -2.0
    sims = _mask(sims, stems, allowed)
    top = np.argsort(-sims)[:k]
    return [(stems[j], float(sims[j])) for j in top if sims[j] > -2.0]


# --- second encoder: self-attention pool (cached per-stem embeddings) ---------
_SA_CACHE = {}
def _selfattn_index():
    """{stem: self-attn pooled vector} from the prebuilt cache (part-type-eligible subset)."""
    if "stems" not in _SA_CACHE:
        f = paths.VISUAL_PIPE / "adapted" / "iam_selfattn_emb.npz"
        if not f.exists():
            _SA_CACHE.update(stems=[], M=np.zeros((0, 384), np.float32), pos={})
        else:
            z = np.load(f, allow_pickle=True)
            stems = list(z["stems"]); M = z["M"].astype(np.float32)
            M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
            _SA_CACHE.update(stems=stems, M=M, pos={s: i for i, s in enumerate(stems)})
    return _SA_CACHE


def selfattn_neighbours(query_stem, k=5, allowed=None):
    """Top-k via the self-attention pool encoder. Returns None if the stem isn't cached.
    `allowed` restricts the candidate pool."""
    SA = _selfattn_index()
    if query_stem not in SA["pos"]:
        return None
    qi = SA["pos"][query_stem]
    sims = SA["M"] @ SA["M"][qi]; sims[qi] = -2.0
    sims = _mask(sims, SA["stems"], allowed)
    top = np.argsort(-sims)[:k]
    return [(SA["stems"][j], float(sims[j])) for j in top if sims[j] > -2.0]


def union_neighbours(query_stem, k_each=5, allowed=None):
    """POOLED retrieval: top-`k_each` from EACH encoder (frozen mean-pool +
    self-attn), unioned and deduped -> up to 2*k_each candidates. Recall-oriented; each candidate
    is tagged by which encoder(s) surfaced it. Falls back to frozen-only when self-attn lacks the stem."""
    mean = visual_neighbours(query_stem, k_each, allowed) or []
    sa = selfattn_neighbours(query_stem, k_each, allowed)
    merged = {}
    for s, sim in mean:
        merged[s] = dict(stem=s, mean=sim, selfattn=None)
    if sa:
        for s, sim in sa:
            merged.setdefault(s, dict(stem=s, mean=None, selfattn=None))["selfattn"] = sim
    out = []
    for s, d in merged.items():
        src = "both" if d["mean"] is not None and d["selfattn"] is not None else \
              ("mean" if d["mean"] is not None else "selfattn")
        score = max(v for v in (d["mean"], d["selfattn"]) if v is not None)
        out.append(dict(stem=s, source=src, mean=d["mean"], selfattn=d["selfattn"], score=score))
    out.sort(key=lambda r: (-(r["source"] == "both"), -r["score"]))   # agreed-by-both first, then score
    return out


# ----------------------------------------------------------------------------- text retrieval (ERP)
_TEXT_CACHE = {}

def _text_index():
    """Corpus text fingerprints (input-signal only) + TF-IDF cosine matrix builder. Cached."""
    if "stems" not in _TEXT_CACHE:
        stems, docs = e1.load()                       # signal.json -> fingerprint, whole corpus
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = vec.fit_transform(docs)                    # already L2-normalised rows
        _TEXT_CACHE.update(stems=stems, X=X, vec=vec, pos={s: i for i, s in enumerate(stems)})
    return _TEXT_CACHE


def text_neighbours(query_stem, k=5, allowed=None):
    """Top-k text-similar drawings (TF-IDF cosine) — the primary ERP-proposal driver.
    `allowed` restricts the candidate pool (e.g. the train split)."""
    T = _text_index()
    if query_stem not in T["pos"]:
        return None
    qi = T["pos"][query_stem]
    sims = (T["X"] @ T["X"][qi].T).toarray().ravel(); sims[qi] = -2.0
    sims = _mask(sims, T["stems"], allowed)
    top = np.argsort(-sims)[:k]
    return [(T["stems"][j], float(sims[j])) for j in top if sims[j] > -2.0]


# ----------------------------------------------------------------------------- ERP proposal (CBR)
_ERP = None
def _erp():
    global _ERP
    if _ERP is None:
        _ERP = ErpTruth()
    return _ERP

_PL = None
def _prod_lines():
    """stem -> set of production-line codes (the production-line work-phase target)."""
    global _PL
    if _PL is None:
        import phase_classify as PC
        _PL = PC.load_prod_lines()
    return _PL


def propose_erp(query_stem, neighbours, vote=1):
    """Case-based proposal: aggregate the ERP truth of the text-similar NEIGHBOURS.
    - work-phases : production-line codes shared by >= `vote` neighbours
    - BOM pool    : component item-IDs shared by >= `vote` neighbours (candidate pool)
    Returns dict with proposals + the per-candidate neighbour support count."""
    erp = _erp(); pl = _prod_lines()
    nb = [s for s, _ in neighbours]

    phase_votes = collections.Counter(p for s in nb for p in pl.get(s, set()))
    bom_votes   = collections.Counter(c for s in nb for c in erp.stem_to_components(s))

    phases = sorted((p for p, n in phase_votes.items() if n >= vote), key=lambda p: -phase_votes[p])
    boms   = sorted((c for c, n in bom_votes.items() if n >= vote), key=lambda c: -bom_votes[c])
    return dict(
        phases=[(p, phase_votes[p]) for p in phases],
        bom=[(c, bom_votes[c]) for c in boms],
        n_neighbours=len(nb),
    )


# ----------------------------------------------------------------------------- extracted content
def extract_content(query_stem):
    """Surface the drawing's OWN extracted content from the prebuilt signal.json (F2):
    classified title-block fields + the internal on-drawing BOM table. No ERP join here."""
    f = paths.TEXT_PIPE / query_stem / "signal.json"
    if not f.exists():
        return None
    sig = json.load(open(f))
    cl = sig.get("classified", {})
    fields = cl.get("fields", {}) or {}
    bom_cells = [c.get("text", "") for c in sig.get("bom", {}).get("cells", [])]
    return dict(fields=fields, internal_bom=bom_cells)


# ----------------------------------------------------------------------------- ground-truth (verify)
def ground_truth(query_stem):
    """The query's OWN ERP truth — for verification only (never an input feature)."""
    erp = _erp(); pl = _prod_lines()
    return dict(phases=sorted(pl.get(query_stem, set())),
                components=sorted(erp.stem_to_components(query_stem)),
                family=erp.stem_to_family(query_stem))


# ----------------------------------------------------------------------------- one-shot run
def run(query_stem, k=5):
    """Full inference for one corpus drawing. Returns a result dict (UI/render consume it)."""
    return dict(
        stem=query_stem,
        visual=visual_neighbours(query_stem, k),
        text=text_neighbours(query_stem, k),
        content=extract_content(query_stem),
        truth=ground_truth(query_stem),
    )
