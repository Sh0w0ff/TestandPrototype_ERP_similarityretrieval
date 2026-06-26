"""e1_embed — A/B the text retrieval channel: TF-IDF bag-of-tokens (the current E1 baseline) vs a
neural sentence-embedding of the SAME signal text. Same leave-one-out loop, same ERP truth, so the
only thing that changes is the document representation -> this isolates "does a domain/general text
encoder beat bag-of-words on our templated, abbreviation-heavy drawing text?" (thesis G4).

The encoder is swappable with --model:
  sentence-transformers/all-MiniLM-L6-v2   general strong retriever (default; small, fast)
  m3rg-iitd/matscibert                     materials-science domain BERT (mean-pool; not sentence-tuned)
  allenai/scibert_scivocab_uncased         scientific BERT
  intfloat/e5-base-v2 / BAAI/bge-base-en   stronger general retrievers (need 'query:'/'passage:' prefix for e5)

Frozen-encoder caveat (mirrors the visual channel's frozen-DINO finding): a frozen text encoder may
UNDERPERFORM TF on this very templated text — measuring that gap is itself the contribution, and the
next step would be light adaptation (contrastive on ERP-sibling pairs), not a bigger frozen model.
"""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/sh0w0ff/FYP")
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import e1
from erp_truth import ErpTruth


def embed(docs, model_name, batch=16, prefix=""):
    """Mean-pooled, L2-normalized sentence embeddings via plain transformers (no sentence-transformers
    dep needed). prefix lets e5-style models get their required 'passage: ' tag."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(docs), batch):
            chunk = [prefix + (d or " ") for d in docs[i:i + batch]]
            enc = tok(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
            hid = mdl(**enc).last_hidden_state                       # (B,T,H)
            mask = enc["attention_mask"].unsqueeze(-1).float()       # (B,T,1)
            emb = (hid * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean-pool over real tokens
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.cpu().numpy())
    return np.vstack(out)


# The three structured views the encoder is tested on (each maps to fingerprint component toggles):
#   notes  = free-text prose (tag-row text + unclassified body)  -> where embeddings should win
#   fields = normalized structured tokens (typed fields + vocab tags) -> TF likely already optimal
#   all    = the full leak-safe fingerprint (notes + fields + residual cells)
#   prose  = natural-language rendering of the signal (fair baseline for sentence encoders)
VIEWS = {
    "notes":  dict(notes=True),
    "fields": dict(typed=True, vocab=True),
    "all":    e1.FP_DEFAULT,
}


# Material/process code → natural language (drawn from ERP raw-material descriptions).
# Only codes that appear verbatim in drawing text and have unambiguous ERP meaning.
_MAT_EXPAND = {
    "HR":    "hot-rolled",
    "CR":    "cold-rolled",
    "HDG":   "hot-dip galvanized",
    "GALV":  "hot-dip galvanized",
    "GALVANIZED": "hot-dip galvanized",
    "SS":    "stainless steel",
    "AL":    "aluminium",
    "LASER": "laser-cut",
    "PLASMA": "plasma-cut",
}
_COAT_EXPAND = {
    "RAL7021": "painted dark grey RAL7021",
    "RAL 7021": "painted dark grey RAL7021",
    "RAL1021": "painted yellow RAL1021",
    "RAL 1021": "painted yellow RAL1021",
    "RAL9002": "painted off-white RAL9002",
    "SINKKIPOHJA": "zinc primer",
    "GALV": "hot-dip galvanized",
}


def render_prose(sig: dict) -> str:
    """Render a signal as natural-language prose suitable for sentence encoders.
    ERP-grounded design: material grades, part types, coatings, tolerances and
    notes are expressed as the encoder's training distribution expects them,
    not as synthetic TF tokens (field_parttype_X etc.)."""
    cl = sig.get("classified", {})
    tf = cl.get("fields") or {}
    parts = []

    # --- identity sentence: part type + name ---
    name_toks = tf.get("part_name", [])
    type_toks = tf.get("part_type", [])
    if name_toks or type_toks:
        ident = " ".join(type_toks + name_toks).strip()
        parts.append(ident.capitalize() + ".")

    # --- material ---
    mats = tf.get("material", []) + tf.get("material_class", [])
    if mats:
        expanded = []
        for m in mats:
            up = m.upper()
            expanded.append(_MAT_EXPAND.get(up, m))
        parts.append("Material: " + ", ".join(expanded) + ".")

    # --- coating / surface ---
    coats = tf.get("coating", [])
    if coats:
        expanded = []
        for c in coats:
            up = c.upper().strip()
            expanded.append(_COAT_EXPAND.get(up, c))
        parts.append("Surface treatment: " + ", ".join(expanded) + ".")

    # --- tolerances + scale ---
    for gt in tf.get("general_tolerance", []):
        parts.append(f"General tolerance: {gt}.")
    for sc in tf.get("scale", []):
        parts.append(f"Scale {sc}.")

    # --- standards ---
    for st in tf.get("standards", []):
        fam, num = st.get("family", ""), st.get("number", "")
        if fam or num:
            parts.append(f"Standard: {fam}{num}.")

    # --- specification ---
    for sp in tf.get("specification", []):
        parts.append(sp)

    # --- tag-row full text (classified prose notes) ---
    for u in cl.get("units", []):
        if u.get("reason") == "tag":
            t = u.get("text", "").strip()
            if t:
                parts.append(t)

    # --- unclassified body (free-text notes) ---
    for ub in sig.get("unclassified", {}).get("body", []):
        t = (ub.get("notes") or [ub.get("text", "")])[0] if isinstance(ub.get("notes"), list) else ub.get("text", "")
        if t:
            parts.append(t.strip())

    return " ".join(parts) if parts else " "


def load_views(signal_name, limit):
    """Read each signal ONCE; render all views per stem. Returns (stems, {view: [docs]})."""
    import json
    all_views = {**VIEWS, "prose": None}   # prose rendered separately
    stems, docs = [], {v: [] for v in all_views}
    for d in sorted(e1.SIG.iterdir()):
        f = d / signal_name
        if not f.exists():
            continue
        try:
            sig = json.load(open(f))
        except Exception:
            continue
        stems.append(d.name)
        for v, cfg in VIEWS.items():
            docs[v].append(e1.fingerprint(sig, cfg))
        docs["prose"].append(render_prose(sig))
        if limit and len(stems) >= limit:
            break
    return stems, docs


def run(name, sim, stems, comp_truth, phase_truth, k, vote):
    print(f"\n##### {name} #####")
    cr, cb = e1.evaluate(stems, sim, comp_truth, k, vote)
    e1.report(f"BOM components  K={k} vote={vote}", cr, cb)
    pr, pb = e1.evaluate(stems, sim, phase_truth, k, vote)
    e1.report(f"Work-phases     K={k} vote={vote}", pr, pb)
    return cr[:, 2].mean(), pr[:, 2].mean()                          # BOM F1, phase F1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--vote", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--prefix", default="", help="e.g. 'passage: ' for e5 models")
    ap.add_argument("--signal-name", default="signal.json")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    stems, views = load_views(args.signal_name, args.limit)
    print(f"loaded {len(stems)} signals ({args.signal_name})  views={list(VIEWS)}")
    erp = ErpTruth()
    comp_truth = [erp.stem_to_components(s) for s in stems]
    phase_truth = [erp.stem_to_phases(s) for s in stems]
    print(f"  with BOM truth: {sum(1 for t in comp_truth if t)}  with phase truth: {sum(1 for t in phase_truth if t)}")

    head = {}
    # --- TF-IDF baseline on the FULL structured signal (identical to e1) ---
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-_]+\b",
                          min_df=args.min_df, sublinear_tf=True)
    Xtf = vec.fit_transform(views["all"])
    print(f"  TF vocab: {len(vec.vocabulary_)}  matrix: {Xtf.shape}")
    head["TF-IDF (all, baseline)"] = run("TF-IDF bag-of-tokens  [view=all]", cosine_similarity(Xtf),
                                         stems, comp_truth, phase_truth, args.k, args.vote)

    # --- neural embedding: TF views + prose (fair NL rendering) ---
    all_embed_views = list(VIEWS) + ["prose"]
    for v in all_embed_views:
        print(f"\nembedding [{args.model}]  view={v} ...")
        E = embed(views[v], args.model, prefix=args.prefix)
        head[f"emb {v}"] = run(f"embedding {args.model}  [view={v}]", E @ E.T,
                               stems, comp_truth, phase_truth, args.k, args.vote)

    tf_b, tf_p = head["TF-IDF (all, baseline)"]
    print("\n================ HEADLINE (BOM F1 / phase F1, delta vs TF-all) ================")
    for name, (b, p) in head.items():
        tag = "" if name.startswith("TF") else f"   (d {b - tf_b:+.3f} / {p - tf_p:+.3f})"
        print(f"  {name:26}: {b:.3f} / {p:.3f}{tag}")
    print("  ^ expect: neural lift concentrated in 'notes', ~0 in 'fields' -> motivates the G4 hybrid")


if __name__ == "__main__":
    main()
