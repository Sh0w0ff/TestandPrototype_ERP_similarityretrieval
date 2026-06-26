"""e1 — leave-one-out retrieval-prediction harness (gates G1-G4).

Pipeline per held-out drawing:
  signal.json -> TEXT FINGERPRINT (input signal only; no target, no dims) -> similarity ->
  top-K neighbours -> aggregate their ERP truth (erp_truth) -> score vs the held-out drawing's truth.

CHANNEL-AGNOSTIC by design: retrieval consumes a similarity MATRIX. Today that matrix is text
(TF-IDF cosine); when the visual channel resumes, build a visual similarity matrix the same shape
and FUSE (weighted sum) — fusion then becomes an ablation (text vs visual vs hybrid), per G4.

Targets (erp_truth): BOM component set (G2), work-phase set (G3). Metric = set-overlap
(precision/recall/F1/Jaccard) averaged over queries with non-empty truth, vs a random-K baseline
(the 4.9% BOM / 26.2% phase background-sharing floors from the premise-check).

Run:  python scripts/eval/e1.py [--k 5] [--limit N]
"""
import sys, json, argparse, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]     # repo root (use the repo's own script copies)
sys.path.insert(0, str(ROOT / "scripts"))
import paths
SIG = paths.TEXT_PIPE                          # <stem>/signal.json
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from erp_truth import ErpTruth


# ---- TEXT FINGERPRINT: input signal only (no dims, no bom_cells/rows=TARGET, no noise/zone) ----
# Components are individually toggleable so each lever can be ABLATED. `bom_leak` is a LEAKAGE
# probe ONLY (folds the embedded-BOM TARGET back into the input to read the cheating ceiling) —
# never a legitimate config; off by default.
FP_DEFAULT = dict(cells=True, notes=True, vocab=True, typed=True,
                  include_noise=False, bom_leak=False)


def _notes_or_text(rec):
    return rec.get("notes") or ([rec["text"]] if rec.get("text") else [])


def fingerprint(sig, cfg=FP_DEFAULT):
    parts = []
    cl = sig.get("classified", {})
    un = sig.get("unclassified", {})
    tf = cl.get("fields") or {}
    if cfg.get("cells"):                                                # unclassified title-block cells
        parts += [b.get("text", "") for b in un.get("blocks", [])]
    if cfg.get("notes"):                                               # body text (tagged + unclassified)
        parts += [u.get("text", "") for u in cl.get("units", []) if u.get("reason") == "tag"]
        for ub in un.get("body", []):
            parts += _notes_or_text(ub)
    if cfg.get("vocab"):                                               # normalized vocab tags (inline on tag units)
        for u in cl.get("units", []):
            if u.get("reason") != "tag":
                continue
            for r in u.get("tags", []):
                pre = "tagneg" if r.get("negated") else "tag"
                for cat, anchors in r.get("tags", {}).items():
                    for a in anchors:
                        parts.append(f"{pre}_{cat}_{a}".replace(" ", ""))
    if cfg.get("typed"):                                               # classified named fields (values + tokens)
        parts += tf.get("part_name", []) + tf.get("material", []) + tf.get("coating", []) + tf.get("specification", [])
        parts += [f"field_parttype_{p}" for p in tf.get("part_type", [])]
        parts += [f"field_material_{m}" for m in tf.get("material_class", [])]
        parts += ["field_gentol_" + re.sub(r"[^a-z0-9]", "", g.lower()) for g in tf.get("general_tolerance", [])]
        parts += ["field_scale_" + s.replace(":", "_") for s in tf.get("scale", [])]
        for co in tf.get("coating", []):
            parts += [f"field_coating_{t}" for t in re.findall(r"[a-z0-9]+", co.lower()) if len(t) >= 3]
        parts += [s["family"] + s["number"] for s in tf.get("standards", [])]
    if cfg.get("include_noise"):                                       # debug: fold noise blocks back in
        for nb in sig.get("debug", {}).get("noise_blocks", []):
            parts += _notes_or_text(nb)
    if cfg.get("bom_leak"):                                            # LEAKAGE probe (target -> input)
        parts += [c.get("text", "") for c in sig.get("bom", {}).get("cells", [])]
    return " ".join(parts)


def load(limit=None, signal_name="signal.json", cfg=FP_DEFAULT):
    stems, docs = [], []
    for d in sorted(SIG.iterdir()):
        f = d / signal_name
        if not f.exists():
            continue
        try:
            sig = json.load(open(f))
        except Exception:
            continue
        stems.append(d.name); docs.append(fingerprint(sig, cfg))
        if limit and len(stems) >= limit:
            break
    return stems, docs


# ---- set-overlap metrics ----
def overlap(pred, true):
    if not true:
        return None
    inter = len(pred & true)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(true)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    j = inter / len(pred | true) if (pred | true) else 0.0
    return p, r, f1, j, (1.0 if inter else 0.0)


def _vote(neigh_sets, vote):
    """Aggregate neighbour truth-sets: keep elements appearing in >= `vote` neighbours."""
    if not neigh_sets:
        return set()
    from collections import Counter
    c = Counter(x for s in neigh_sets for x in s)
    return {x for x, n in c.items() if n >= vote}


def evaluate(stems, sim, truth, K, vote=1):
    """sim: (n,n) similarity matrix (diagonal ignored). truth: list of true-sets per stem.
    vote: a candidate is predicted only if it appears in >= vote of the K neighbours (1 = union)."""
    n = len(stems)
    rng = np.random.default_rng(0)
    rows, base = [], []
    for i in range(n):
        if not truth[i]:
            continue
        order = np.argsort(-sim[i]); order = order[order != i][:K]
        rows.append(overlap(_vote([truth[j] for j in order], vote), truth[i]))
        ridx = rng.choice([j for j in range(n) if j != i], size=min(K, n - 1), replace=False)
        base.append(overlap(_vote([truth[j] for j in ridx], vote), truth[i]))
    return np.array(rows), np.array(base)


def report(name, rows, base):
    m = rows.mean(0); b = base.mean(0)
    print(f"\n=== {name}  (n={len(rows)} queries with truth) ===")
    print(f"  {'':10}{'P':>7}{'R':>7}{'F1':>7}{'Jacc':>7}{'Hit@K':>7}")
    print(f"  retrieval {m[0]:7.3f}{m[1]:7.3f}{m[2]:7.3f}{m[3]:7.3f}{m[4]:7.3f}")
    print(f"  random    {b[0]:7.3f}{b[1]:7.3f}{b[2]:7.3f}{b[3]:7.3f}{b[4]:7.3f}")
    print(f"  lift x    {m[2]/b[2] if b[2] else float('nan'):6.1f} (F1)   "
          f"{m[4]/b[4] if b[4] else float('nan'):.1f} (Hit@K)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--vote", type=int, default=1, help=">=vote neighbours must agree (1=union)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--signal-name", default="signal.json",
                    help="which per-stem signal file to read (signal_allpages.json for the all-pages ablation)")
    # fingerprint-component ablation toggles (default = the leak-safe full fingerprint)
    ap.add_argument("--no-cells", action="store_true")
    ap.add_argument("--no-notes", action="store_true")
    ap.add_argument("--no-vocab", action="store_true")
    ap.add_argument("--no-typed", action="store_true", help="drop the unified typed-field + standards tokens")
    ap.add_argument("--include-noise", action="store_true", help="fold noise-flagged body blocks back in")
    ap.add_argument("--bom-leak", action="store_true", help="LEAKAGE probe: fold embedded-BOM target into input")
    args = ap.parse_args()
    cfg = dict(cells=not args.no_cells, notes=not args.no_notes, vocab=not args.no_vocab,
               typed=not args.no_typed, include_noise=args.include_noise, bom_leak=args.bom_leak)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    stems, docs = load(args.limit, args.signal_name, cfg)
    print(f"loaded {len(stems)} signals ({args.signal_name})  fp={ {k:v for k,v in cfg.items() if v} }")
    erp = ErpTruth()
    comp_truth = [erp.stem_to_components(s) for s in stems]
    phase_truth = [erp.stem_to_phases(s) for s in stems]
    print(f"  with BOM truth: {sum(1 for t in comp_truth if t)}  with phase truth: {sum(1 for t in phase_truth if t)}")

    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-_]+\b",
                          min_df=args.min_df, sublinear_tf=True)
    X = vec.fit_transform(docs)
    print(f"  vocab size: {len(vec.vocabulary_)}  fingerprint matrix: {X.shape}")
    sim = cosine_similarity(X)                      # <-- the pluggable similarity (text channel)

    cr, cb = evaluate(stems, sim, comp_truth, args.k, args.vote)
    report(f"BOM components  K={args.k} vote={args.vote}", cr, cb)
    pr, pb = evaluate(stems, sim, phase_truth, args.k, args.vote)
    report(f"Work-phases     K={args.k} vote={args.vote}", pr, pb)


if __name__ == "__main__":
    main()
