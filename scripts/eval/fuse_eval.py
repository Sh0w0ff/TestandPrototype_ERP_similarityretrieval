"""fuse_eval — text vs visual vs FUSED retrieval on the visual-covered subset (gate G4).

NEXT #2 from the E1 handoff. The cheap dim-strip lever is dead (lineweight can't separate
dimension lines on ~half of KC — see project_lineweight_finding), so frozen DINOv2 @ AUC 0.616
is the real operating point, not a buyable-back lower bound. This script answers the only question
that decides whether weak visual is worth ANY further investment (SSL adaptation): does fusing it
with text LIFT BOM/work-phase retrieval over text alone?

Design:
  * Restrict to the stems that have visual embeddings (review/embed_views), intersected with the
    text signals (cache/signal_v2). ~59 stems — a TINY corpus, so absolute numbers are noisy; the
    RELATIVE text vs visual vs fused comparison on ONE fixed set is the signal.
  * TEXT sim   = TF-IDF cosine over e1.fingerprint (identical fingerprint to the full-corpus baseline).
  * VISUAL sim = per-view assignment matcher (encoder_eval.matcher_score; matcher > pooled, §9.8.16).
  * Each matrix min-max normalised over off-diagonal entries so the fusion weight is meaningful.
    FUSED = alpha*textN + (1-alpha)*visualN, swept over alpha.
  * Score with e1.evaluate/report (set-overlap vs random-K) on BOM (G2) + work-phase (G3) truth.

Run:  python scripts/eval/fuse_eval.py [--k 5] [--vote 2] [--pooled]
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

ROOT = Path("/Users/sh0w0ff/FYP")
sys.path.insert(0, str(ROOT / "scripts"))
import paths
SIG = paths.TEXT_PIPE                          # <stem>/signal.json
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from erp_truth import ErpTruth
from e1 import fingerprint, evaluate, report
from encoder_eval import matcher_score


def topk_score(EA, EB, k=2):
    """'do these two drawings share their k most-similar views?' = mean of the k largest
    view-pair cosines. k=1 = pure max (one shared view); k=2 = robust top-2. Sidesteps the
    set-size dilution of pooled/matcher: a single shared principal view scores high even when
    the rest of the view sets diverge."""
    S = (EA @ EB.T).ravel()
    k = min(k, S.size)
    return float(np.sort(S)[-k:].mean())


def visual_sim_matrix(stems, emb, agg, k):
    n = len(stems)
    M = np.zeros((n, n))
    if agg == "pooled":
        pooled = {s: (emb[s].mean(0) / (np.linalg.norm(emb[s].mean(0)) + 1e-9)) for s in stems}
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = M[j, i] = float(pooled[stems[i]] @ pooled[stems[j]])
    else:
        fn = matcher_score if agg == "matcher" else (lambda a, b: topk_score(a, b, k))
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = M[j, i] = fn(emb[stems[i]], emb[stems[j]])
    return M


def minmax_offdiag(M):
    """Min-max scale a similarity matrix to [0,1] using off-diagonal entries only."""
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    lo, hi = M[mask].min(), M[mask].max()
    if hi - lo < 1e-12:
        return np.zeros_like(M)
    out = (M - lo) / (hi - lo)
    np.fill_diagonal(out, 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--vote", type=int, default=2)
    ap.add_argument("--emb", default=str(paths.VISUAL_INDEX))
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--pooled", action="store_true", help="(alias) --visual-agg pooled")
    ap.add_argument("--visual-agg", choices=["matcher", "pooled", "topk"], default="matcher")
    ap.add_argument("--topk", type=int, default=2, help="k for --visual-agg topk (1=max)")
    ap.add_argument("--alphas", default="1.0,0.75,0.5,0.25,0.0")
    args = ap.parse_args()

    # ---- visual side: stem -> per-view embeddings ----
    embdir = Path(args.emb)
    E = np.load(embdir / "embeddings.npy")
    manifest = json.loads((embdir / "manifest.json").read_text())
    vis_stems = sorted(set(m["stem"] for m in manifest))
    vidx = {s: [i for i, m in enumerate(manifest) if m["stem"] == s] for s in vis_stems}
    emb = {s: E[vidx[s]] for s in vis_stems}

    # ---- intersect with text signals, fix a single ordering ----
    stems = [s for s in vis_stems if (SIG / s / "signal.json").exists()]
    docs = [fingerprint(json.load(open(SIG / s / "signal.json"))) for s in stems]
    print(f"visual stems={len(vis_stems)}  with text signal={len(stems)}  (fusion set)")

    # ---- TEXT similarity ----
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-_]+\b",
                          min_df=args.min_df, sublinear_tf=True)
    Xt = vec.fit_transform(docs)
    text_sim = cosine_similarity(Xt)
    print(f"  text vocab={len(vec.vocabulary_)}  fingerprint={Xt.shape}")

    # ---- VISUAL similarity (matcher default; pooled / topk alternatives) ----
    agg = "pooled" if args.pooled else args.visual_agg
    vis_sim = visual_sim_matrix(stems, emb, agg, args.topk)
    print(f"  visual sim = {agg}" + (f" k={args.topk}" if agg == "topk" else ""))

    tN, vN = minmax_offdiag(text_sim), minmax_offdiag(vis_sim)

    # ---- truth on this subset ----
    erp = ErpTruth()
    comp_truth = [erp.stem_to_components(s) for s in stems]
    phase_truth = [erp.stem_to_phases(s) for s in stems]
    print(f"  with BOM truth={sum(1 for t in comp_truth if t)}  with phase truth={sum(1 for t in phase_truth if t)}")

    alphas = [float(a) for a in args.alphas.split(",")]
    for alpha in alphas:
        fused = alpha * tN + (1 - alpha) * vN
        tag = "TEXT" if alpha == 1.0 else ("VISUAL" if alpha == 0.0 else f"FUSED a={alpha}")
        cr, cb = evaluate(stems, fused, comp_truth, args.k, args.vote)
        report(f"BOM   {tag}  K={args.k} v={args.vote}", cr, cb)
        pr, pb = evaluate(stems, fused, phase_truth, args.k, args.vote)
        report(f"PHASE {tag}  K={args.k} v={args.vote}", pr, pb)


if __name__ == "__main__":
    main()
