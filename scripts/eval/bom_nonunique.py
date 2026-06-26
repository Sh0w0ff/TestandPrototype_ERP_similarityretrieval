"""bom_nonunique.py — BOM proposal evaluated only on RECOVERABLE (non-unique) components.

Singletons (components on exactly one drawing) can never be proposed from a neighbour, so they
impose an artificial ceiling (~79% of components are singletons). Here we (a) report the standard
full-set F1, and (b) restrict the ground truth to NON-UNIQUE components (appear on >=2 drawings) —
the subset retrieval-aggregation can actually recover — to measure real performance unmasked by the
singleton ceiling. Text retrieval-aggregation baseline (TF-IDF top-k neighbours, vote threshold).
CPU. Usage: python scripts/eval/bom_nonunique.py [--k 10] [--vote 2]
"""
import sys, collections
from pathlib import Path
import numpy as np
ROOT = Path("/Users/sh0w0ff/FYP")
for p in ["scripts", "scripts/eval", "scripts/textpipe"]:
    sys.path.insert(0, str(ROOT / p))
import phase_mega_sweep as M
from erp_truth import ErpTruth
from sklearn.feature_extraction.text import TfidfVectorizer

K = int(sys.argv[sys.argv.index("--k")+1]) if "--k" in sys.argv else 10
VOTE = int(sys.argv[sys.argv.index("--vote")+1]) if "--vote" in sys.argv else 2


def prf(pred, gt):
    tp = len(pred & gt)
    p = tp/len(pred) if pred else 0.0
    r = tp/len(gt) if gt else 0.0
    return tp, len(pred), len(gt)


def main():
    erp = ErpTruth()
    stems, docs, _ = M.build_corpus("raw_tb1_bodyall")
    sidx = {s: i for i, s in enumerate(stems)}
    comps = {s: erp.stem_to_components(s) for s in stems}
    kept = [s for s in stems if comps[s]]
    # component -> number of distinct drawings it appears on
    parents = collections.Counter()
    for s in kept:
        for c in comps[s]:
            parents[c] += 1
    nonuniq = {c for c, n in parents.items() if n >= 2}
    tot = len(parents)
    print(f"pool={len(kept)} drawings with a BOM;  distinct components={tot};  "
          f"non-unique (>=2 drawings)={len(nonuniq)} ({100*len(nonuniq)/tot:.1f}%);  "
          f"singletons={100*(1-len(nonuniq)/tot):.1f}%")
    print(f"retrieval-aggregation: TF-IDF top-{K} neighbours, propose components in >= {VOTE} of them\n")

    T = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=40000)\
        .fit_transform([docs[sidx[s]] for s in kept])
    n = len(kept)
    S = (T @ T.T).toarray(); np.fill_diagonal(S, -2)
    order = np.argsort(-S, axis=1)

    # micro accumulators for full vs non-unique-only ground truth
    def run(restrict):
        TP = PP = GG = 0
        for qi, s in enumerate(kept):
            nb = order[qi][:K]
            cnt = collections.Counter()
            for j in nb:
                for c in comps[kept[j]]:
                    cnt[c] += 1
            pred = {c for c, v in cnt.items() if v >= VOTE}
            gt = comps[s]
            if restrict:
                pred &= nonuniq; gt = gt & nonuniq
            tp, pp, gg = prf(pred, gt)
            TP += tp; PP += pp; GG += gg
        P = TP/PP if PP else 0.0; R = TP/GG if GG else 0.0
        F = 2*P*R/(P+R) if (P+R) else 0.0
        return P, R, F
    for label, restrict in [("ALL components (full set, singleton-capped)", False),
                            ("NON-UNIQUE components only (recoverable subset)", True)]:
        P, R, F = run(restrict)
        print(f"  {label:48s}  P={P:.3f} R={R:.3f} F1={F:.3f}")


if __name__ == "__main__":
    main()
