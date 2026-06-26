"""phase_length_analysis.py — does document length (token count) affect the model?

Answers two questions on the corrected 21-class production-line target:
  1. Capture: how many raw tokens per doc, and how many survive TF-IDF min_df pruning?
     (TF-IDF is fixed-dim + order-free, so length never truncates — but min_df drops
      tokens seen in <2-3 docs, which hits dimension-dump-heavy docs hardest.)
  2. Training parity: at the same epochs, every doc is one sample seen once/epoch.
     Does exact-match correctness vary with doc length? (bucket test docs by length)
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/sh0w0ff/FYP")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "textpipe"))

import torch
import phase_mega_sweep as M

MODE   = sys.argv[1] if len(sys.argv) > 1 else "raw_tb1_bodyall"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 150


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Mode={MODE}  epochs={EPOCHS}  device={device}\n")

    # --- build corpus + 21-class target + iterative split (same path as the sweep) ---
    stems, docs, phase_sets = M.build_corpus(MODE)
    import collections
    cnt = collections.Counter()
    ordered = []
    for s in stems:
        ph = M.stem_phases(s)
        if not ph:
            continue
        ordered.append(s)
        for p in ph:
            cnt[p] += 1
    classes = sorted(p for p, c in cnt.items() if c >= 20)
    c2i = {c: i for i, c in enumerate(classes)}
    sidx = {s: i for i, s in enumerate(stems)}

    Y_all = np.zeros((len(ordered), len(classes)), dtype=np.float32)
    for i, s in enumerate(ordered):
        for p in M.stem_phases(s):
            if p in c2i:
                Y_all[i, c2i[p]] = 1.0
    tr, va, te = M.iterative_split(Y_all, ratios=(0.70, 0.15, 0.15), seed=42)
    split = {}
    for i in tr: split[ordered[i]] = "train"
    for i in va: split[ordered[i]] = "val"
    for i in te: split[ordered[i]] = "test"

    def rows(names):
        ds, ys = [], []
        for s in names:
            ds.append(docs[sidx[s]])
            r = np.zeros(len(classes), dtype=np.float32)
            for p in M.stem_phases(s):
                if p in c2i:
                    r[c2i[p]] = 1.0
            ys.append(r)
        return ds, np.array(ys)

    tr_names = [ordered[i] for i in tr]
    te_names = [ordered[i] for i in te]
    d_tr, Y_tr = rows(tr_names)
    d_te, Y_te = rows(te_names)

    # --- Q1: token capture ---
    tok = lambda d: len(d.split())
    raw_tr = np.array([tok(d) for d in d_tr])
    raw_te = np.array([tok(d) for d in d_te])
    print("=== Q1: TOKEN CAPTURE ===")
    print(f"raw tokens/doc (train): min={raw_tr.min()} med={int(np.median(raw_tr))} "
          f"mean={raw_tr.mean():.0f} max={raw_tr.max()} empty={(raw_tr==0).sum()}")
    print(f"raw tokens/doc (test):  min={raw_te.min()} med={int(np.median(raw_te))} "
          f"mean={raw_te.mean():.0f} max={raw_te.max()} empty={(raw_te==0).sum()}")

    X_tr, X_va, X_te, dim = M.build_features(d_tr, d_te[:1], d_te, "wordchar")
    # unique tokens per doc that survive vocab (word part only, before min_df vs after)
    from sklearn.feature_extraction.text import TfidfVectorizer
    full = TfidfVectorizer(lowercase=True,
                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-_]+\b",
                           ngram_range=(1, 2), min_df=1)
    pruned = TfidfVectorizer(lowercase=True,
                             token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-_]+\b",
                             ngram_range=(1, 2), min_df=2)
    full.fit(d_tr); pruned.fit(d_tr)
    print(f"\nword vocab (train): min_df=1 → {len(full.vocabulary_):,}   "
          f"min_df=2 → {len(pruned.vocabulary_):,}   "
          f"dropped {len(full.vocabulary_)-len(pruned.vocabulary_):,} rare "
          f"({100*(1-len(pruned.vocabulary_)/len(full.vocabulary_)):.0f}%)")
    print(f"final wordchar feature dim (with char n-grams): {dim:,}")
    print("→ no length truncation (fixed-dim bag); but rare/unique tokens (incl. one-off")
    print("  dimension values) are pruned by min_df, so token-rich docs lose their unique numbers.")

    # --- Q2: train one model, bucket test exact-match by doc length ---
    print("\n=== Q2: TRAINING PARITY vs DOC LENGTH ===")
    print("(every doc = 1 sample seen once/epoch; check if length affects correctness)")
    X_tr2, X_va2, X_te2, _ = M.build_features(d_tr, d_te, d_te, "wordchar")
    res = M.run_one(X_tr2, Y_tr, X_te2, Y_te, X_te2, Y_te, "selfattn", EPOCHS, device)
    preds = res["preds"]
    exact = (preds == Y_te).all(1).astype(int)
    nlab  = Y_te.sum(1).astype(int)

    # buckets by raw token count
    qs = np.quantile(raw_te, [0.25, 0.5, 0.75])
    buckets = [("Q1 shortest", raw_te <= qs[0]),
               ("Q2", (raw_te > qs[0]) & (raw_te <= qs[1])),
               ("Q3", (raw_te > qs[1]) & (raw_te <= qs[2])),
               ("Q4 longest", raw_te > qs[2])]
    print(f"\nselfattn overall test exact = {exact.mean():.3f}  (n={len(exact)})")
    print(f"{'bucket':<13} {'tok range':<16} {'n':>4} {'exact':>7} {'avg#labels':>11}")
    for name, m in buckets:
        if m.sum() == 0:
            continue
        lo, hi = int(raw_te[m].min()), int(raw_te[m].max())
        print(f"{name:<13} {f'{lo}-{hi}':<16} {int(m.sum()):>4} "
              f"{exact[m].mean():>7.3f} {nlab[m].mean():>11.2f}")
    # correlation
    if raw_te.std() > 0:
        r = np.corrcoef(raw_te, exact)[0, 1]
        rl = np.corrcoef(raw_te, nlab)[0, 1]
        print(f"\ncorr(tokens, correct) = {r:+.3f}   corr(tokens, #labels) = {rl:+.3f}")


if __name__ == "__main__":
    main()
