"""bom_classify.py — predict the BOM component SET as a multi-label classifier, exactly like the
work-phase (G3) setup: word+char TF-IDF -> self-attention MLP with focal BCE, iterative multi-label
stratified split, macro/micro-F1 + exact + Jaccard. Label space = components with corpus frequency
>= T (singletons are unlearnable, so we sweep the threshold). Mirrors phase_mega_sweep.run_one.

CPU (so it coexists with the GPU/MPS training). Usage: python scripts/eval/bom_classify.py
"""
import sys, collections
from pathlib import Path
import numpy as np
ROOT = Path("/Users/sh0w0ff/FYP")
for p in ["scripts", "scripts/eval", "scripts/textpipe"]:
    sys.path.insert(0, str(ROOT / p))
import phase_mega_sweep as M
from erp_truth import ErpTruth

DEVICE = "cpu"
MODEL = "selfattn"
EPOCHS = 120


def main():
    erp = ErpTruth()
    stems, docs, _ = M.build_corpus("raw_tb1_bodyall")
    sidx = {s: i for i, s in enumerate(stems)}
    comps = {s: erp.stem_to_components(s) for s in stems}
    freq = collections.Counter(c for s in stems for c in comps[s])
    print(f"corpus: {len(stems)} drawings, {len(freq)} distinct components "
          f"(singletons {100*sum(1 for n in freq.values() if n==1)/len(freq):.0f}%)\n")
    print(f"{'thresh':>6} {'#labels':>8} {'#draw':>6} | {'macroF1':>8} {'microF1':>8} "
          f"{'exact':>7} {'jacc':>6} {'±1':>6}")

    for T in [20, 10, 5, 2]:
        vocab = sorted(c for c, n in freq.items() if n >= T)
        c2i = {c: i for i, c in enumerate(vocab)}
        kept = [s for s in stems if any(c in c2i for c in comps[s])]
        if len(vocab) < 2 or len(kept) < 50:
            print(f"{T:>6} {len(vocab):>8} {len(kept):>6} | (too small)"); continue
        Y = np.zeros((len(kept), len(vocab)), dtype=np.float32)
        for i, s in enumerate(kept):
            for c in comps[s]:
                if c in c2i:
                    Y[i, c2i[c]] = 1.0
        tr, va, te = M.iterative_split(Y, ratios=(0.70, 0.15, 0.15), seed=42)
        d = [docs[sidx[s]] for s in kept]
        Xtr, Xva, Xte, _ = M.build_features([d[i] for i in tr], [d[i] for i in va],
                                            [d[i] for i in te], "wordchar")
        res = M.run_one(Xtr, Y[tr], Xva, Y[va], Xte, Y[te], MODEL, EPOCHS, DEVICE)
        print(f"{T:>6} {len(vocab):>8} {len(kept):>6} | {res['macro']:>8.3f} {res['micro']:>8.3f} "
              f"{res['exact']:>7.3f} {res['jaccard']:>6.3f} {res['offby1']:>6.3f}")
        if T == 5:   # show a few example predictions for intuition
            preds = res["preds"]; te_s = [kept[i] for i in te]; inv = {i: c for c, i in c2i.items()}
            print("\n   sample predictions (threshold 5):")
            for r in range(min(3, len(te))):
                pr = {inv[j] for j in np.where(preds[r])[0]}
                gt = {inv[j] for j in np.where(Y[te][r])[0]}
                print(f"     {te_s[r][:34]:36s} pred∩gt={len(pr&gt)}  pred={len(pr)}  gt={len(gt)}")
            print()


if __name__ == "__main__":
    main()
