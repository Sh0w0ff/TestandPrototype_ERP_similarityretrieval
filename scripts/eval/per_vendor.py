"""per_vendor.py — vendor-agnosticity check: do the READY tasks (work-phase G3, BOM G2) work
for BOTH vendors, or are they carried by the easier one? Trains the same text classifier as
elsewhere, then splits the held-out test set by vendor (ABB vs KC) and reports metrics per vendor.
Vendor agnosticity = comparable performance across vendors, on a shared cross-vendor target, with
vendor never used as a feature/label. CPU. Usage: python scripts/eval/per_vendor.py
"""
import sys, collections
from pathlib import Path
import numpy as np
ROOT = Path("/Users/sh0w0ff/FYP")
for p in ["scripts", "scripts/eval", "scripts/textpipe"]:
    sys.path.insert(0, str(ROOT / p))
import phase_mega_sweep as M
import phase_classify as PC
from erp_truth import ErpTruth

DEVICE, MODEL, EPOCHS = "cpu", "selfattn", 120


def vendor_of(fam):
    f = (fam or "").upper()
    return "ABB" if "ABB" in f else ("KC" if "KONE" in f else "?")


def submetrics(preds, Y, mask):
    P, Yt = preds[mask], Y[mask]
    n = len(P)
    if n == 0: return None
    exact = float(np.mean((P == Yt).all(1)))
    tp = int(((P == 1) & (Yt == 1)).sum()); pp = int(P.sum()); gg = int(Yt.sum())
    prec = tp / pp if pp else 0.0; rec = tp / gg if gg else 0.0
    micro = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    jac = float(np.mean([len(set(np.where(P[i])[0]) & set(np.where(Yt[i])[0])) /
                         max(1, len(set(np.where(P[i])[0]) | set(np.where(Yt[i])[0])))
                         for i in range(n)]))
    return n, exact, micro, jac


def run_task(name, target_sets, classes, kept, docs, sidx, vend):
    c2i = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(kept), len(classes)), dtype=np.float32)
    for i, s in enumerate(kept):
        for c in target_sets[s]:
            if c in c2i: Y[i, c2i[c]] = 1.0
    tr, va, te = M.iterative_split(Y, ratios=(0.70, 0.15, 0.15), seed=42)
    d = [docs[sidx[s]] for s in kept]
    Xtr, Xva, Xte, _ = M.build_features([d[i] for i in tr], [d[i] for i in va],
                                        [d[i] for i in te], "wordchar")
    res = M.run_one(Xtr, Y[tr], Xva, Y[va], Xte, Y[te], MODEL, EPOCHS, DEVICE)
    preds, Yte = res["preds"], Y[te]
    vte = np.array([vend[kept[i]] for i in te])
    print(f"\n=== {name}  ({len(classes)} classes, test n={len(te)}) ===")
    print(f"  {'subset':8s} {'n':>5} {'exact':>7} {'microF1':>8} {'jacc':>6}")
    for sub, mask in [("ALL", np.ones(len(te), bool)), ("ABB", vte == "ABB"), ("KC", vte == "KC")]:
        m = submetrics(preds, Yte, mask)
        if m: print(f"  {sub:8s} {m[0]:>5} {m[1]:>7.3f} {m[2]:>8.3f} {m[3]:>6.3f}")


def main():
    erp = ErpTruth()
    stems, docs, _ = M.build_corpus("raw_tb1_bodyall")
    sidx = {s: i for i, s in enumerate(stems)}
    vend = {s: vendor_of(erp.stem_to_family(s)) for s in stems}
    vc = collections.Counter(vend.values())
    print(f"corpus vendors: {dict(vc)}")

    # ---- work-phase (production-line, freq>=20) ----
    pl = PC.load_prod_lines()
    plf = collections.Counter(c for s in stems for c in pl.get(s, set()))
    plcls = sorted(c for c, n in plf.items() if n >= 20)
    plkept = [s for s in stems if pl.get(s)]
    run_task("WORK-PHASE (production-line)", {s: pl.get(s, set()) for s in stems}, plcls, plkept, docs, sidx, vend)

    # ---- BOM (components, freq>=10) ----
    comps = {s: erp.stem_to_components(s) for s in stems}
    cf = collections.Counter(c for s in stems for c in comps[s])
    bcls = sorted(c for c, n in cf.items() if n >= 10)
    bkept = [s for s in stems if any(cf[c] >= 10 for c in comps[s])]
    run_task("BOM (components freq>=10)", comps, bcls, bkept, docs, sidx, vend)


if __name__ == "__main__":
    main()
