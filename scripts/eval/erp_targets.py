"""erp_targets.py — additional ERP prediction tasks beyond G2(BOM)/G3(work-phase):
  (1) net-weight regression      — Item_Basic_Data 'Net weight'
  (2) work-content regression    — Σ Work_Phases setup+work times per item (hours)
  (3) part-type classification   — erp_truth.stem_to_parttype (head-noun)

Each evaluated text-only vs visual-only (frozen DINOv2 top-2 pool) vs fused, on an identical
stem set (target ∩ text ∩ visual), random 70/30 split (seed 42). CPU/sklearn — does NOT use MPS,
so it runs alongside the visual training. Records why product-family / work-centre / phase-sequence
were deferred (see PRINT at end).

Run: /opt/anaconda3/envs/fyp/bin/python scripts/eval/erp_targets.py
"""
import sys, csv, collections
from pathlib import Path
import numpy as np

ROOT = Path("/Users/sh0w0ff/FYP")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "textpipe"))
import paths, sys as _sys
import phase_mega_sweep as M
import phase_fusion as PF
from erp_truth import ErpTruth
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, f1_score, accuracy_score

TEXT_MODE = "raw_tb1_bodyall"
rng = np.random.default_rng(42)


def _f(x):
    try: return float(str(x).replace(",", "."))
    except: return 0.0


def load_item_weight():
    w = {}
    for r in csv.DictReader(open(ROOT / "Item_Basic_Data.csv", encoding="latin-1"), delimiter=";"):
        v = _f(r.get("Net weight", ""))
        if v > 0:
            w[r["Item"].strip()] = v
    return w


def load_item_workcontent():
    t = collections.defaultdict(float)
    for r in csv.DictReader(open(ROOT / "Work_Phases.csv", encoding="latin-1"), delimiter=";"):
        it = r["Item"].strip()
        t[it] += (_f(r["Machine setup time"]) + _f(r["Machine work time"])
                  + _f(r["Labor setup time"]) + _f(r["Labor work time"]))
    return {k: v for k, v in t.items() if v > 0}


def main():
    erp = ErpTruth()
    item_w = load_item_weight()
    item_t = load_item_workcontent()

    # text corpus + frozen visual
    stems, docs, _ = M.build_corpus(TEXT_MODE)
    sidx = {s: i for i, s in enumerate(stems)}
    _vi = "visual_pipe/index"
    if "--index" in _sys.argv: _vi = _sys.argv[_sys.argv.index("--index")+1]
    vis = PF.load_visual_top2(ROOT / _vi); print("VIS_INDEX:", _vi)

    # per-stem targets (average over the stem's items where multiple)
    def stem_val(stem, table):
        vals = [table[it] for it in erp.stem_to_items(stem) if it in table]
        return float(np.mean(vals)) if vals else None
    weight = {s: stem_val(s, item_w) for s in stems}
    work = {s: stem_val(s, item_t) for s in stems}
    ptype = {s: erp.stem_to_parttype(s) for s in stems}
    family = {s: erp.stem_to_family(s) for s in stems}   # ABB DRIVES / KONECRANES (= vendor)

    def report_reg(name, target, unit):
        kept = [s for s in stems if target.get(s) and s in vis]
        rng2 = np.random.default_rng(42); idx = np.arange(len(kept)); rng2.shuffle(idx)
        cut = int(0.7 * len(kept))
        tr = [kept[i] for i in idx[:cut]]; te = [kept[i] for i in idx[cut:]]
        y = np.array([target[s] for s in kept]); ylog = np.log1p(y)
        ytr = np.log1p(np.array([target[s] for s in tr]))
        yte = np.log1p(np.array([target[s] for s in te]))
        yte_lin = np.array([target[s] for s in te])
        Xt_tr, _, Xt_te, _ = M.build_features([docs[sidx[s]] for s in tr],
                                              [docs[sidx[s]] for s in te[:1]],
                                              [docs[sidx[s]] for s in te], "wordchar")
        Xv_tr = csr_matrix(np.stack([vis[s] for s in tr]))
        Xv_te = csr_matrix(np.stack([vis[s] for s in te]))
        base_mae = mean_absolute_error(yte_lin, np.full(len(te), np.expm1(ytr.mean())))
        print(f"\n### {name}  (n={len(kept)}, train={len(tr)} test={len(te)}, unit={unit})")
        print(f"  baseline (predict mean) MAE={base_mae:.2f}{unit}")
        for ch, Xtr, Xte in [("text", Xt_tr, Xt_te), ("visual", Xv_tr, Xv_te),
                             ("fused", hstack([Xt_tr, Xv_tr]).tocsr(), hstack([Xt_te, Xv_te]).tocsr())]:
            m = Ridge(alpha=10.0).fit(Xtr, ytr)
            pred_log = m.predict(Xte)
            pred_lin = np.expm1(pred_log)
            r2 = r2_score(yte, pred_log)
            mae = mean_absolute_error(yte_lin, pred_lin)
            print(f"  {ch:7s} R2(log)={r2:.3f}  MAE={mae:.2f}{unit}")

    def report_clf(name, target):
        kept = [s for s in stems if target.get(s) and s in vis]
        classes = sorted(set(target[s] for s in kept))
        c2i = {c: i for i, c in enumerate(classes)}
        rng2 = np.random.default_rng(42); idx = np.arange(len(kept)); rng2.shuffle(idx)
        cut = int(0.7 * len(kept))
        tr = [kept[i] for i in idx[:cut]]; te = [kept[i] for i in idx[cut:]]
        ytr = np.array([c2i[target[s]] for s in tr]); yte = np.array([c2i[target[s]] for s in te])
        Xt_tr, _, Xt_te, _ = M.build_features([docs[sidx[s]] for s in tr],
                                              [docs[sidx[s]] for s in te[:1]],
                                              [docs[sidx[s]] for s in te], "wordchar")
        Xv_tr = csr_matrix(np.stack([vis[s] for s in tr]))
        Xv_te = csr_matrix(np.stack([vis[s] for s in te]))
        maj = collections.Counter(ytr).most_common(1)[0][0]
        base = accuracy_score(yte, np.full(len(te), maj))
        print(f"\n### {name}  (n={len(kept)}, {len(classes)} classes, train={len(tr)} test={len(te)})")
        print(f"  baseline (majority) acc={base:.3f}")
        for ch, Xtr, Xte in [("text", Xt_tr, Xt_te), ("visual", Xv_tr, Xv_te),
                             ("fused", hstack([Xt_tr, Xv_tr]).tocsr(), hstack([Xt_te, Xv_te]).tocsr())]:
            m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
            pred = m.predict(Xte)
            print(f"  {ch:7s} acc={accuracy_score(yte, pred):.3f}  macroF1={f1_score(yte, pred, average='macro'):.3f}")

    def report_parttype_stated(target):
        """Split the part-type TEST set into STATED (true part-type token appears in the drawing's
        title-block text → an extraction case) vs UNSTATED (genuine shape-inference case). Reports
        accuracy on each subset per channel — the two numbers the deliverable vs inference framing needs."""
        kept = [s for s in stems if target.get(s) and s in vis]
        classes = sorted(set(target[s] for s in kept)); c2i = {c: i for i, c in enumerate(classes)}
        rng2 = np.random.default_rng(42); idx = np.arange(len(kept)); rng2.shuffle(idx)
        cut = int(0.7 * len(kept)); tr = [kept[i] for i in idx[:cut]]; te = [kept[i] for i in idx[cut:]]
        ytr = np.array([c2i[target[s]] for s in tr]); yte = np.array([c2i[target[s]] for s in te])
        stated = np.array([target[s].upper() in docs[sidx[s]].upper() for s in te])
        Xt_tr, _, Xt_te, _ = M.build_features([docs[sidx[s]] for s in tr],
                                              [docs[sidx[s]] for s in te[:1]],
                                              [docs[sidx[s]] for s in te], "wordchar")
        Xv_tr = csr_matrix(np.stack([vis[s] for s in tr])); Xv_te = csr_matrix(np.stack([vis[s] for s in te]))
        print(f"\n### PART-TYPE stated-vs-unstated  (test={len(te)}: stated={int(stated.sum())}, "
              f"unstated={int((~stated).sum())})")
        for ch, Xtr, Xte in [("text", Xt_tr, Xt_te), ("visual", Xv_tr, Xv_te),
                             ("fused", hstack([Xt_tr, Xv_tr]).tocsr(), hstack([Xt_te, Xv_te]).tocsr())]:
            pred = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr).predict(Xte)
            a_all = accuracy_score(yte, pred)
            a_st = accuracy_score(yte[stated], pred[stated]) if stated.any() else float("nan")
            a_un = accuracy_score(yte[~stated], pred[~stated]) if (~stated).any() else float("nan")
            print(f"  {ch:7s} all={a_all:.3f}  stated(extract)={a_st:.3f}  UNSTATED(infer)={a_un:.3f}")

    print("=" * 70)
    report_reg("NET WEIGHT (regression)", weight, "kg")
    report_reg("WORK-CONTENT / total time (regression)", work, "h")
    report_clf("PART-TYPE (classification)", {s: p for s, p in ptype.items() if p})
    report_parttype_stated({s: p for s, p in ptype.items() if p})
    # Product family = vendor: NOT a manufacturing-similarity result, but a useful triage output AND
    # the vendor-cosmetic separability probe (positive control + quantifies the cross-vendor confound,
    # RQ2). Expect visual to be strong here (cosmetics are visual) — the opposite of work-phase.
    report_clf("PRODUCT FAMILY / vendor (classification — cosmetic-separability probe)",
               {s: f for s, f in family.items() if f})

    print("\n" + "=" * 70)
    print("DEFERRED (recorded for the thesis):")
    print(" - Work centre (311 classes): far too granular/sparse — the same noisy-label failure")
    print("   mode as the 185 raw phase descriptions; the 21-line abstraction is the learnable level.")
    print(" - Ordered phase SEQUENCE: richer than set-of-lines, but the work-phase INFORMATION")
    print("   ceiling (short docs under-determined, §9.8.40) caps a harder framing too — deferred.")
    print("NOTE: product family is INCLUDED above (vendor triage + cosmetic-separability probe),")
    print("      not used as a similarity feature (that would be leakage).")


if __name__ == "__main__":
    main()
