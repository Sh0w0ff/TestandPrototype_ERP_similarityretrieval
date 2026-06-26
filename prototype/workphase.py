"""workphase.py — the direct work-phase classifier.

A multi-label classifier over the 21 production-line classes: word+char TF-IDF -> ResMLP-BN,
trained with focal loss on an iterative multi-label stratified split. It reuses the exact model +
training regime from scripts/eval/phase_mega_sweep.py (no re-tuning); we add persistence + a
single-document predict() that the sweep harness does not expose.

    train_and_cache()  -> fits vectorizers + model on the corpus, pickles to cache/, returns metrics
    load()             -> restore the cached classifier
    predict(doc, ...)  -> work-phase labels (+ scores) for one drawing's text fingerprint
"""
from __future__ import annotations
import os, sys, json, pickle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
for p in (SCRIPTS, SCRIPTS / "eval", SCRIPTS / "textpipe"):
    sys.path.insert(0, str(p))

import torch
import torch.nn as nn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

import paths
import e1
import phase_mega_sweep as PMS          # reuse model classes, loss, split, target loader

CACHE = REPO / "cache"
CACHE.mkdir(exist_ok=True)

# Two model scopes:
#   "eval"   trained on the TRAIN split only, scored on the held-out TEST split -> the honest,
#            reportable accuracy. Use it when evaluating a held-out test drawing (test mode).
#   "deploy" trained on ALL labelled drawings -> maximum performance for a genuinely new drawing.
def _paths(scope):
    return (CACHE / f"workphase_{scope}_vectorizers.pkl",
            CACHE / f"workphase_{scope}_resmlp_bn.pt",
            CACHE / f"workphase_{scope}_meta.json")

MIN_COUNT = 20          # freq>=20 -> 21 production-line classes
EPOCHS = 120
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------- corpus (classified_p1 fingerprint)
def _corpus():
    """stem -> (fingerprint doc, production-line label set), over every signal.json in the pool."""
    stems, docs, labels = [], [], []
    for d in sorted(paths.TEXT_PIPE.iterdir()):
        if not d.is_dir():
            continue
        f = d / "signal.json"
        if not f.exists():
            continue
        try:
            doc = e1.fingerprint(json.load(open(f)))
        except Exception:
            continue
        pls = PMS.stem_phases(d.name)         # production-line target
        if not pls:
            continue
        stems.append(d.name); docs.append(doc); labels.append(pls)
    return stems, docs, labels


def _vectorize_fit(docs):
    wv = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-_]+\b",
                         ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, sublinear_tf=True)
    Xw = wv.fit_transform(docs); Xc = cv.fit_transform(docs)
    return wv, cv, hstack([Xw, Xc], format="csr")


def _vectorize(wv, cv, docs):
    return hstack([wv.transform(docs), cv.transform(docs)], format="csr")


# ---------------------------------------------------------------- train
def train_and_cache(scope="eval", epochs=EPOCHS, verbose=True):
    """scope='eval'   -> train on the TRAIN split, report held-out TEST metrics (honest accuracy).
       scope='deploy' -> train on ALL labelled drawings (no held-out metrics), for new-drawing use."""
    VEC_PATH, MODEL_PATH, META_PATH = _paths(scope)
    stems, docs, labels = _corpus()
    classes = sorted({p for L in labels for p in L if sum(p in M for M in labels) >= MIN_COUNT})
    c2i = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(stems), len(classes)), np.float32)
    for i, L in enumerate(labels):
        for p in L:
            if p in c2i:
                Y[i, c2i[p]] = 1.0

    tr, va, te = PMS.iterative_split(Y, ratios=(0.70, 0.15, 0.15), seed=42)
    train_idx = tr if scope == "eval" else np.concatenate([tr, va, te])   # deploy uses every drawing
    wv, cv, _ = _vectorize_fit([docs[i] for i in train_idx])
    X_tr = _vectorize(wv, cv, [docs[i] for i in train_idx])
    X_va = _vectorize(wv, cv, [docs[i] for i in va])      # va = checkpoint set for both scopes
    X_te = _vectorize(wv, cv, [docs[i] for i in te])
    Y_tr, Y_va, Y_te = Y[train_idx], Y[va], Y[te]
    if verbose:
        print(f"workphase[{scope}]: {len(stems)} drawings, {len(classes)} classes, "
              f"dim={X_tr.shape[1]}, train={len(train_idx)} (val={len(va)} test={len(te)}) on {DEVICE}")

    model = PMS.build_model("resmlp_bn", X_tr.shape[1], len(classes)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos = Y_tr.sum(0); neg = Y_tr.shape[0] - pos
    pw = torch.tensor(np.where(pos > 0, neg / np.maximum(pos, 1), 1.0), dtype=torch.float32).to(DEVICE)
    crit = PMS.FocalBCE(gamma=2.0, pos_weight=pw)

    Xtr_t = torch.tensor(X_tr.toarray(), dtype=torch.float32).to(DEVICE)
    Ytr_t = torch.tensor(Y_tr, dtype=torch.float32).to(DEVICE)
    Xva_t = torch.tensor(X_va.toarray(), dtype=torch.float32).to(DEVICE)

    best_f1, best_state, warmup = -1.0, None, max(5, epochs // 10)
    BATCH, n_tr = 256, Xtr_t.shape[0]
    for ep in range(1, epochs + 1):
        if ep <= warmup:
            for pg in opt.param_groups:
                pg["lr"] = 1e-3 * ep / warmup
        model.train(); perm = torch.randperm(n_tr, device=DEVICE)
        for i in range(0, n_tr, BATCH):
            bi = perm[i:i + BATCH]
            opt.zero_grad(); crit(model(Xtr_t[bi]), Ytr_t[bi]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if ep > warmup:
            sched.step()
        if ep % (epochs // 5) == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                pv = (model(Xva_t).cpu().numpy() > 0).astype(int)
            f1s = []
            for j in range(len(classes)):
                ap = int(Y_va[:, j].sum())
                if ap == 0:
                    continue
                tp = int(((pv[:, j] == 1) & (Y_va[:, j] == 1)).sum()); pp = int(pv[:, j].sum())
                pr = tp / pp if pp else 0.0; rc = tp / ap
                f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
            vf1 = float(np.mean(f1s)) if f1s else 0.0
            if vf1 > best_f1:
                best_f1 = vf1; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  ep {ep:3d}  val-macroF1 {vf1:.3f}")

    model.load_state_dict(best_state); model.eval()
    # held-out TEST metrics are meaningful only for the eval scope (deploy trained ON the test rows)
    metrics = _eval(model, X_te, Y_te, classes) if scope == "eval" else None
    if verbose and metrics:
        print(f"workphase[eval] TEST  macro={metrics['macro']:.3f}  micro={metrics['micro']:.3f}  "
              f"exact={metrics['exact']:.3f}  (expected macro ~0.91 / exact ~0.60 on this corpus)")
    elif verbose:
        print(f"workphase[deploy] trained on all {len(train_idx)} drawings (no held-out metric)")

    pickle.dump(dict(wv=wv, cv=cv, classes=classes), open(VEC_PATH, "wb"))
    torch.save({"state": best_state, "in_d": X_tr.shape[1], "n_out": len(classes)}, MODEL_PATH)
    json.dump(dict(scope=scope, classes=classes, metrics=metrics, min_count=MIN_COUNT,
                   epochs=epochs, n_train=len(train_idx)), open(META_PATH, "w"), indent=2)
    return metrics


def _eval(model, X_te, Y_te, classes):
    Xte_t = torch.tensor(X_te.toarray(), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        P = (model(Xte_t).cpu().numpy() > 0).astype(int)
    macro, mtp, mpp, map_, exact, n = [], 0, 0, 0, 0, 0
    for i in range(len(Y_te)):
        pred = set(np.where(P[i])[0]); truth = set(np.where(Y_te[i])[0])
        if not truth:
            continue
        n += 1; tp = len(pred & truth); mtp += tp; mpp += len(pred); map_ += len(truth)
        pr = tp / len(pred) if pred else 0.0; rc = tp / len(truth)
        macro.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
        exact += int(pred == truth)
    return dict(macro=float(np.mean(macro)) if macro else 0.0,
                micro=mtp / mpp if mpp else 0.0, exact=exact / n if n else 0.0, n=n)


# ---------------------------------------------------------------- load + predict
_CLF = {}
def load(scope="eval"):
    if scope not in _CLF:
        VEC_PATH, MODEL_PATH, _ = _paths(scope)
        if not (VEC_PATH.exists() and MODEL_PATH.exists()):
            raise FileNotFoundError(f"work-phase '{scope}' classifier not trained — run train() first.")
        vp = pickle.load(open(VEC_PATH, "rb"))
        ck = torch.load(MODEL_PATH, map_location=DEVICE)
        model = PMS.build_model("resmlp_bn", ck["in_d"], ck["n_out"]).to(DEVICE)
        model.load_state_dict(ck["state"]); model.eval()
        _CLF[scope] = dict(model=model, wv=vp["wv"], cv=vp["cv"], classes=vp["classes"])
    return _CLF[scope]


def predict(doc, scope="eval", topn=None):
    """Predict work-phase production-lines for one drawing's text fingerprint, using the model of
    the given scope ('eval' = train-split model, 'deploy' = all-data model).
    Returns [(label, prob)] sorted by prob; thresholded at logit>0 unless topn given."""
    c = load(scope)
    X = _vectorize(c["wv"], c["cv"], [doc])
    with torch.no_grad():
        logits = c["model"](torch.tensor(X.toarray(), dtype=torch.float32).to(DEVICE)).cpu().numpy().ravel()
    probs = 1 / (1 + np.exp(-logits))
    order = np.argsort(-logits)
    keep = order[:topn] if topn else [j for j in order if logits[j] > 0]
    return [(c["classes"][j], float(probs[j])) for j in keep]


def is_trained(scope="eval"):
    VEC_PATH, MODEL_PATH, _ = _paths(scope)
    return VEC_PATH.exists() and MODEL_PATH.exists()


if __name__ == "__main__":
    train_and_cache("eval")
    train_and_cache("deploy")
