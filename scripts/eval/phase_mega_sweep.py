"""phase_mega_sweep.py — Comprehensive multi-axis comparison for work-phase classification.

Axes swept:
  TEXT VARIANTS (10):
    raw_p1          raw cells+body from page 0 only          (no cleaning)
    raw_allpages    raw cells+body from ALL pages             (no cleaning)
    raw_tb1_bodyall raw cells page 0 + raw body ALL pages    (no cleaning)
    raw_tb2_bodyall raw cells pages 0-1 + raw body ALL pages (no cleaning)
    lc_p1           light_clean cells+body page 0            (strip noise tokens)
    lc_allpages     light_clean cells+body ALL pages
    lc_tb1_bodyall  light_clean cells p0 + body ALL pages
    lc_tb2_bodyall  light_clean cells p0-1 + body ALL pages
    classified_p1   full classified signal.json fingerprint  (current best)
    classified_allpages  signal_allpages.json fingerprint

  FEATURE TYPES (3):
    unigram   TF-IDF unigrams
    bigram    TF-IDF word bigrams
    wordchar  TF-IDF word bigrams + char 3-4grams  (best from v2 study)

  MODELS (6):
    mlp       plain MLP (3 layers, 512-d)
    resmlp4   ResMLP 512-d 4 ResBlocks + skip connections
    resmlp8   ResMLP 1024-d 8 ResBlocks (best from neural_search study)
    geglu     GEGLU-gated MLP
    selfattn  Self-attention MLP
    resmlp_bn ResMLP-BN: batch norm instead of layer norm (better for noisy raw features)

Run:
  python scripts/eval/phase_mega_sweep.py                         # full sweep
  python scripts/eval/phase_mega_sweep.py --texts raw_p1 lc_p1   # subset of texts
  python scripts/eval/phase_mega_sweep.py --models resmlp8        # single model
  python scripts/eval/phase_mega_sweep.py --features wordchar     # single feature type
  python scripts/eval/phase_mega_sweep.py --quick                 # 80 epochs, wordchar only

Output: allpages_review/phase_mega_sweep.json  +  console table
"""
import sys, json, argparse, collections, time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "textpipe"))
import paths
from erp_truth import ErpTruth
import e1
import phase_classify as _pc

# TARGET = production-line codes (the 21 work phases), matching phase_neural_v2.py.
# NOT erp.stem_to_phases() which returns 246 raw multilingual work-phase descriptions
# (§9.8.38: that was the wrong, far harder target — inflated label set, deflated exact match).
_STEM_PL = None
def stem_phases(stem):
    global _STEM_PL
    if _STEM_PL is None:
        _STEM_PL = _pc.load_prod_lines()
    return _STEM_PL.get(stem, set())

import re as _re

_RE_SYMBOL = _re.compile(r"^[^\w]+$")
_RE_ONLY_O = _re.compile(r"^[OoＯ0º]+$")

def _lc(text):
    """Light noise removal applied directly to raw token stream.
    Keeps: ASCII tokens ≥2 meaningful chars (words, codes, numbers, dims).
    Drops: non-ASCII glyphs, symbol-only runs, O-run artefacts, 1-char fragments.
    Returns a plain string — does NOT import textpipe; built from raw data directly.
    """
    parts = []
    for tok in text.split():
        t = tok.strip()
        if not t or not t.isascii():
            continue
        if _RE_ONLY_O.match(t):
            continue
        if _RE_SYMBOL.match(t):
            continue
        if len(t.strip(".,:;()-/")) < 2:
            continue
        parts.append(t)
    return " ".join(parts)


# ============================================================
# TEXT VARIANT BUILDERS
# ============================================================

def _cell_texts(page, clean=False):
    out = []
    for c in page.get("cells", []):
        t = c.get("text", "") if isinstance(c, dict) else str(c)
        if clean:
            t = _lc(t)
        if t.strip():
            out.append(t.strip())
    return out


def _body_texts(page, clean=False):
    out = []
    for b in page.get("body_blocks", []):
        t = b.get("text", "") if isinstance(b, dict) else str(b)
        if clean:
            t = _lc(t)
        if t.strip():
            out.append(t.strip())
    return out


def raw_fingerprint(raw, mode="raw_p1"):
    """Build a text string from raw_pages.json according to `mode`."""
    pages = raw.get("pages", [])
    if not pages:
        return ""

    clean = mode.startswith("lc_")

    if mode in ("raw_p1", "lc_p1"):
        parts = _cell_texts(pages[0], clean) + _body_texts(pages[0], clean)

    elif mode in ("raw_allpages", "lc_allpages"):
        parts = []
        for pg in pages:
            parts += _cell_texts(pg, clean) + _body_texts(pg, clean)

    elif mode in ("raw_tb1_bodyall", "lc_tb1_bodyall"):
        # title block from page 0 only; body from all pages
        parts = _cell_texts(pages[0], clean)
        for pg in pages:
            parts += _body_texts(pg, clean)

    elif mode in ("raw_tb2_bodyall", "lc_tb2_bodyall"):
        # title block from pages 0-1; body from all pages
        parts = _cell_texts(pages[0], clean)
        if len(pages) > 1:
            parts += _cell_texts(pages[1], clean)
        for pg in pages:
            parts += _body_texts(pg, clean)

    else:
        parts = _cell_texts(pages[0], clean) + _body_texts(pages[0], clean)

    return " ".join(parts)


def classified_fingerprint(sig, all_pages=False):
    """Use existing e1.fingerprint on signal.json / signal_allpages.json."""
    return e1.fingerprint(sig, e1.FP_DEFAULT)


# ============================================================
# CORPUS BUILDER
# ============================================================

def build_corpus(text_mode, signal_name="signal.json"):
    erp = ErpTruth()
    stems, docs, phase_sets = [], [], []

    use_classified = text_mode.startswith("classified")
    # legacy_body / legacy_default: read the PRE-2026-06-05 signal.json (rich body text
    # with dimension dumps) — for §9.8.38 regression confirmation. extract_text mode="body"
    # gives unclassified prose; "default" adds classified.units too.
    legacy_modes = {"legacy_body": "body", "legacy_default": "default"}
    if text_mode in legacy_modes:
        import phase_classify as _pc
        legacy_dir = paths.ROOT / "legacy_signals_pre_2026_06_05"
        ex_mode = legacy_modes[text_mode]
        for d in sorted(legacy_dir.iterdir()):
            if not d.is_dir():
                continue
            f = d / "signal.json"
            if not f.exists():
                continue
            try:
                doc = _pc.extract_text(json.load(open(f)), mode=ex_mode)
            except Exception:
                continue
            stems.append(d.name)
            docs.append(doc)
            phase_sets.append(stem_phases(d.name))
        return stems, docs, phase_sets

    sig_file = "signal_allpages.json" if text_mode == "classified_allpages" else "signal.json"

    for d in sorted(paths.TEXT_PIPE.iterdir()):
        if not d.is_dir():
            continue
        stem = d.name

        if use_classified:
            f = d / sig_file
            if not f.exists():
                continue
            try:
                doc = classified_fingerprint(json.load(open(f)))
            except Exception:
                continue
        else:
            rp = d / "raw_pages.json"
            if not rp.exists():
                continue
            try:
                doc = raw_fingerprint(json.load(open(rp)), mode=text_mode)
            except Exception:
                continue

        phases = stem_phases(stem)
        stems.append(stem)
        docs.append(doc)
        phase_sets.append(phases)

    return stems, docs, phase_sets


# ============================================================
# FEATURES
# ============================================================

def build_features(docs_tr, docs_va, docs_te, feature_type="wordchar"):
    from scipy.sparse import hstack

    if feature_type == "unigram":
        vec = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-_]+\b",
            min_df=2, sublinear_tf=True,
        )
        X_tr = vec.fit_transform(docs_tr)
        X_va = vec.transform(docs_va)
        X_te = vec.transform(docs_te)
        return X_tr, X_va, X_te, X_tr.shape[1]

    elif feature_type == "bigram":
        vec = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-_]+\b",
            ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        )
        X_tr = vec.fit_transform(docs_tr)
        X_va = vec.transform(docs_va)
        X_te = vec.transform(docs_te)
        return X_tr, X_va, X_te, X_tr.shape[1]

    else:  # wordchar
        wv = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-_]+\b",
            ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        )
        cv = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 4),
            min_df=3, sublinear_tf=True,
        )
        Xw_tr = wv.fit_transform(docs_tr); Xw_va = wv.transform(docs_va); Xw_te = wv.transform(docs_te)
        Xc_tr = cv.fit_transform(docs_tr); Xc_va = cv.transform(docs_va); Xc_te = cv.transform(docs_te)
        X_tr = hstack([Xw_tr, Xc_tr], format="csr")
        X_va = hstack([Xw_va, Xc_va], format="csr")
        X_te = hstack([Xw_te, Xc_te], format="csr")
        return X_tr, X_va, X_te, X_tr.shape[1]


# ============================================================
# MODELS
# ============================================================

class PlainMLP(nn.Module):
    def __init__(self, in_d, hidden, n_out, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, n_out),
        )
    def forward(self, x): return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, d, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d, d), nn.LayerNorm(d),
        )
    def forward(self, x): return F.gelu(x + self.net(x))


class ResBlockBN(nn.Module):
    """ResBlock with batch norm — more stable on noisy/sparse raw features."""
    def __init__(self, d, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d, d), nn.BatchNorm1d(d),
        )
    def forward(self, x): return F.gelu(x + self.net(x))


class ResMLP(nn.Module):
    def __init__(self, in_d, hidden, n_out, n_blocks=4, drop=0.3, bn=False):
        super().__init__()
        Block = ResBlockBN if bn else ResBlock
        self.stem = nn.Sequential(nn.Linear(in_d, hidden), nn.LayerNorm(hidden),
                                  nn.GELU(), nn.Dropout(drop))
        self.blocks = nn.ModuleList([Block(hidden, drop) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, n_out)
    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks: x = b(x)
        return self.head(x)


class GEGLU_MLP(nn.Module):
    def __init__(self, in_d, hidden, n_out, drop=0.3):
        super().__init__()
        self.stem = nn.Linear(in_d, hidden * 2)
        self.drop  = nn.Dropout(drop)
        self.ln    = nn.LayerNorm(hidden)
        self.mid   = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.Dropout(drop))
        self.ln2   = nn.LayerNorm(hidden)
        self.head  = nn.Linear(hidden, n_out)
    def _geglu(self, x):
        a, b = self.stem(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)
    def _geglu2(self, x):
        a, b = self.mid(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)
    def forward(self, x):
        x = self.drop(self._geglu(x))
        x = self.ln(x)
        x = self.ln2(x + self.drop(self._geglu2(x)))
        return self.head(x)


class SelfAttnMLP(nn.Module):
    def __init__(self, in_d, hidden, n_out, n_heads=4, drop=0.3):
        super().__init__()
        self.proj  = nn.Linear(in_d, hidden)
        self.attn  = nn.MultiheadAttention(hidden, n_heads, dropout=drop, batch_first=True)
        self.ln1   = nn.LayerNorm(hidden)
        self.ff    = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                   nn.Dropout(drop), nn.Linear(hidden * 2, hidden))
        self.ln2   = nn.LayerNorm(hidden)
        self.head  = nn.Linear(hidden, n_out)
        self.drop  = nn.Dropout(drop)
    def forward(self, x):
        x = F.gelu(self.proj(x)).unsqueeze(1)
        a, _ = self.attn(x, x, x)
        x = self.ln1(x + self.drop(a)).squeeze(1)
        x = self.ln2(x + self.drop(self.ff(x)))
        return self.head(x)


class GatedResMLP(nn.Module):
    """ResMLP with a learned per-feature diagonal sigmoid gate — designed for large-vocab
    raw/lc text where TF-IDF contains many low-signal terms.

    Gate is DIAGONAL (one scalar weight per input dimension, no cross-term matrix).
    Cost: in_d parameters (not in_d²), so it scales to 40K-dim TF-IDF without OOM.
    The gate learns to suppress noise dimensions (dates, names, zone chars) before the
    main projection, effectively doing learned feature selection end-to-end.
    """
    def __init__(self, in_d, hidden, n_out, n_blocks=8, drop=0.3):
        super().__init__()
        self.log_gate = nn.Parameter(torch.zeros(in_d))   # init=0 → sigmoid=0.5 (neutral)
        self.stem     = nn.Sequential(nn.Linear(in_d, hidden), nn.LayerNorm(hidden),
                                      nn.GELU(), nn.Dropout(drop))
        self.blocks   = nn.ModuleList([ResBlock(hidden, drop) for _ in range(n_blocks)])
        self.head     = nn.Linear(hidden, n_out)
    def forward(self, x):
        gate = torch.sigmoid(self.log_gate)    # (in_d,) — learned per-feature weight
        x = x * gate                           # element-wise: suppress noise dims
        x = self.stem(x)
        for b in self.blocks: x = b(x)
        return self.head(x)


def build_model(name, in_d, n_out):
    if name == "mlp":
        return PlainMLP(in_d, 512, n_out, drop=0.3)
    elif name == "resmlp4":
        return ResMLP(in_d, 512, n_out, n_blocks=4, drop=0.3)
    elif name == "resmlp8":
        return ResMLP(in_d, 1024, n_out, n_blocks=8, drop=0.3)
    elif name == "geglu":
        return GEGLU_MLP(in_d, 512, n_out, drop=0.3)
    elif name == "selfattn":
        return SelfAttnMLP(in_d, 512, n_out, n_heads=4, drop=0.3)
    elif name == "resmlp_bn":
        return ResMLP(in_d, 1024, n_out, n_blocks=8, drop=0.3, bn=True)
    elif name == "resmlp_gate":
        return GatedResMLP(in_d, 1024, n_out, n_blocks=8, drop=0.3)
    else:
        raise ValueError(f"Unknown model: {name}")


# ============================================================
# TRAINING
# ============================================================

class FocalBCE(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none")
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


def iterative_split(Y, ratios=(0.70, 0.15, 0.15), seed=42):
    """Iterative multi-label stratification (Szymanski & Kajdanowicz 2017).
    Processes least-frequent labels first; assigns each sample to the split
    that most needs it. Validated in phase_neural_v2.py — produces correct proportions.
    """
    rng   = np.random.default_rng(seed)
    n     = len(Y)
    sizes = [int(r * n) for r in ratios]
    sizes[0] = n - sizes[1] - sizes[2]     # absorb rounding into train

    assignment = np.full(n, -1, dtype=int)
    desired    = np.array(sizes, dtype=float)
    counts     = np.zeros(3, dtype=float)

    label_freq  = Y.sum(0)
    label_order = np.argsort(label_freq)    # rarest first

    for j in label_order:
        unassigned_mask = (assignment == -1) & (Y[:, j] == 1)
        unassigned_idx  = np.where(unassigned_mask)[0]
        if len(unassigned_idx) == 0:
            continue
        rng.shuffle(unassigned_idx)

        label_desired = label_freq[j] * np.array(ratios)
        label_counts  = np.array([Y[assignment == k, j].sum() for k in range(3)])

        for i in unassigned_idx:
            need  = label_desired - label_counts
            avail = desired - counts
            score = np.where(avail > 0, need / (label_desired + 1e-9), -1e9)
            split = int(np.argmax(score))
            assignment[i] = split
            counts[split]      += 1
            label_counts[split] += 1

    remaining = np.where(assignment == -1)[0]
    rng.shuffle(remaining)
    for i in remaining:
        split = int(np.argmax(desired - counts))
        assignment[i] = split
        counts[split] += 1

    return (np.where(assignment == 0)[0],
            np.where(assignment == 1)[0],
            np.where(assignment == 2)[0])


def to_Y(phase_sets, indices, classes):
    c2i = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(indices), len(classes)), dtype=np.float32)
    for row, i in enumerate(indices):
        for ph in phase_sets[i]:
            if ph in c2i:
                Y[row, c2i[ph]] = 1.0
    return Y


def run_one(X_tr, Y_tr, X_va, Y_va, X_te, Y_te, model_name, epochs, device):
    n_out = Y_tr.shape[1]
    in_d  = X_tr.shape[1]

    # Pos weights
    pos = Y_tr.sum(0)
    neg = Y_tr.shape[0] - pos
    pw  = torch.tensor(np.where(pos > 0, neg / np.maximum(pos, 1), 1.0),
                       dtype=torch.float32).to(device)

    model = build_model(model_name, in_d, n_out).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # was 3e-4 (§9.8.38: match v2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = FocalBCE(gamma=2.0, pos_weight=pw)

    Xtr_t = torch.tensor(X_tr.toarray(), dtype=torch.float32).to(device)
    Ytr_t = torch.tensor(Y_tr, dtype=torch.float32).to(device)
    Xva_t = torch.tensor(X_va.toarray(), dtype=torch.float32).to(device)

    best_f1, best_state = -1.0, None
    warmup_ep = max(5, epochs // 10)
    BATCH = 256                                    # mini-batch SGD (was full-batch — §9.8.38)
    n_tr  = Xtr_t.shape[0]

    for ep in range(1, epochs + 1):
        if ep <= warmup_ep:
            for pg in opt.param_groups:
                pg["lr"] = 1e-3 * ep / warmup_ep
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, BATCH):
            bi = perm[i:i + BATCH]
            opt.zero_grad()
            crit(model(Xtr_t[bi]), Ytr_t[bi]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if ep > warmup_ep:
            sched.step()

        if ep % (epochs // 5) == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                logits_va = model(Xva_t).cpu().numpy()
            preds_va = (logits_va > 0.0).astype(int)
            f1s = []
            for j in range(n_out):
                ap = int(Y_va[:, j].sum())
                if ap == 0:
                    continue
                tp = int(((preds_va[:, j] == 1) & (Y_va[:, j] == 1)).sum())
                pp = int(preds_va[:, j].sum())
                p  = tp / pp if pp > 0 else 0.0
                r  = tp / ap
                f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
            vf1 = float(np.mean(f1s)) if f1s else 0.0
            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    Xte_t = torch.tensor(X_te.toarray(), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits_te = model(Xte_t).cpu().numpy()
    preds_te = (logits_te > 0.0).astype(int)

    # Metrics
    macro_f1s, micro_tp, micro_pp, micro_ap = [], 0, 0, 0
    exact, jac_sum = 0, 0.0
    offby = [0, 0, 0, 0, 0]  # 0,1,2,3,4+

    for i in range(len(Y_te)):
        pred = set(np.where(preds_te[i])[0])
        truth = set(np.where(Y_te[i])[0])
        if not truth:
            continue
        tp = len(pred & truth)
        micro_tp += tp; micro_pp += len(pred); micro_ap += len(truth)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(truth)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        macro_f1s.append(f1)
        sym_diff = len(pred.symmetric_difference(truth))
        exact += int(sym_diff == 0)
        for k, k_val in enumerate([0, 1, 2, 3]):
            if sym_diff <= k_val:
                offby[k] += 1
        if sym_diff > 3:
            offby[4] += 1
        union = len(pred | truth)
        jac_sum += tp / union if union > 0 else 0.0

    n = len(macro_f1s)
    macro = float(np.mean(macro_f1s)) if macro_f1s else 0.0
    micro = micro_tp / micro_pp if micro_pp > 0 else 0.0
    ex  = exact / n if n > 0 else 0.0
    jac = jac_sum / n if n > 0 else 0.0
    ob = [x / n if n > 0 else 0.0 for x in offby]

    return dict(macro=macro, micro=micro, exact=ex, jaccard=jac,
                offby0=ob[0], offby1=ob[1], offby2=ob[2], offby3=ob[3], offby4=ob[4],
                n=n, val_f1=best_f1, preds=preds_te)


# ============================================================
# MAIN
# ============================================================

ALL_TEXTS  = ["raw_p1", "raw_allpages", "raw_tb1_bodyall", "raw_tb2_bodyall",
              "lc_p1", "lc_allpages", "lc_tb1_bodyall", "lc_tb2_bodyall",
              "classified_p1", "classified_allpages"]
ALL_FEATS  = ["unigram", "bigram", "wordchar"]
ALL_MODELS = ["mlp", "resmlp4", "resmlp8", "geglu", "selfattn", "resmlp_bn", "resmlp_gate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts",   nargs="+", default=ALL_TEXTS,
                    choices=ALL_TEXTS + ["legacy_body", "legacy_default"])
    ap.add_argument("--features",nargs="+", default=["wordchar"],choices=ALL_FEATS)
    ap.add_argument("--models",  nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--epochs",  type=int, default=120)
    ap.add_argument("--quick",   action="store_true",
                    help="80 epochs, wordchar only, resmlp8+resmlp_bn only")
    ap.add_argument("--full",    action="store_true",
                    help="All features × all models (slow)")
    args = ap.parse_args()

    if args.quick:
        args.epochs   = 80
        args.features = ["wordchar"]
        args.models   = ["resmlp8", "resmlp_bn", "resmlp_gate"]

    if args.full:
        args.features = ALL_FEATS
        args.models   = ALL_MODELS

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Texts: {args.texts}")
    print(f"Features: {args.features}")
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs}")

    # --- Build iterative multi-label stratified split (same method as phase_neural_v2.py) ---
    # Uses ALL stems with phase truth; iterative stratification balances rare label combos.
    erp = ErpTruth()
    all_phase_counter = collections.Counter()
    all_stems_ordered = []
    for d in sorted(paths.TEXT_PIPE.iterdir()):
        if not d.is_dir(): continue
        stem = d.name
        phases = stem_phases(stem)
        if not phases: continue
        all_stems_ordered.append(stem)
        for ph in phases:
            all_phase_counter[ph] += 1

    MIN_COUNT = 20   # match phase_neural_v2.py (was >=2 → 185 classes; inflated exact-match difficulty)
    classes = sorted(ph for ph, c in all_phase_counter.items() if c >= MIN_COUNT)
    c2i = {c: i for i, c in enumerate(classes)}
    print(f"Classes (phases, freq>={MIN_COUNT}): {len(classes)}")

    # Build Y for all stems with truth
    Y_all = np.zeros((len(all_stems_ordered), len(classes)), dtype=np.float32)
    for i, stem in enumerate(all_stems_ordered):
        for ph in stem_phases(stem):
            if ph in c2i: Y_all[i, c2i[ph]] = 1.0

    tr_v, va_v, te_v = iterative_split(Y_all, ratios=(0.70, 0.15, 0.15), seed=42)
    print(f"Split (iterative stratified): train={len(tr_v)} val={len(va_v)} test={len(te_v)}")

    stem_to_split = {}
    for i in tr_v: stem_to_split[all_stems_ordered[i]] = "train"
    for i in va_v: stem_to_split[all_stems_ordered[i]] = "val"
    for i in te_v: stem_to_split[all_stems_ordered[i]] = "test"

    results = []
    total_runs = len(args.texts) * len(args.features) * len(args.models)
    run_n = 0

    for text_mode in args.texts:
        print(f"\n{'='*70}")
        print(f"TEXT MODE: {text_mode}")
        print(f"{'='*70}")

        # Build corpus for this text mode
        stems, docs, phase_sets = build_corpus(text_mode)
        if not stems:
            print(f"  WARNING: no stems loaded for {text_mode}, skipping")
            continue

        # Align to canonical split
        tr_docs, va_docs, te_docs = [], [], []
        tr_Y, va_Y, te_Y = [], [], []
        for stem, doc, pset in zip(stems, docs, phase_sets):
            split = stem_to_split.get(stem)
            if split is None:
                continue
            row = np.zeros(len(classes), dtype=np.float32)
            for ph in pset:
                if ph in c2i: row[c2i[ph]] = 1.0
            if split == "train":
                tr_docs.append(doc); tr_Y.append(row)
            elif split == "val":
                va_docs.append(doc); va_Y.append(row)
            else:
                te_docs.append(doc); te_Y.append(row)

        Y_tr = np.array(tr_Y); Y_va = np.array(va_Y); Y_te = np.array(te_Y)
        print(f"  Aligned: train={len(tr_docs)} val={len(va_docs)} test={len(te_docs)}")

        for feat in args.features:
            print(f"\n  Feature: {feat}")
            X_tr, X_va, X_te, feat_d = build_features(tr_docs, va_docs, te_docs, feat)
            print(f"    TF-IDF dim: {feat_d}")

            for model_name in args.models:
                run_n += 1
                t0 = time.time()
                print(f"    [{run_n}/{total_runs}] {model_name}...", end="", flush=True)
                try:
                    r = run_one(X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
                                model_name, args.epochs, device)
                    elapsed = time.time() - t0
                    print(f"  Macro={r['macro']:.3f}  Exact={r['exact']:.3f}"
                          f"  ±1={r['offby1']:.3f}  ±2={r['offby2']:.3f}"
                          f"  Jac={r['jaccard']:.3f}  ({elapsed:.0f}s)")
                    results.append(dict(text=text_mode, feat=feat, model=model_name,
                                        feat_d=feat_d, **r, elapsed=elapsed))
                except Exception as exc:
                    print(f"  ERROR: {exc}")
                    results.append(dict(text=text_mode, feat=feat, model=model_name,
                                        feat_d=0, macro=0.0, error=str(exc)))

    # --- Save results ---
    out = ROOT / "allpages_review" / "phase_mega_sweep.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved {len(results)} results → {out}")

    # --- Print summary table ---
    valid = [r for r in results if r.get("macro", 0) > 0]
    if not valid:
        print("No valid results.")
        return

    valid.sort(key=lambda r: -r["macro"])

    print("\n" + "="*100)
    print(f"{'Text mode':<22} {'Feat':<10} {'Model':<12} {'Macro':>6} {'Exact':>6}"
          f" {'±1':>6} {'±2':>6} {'±3':>6} {'Jac':>6} {'dim':>7}")
    print("="*100)
    for r in valid:
        print(f"{r['text']:<22} {r['feat']:<10} {r['model']:<12}"
              f" {r['macro']:>6.3f} {r['exact']:>6.3f}"
              f" {r.get('offby1',0):>6.3f} {r.get('offby2',0):>6.3f}"
              f" {r.get('offby3',0):>6.3f} {r.get('jaccard',0):>6.3f}"
              f" {r.get('feat_d',0):>7}")

    best = valid[0]
    print(f"\n★ BEST: {best['text']} | {best['feat']} | {best['model']}"
          f" → Macro={best['macro']:.3f}  Exact={best['exact']:.3f}"
          f"  ±1={best.get('offby1',0):.3f}  ±2={best.get('offby2',0):.3f}")

    # Grouped best-per-text
    print("\n--- Best Macro per text mode ---")
    by_text = collections.defaultdict(list)
    for r in valid:
        by_text[r["text"]].append(r)
    for tm in ALL_TEXTS:
        if tm not in by_text:
            continue
        b = max(by_text[tm], key=lambda r: r["macro"])
        print(f"  {tm:<22}  Macro={b['macro']:.3f}  Exact={b['exact']:.3f}"
              f"  ±1={b.get('offby1',0):.3f}  [{b['feat']}/{b['model']}]")


if __name__ == "__main__":
    main()
