"""phase_classify — Learn the mapping from drawing text to ERP production-line labels.

Answers the question: "can we predict which manufacturing work phases a drawing will route
through, purely from the words in the PDF?"

Setup
-----
Labels   : production lines from Work_Center_Basic_Data.csv joined via Work_Phases.csv.
           Only lines with >=MIN_COUNT drawings kept (drops 5 very sparse lines).
Split    : 80/20 stratified by vendor (ABB/KC) so both are represented in train and test.
Input    : frozen E5-base-v2 notes embedding of the drawing signal (384-d).
Model    : 2-layer MLP  384 -> 256 -> 128 -> n_classes  with sigmoid + BCE.
Eval     : per-class F1 + macro/micro F1 on held-out 20% test set.

Run
---
  python scripts/eval/phase_classify.py [--epochs 60] [--lr 1e-3] [--min-count 20]
"""
import argparse, csv, json, sys, collections
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))
import paths
from erp_truth import ErpTruth

MIN_COUNT  = 20          # drop production lines with fewer drawings
TEST_SIZE  = 0.20        # 80 / 20 split
BATCH      = 64
E5_MODEL   = "intfloat/e5-base-v2"
E5_PREFIX  = "passage: "
MAX_TOKENS = 256


# ---------------------------------------------------------------------------
# 1.  BUILD STEM → PRODUCTION-LINE LABELS
# ---------------------------------------------------------------------------

def load_prod_lines():
    """stem -> set of production-line codes."""
    wc = {}
    for r in csv.DictReader(open(paths.DATA_ROOT / "Work_Center_Basic_Data.csv", encoding="latin1"), delimiter=";"):
        wc[r["Work Center No"].strip()] = r["Production Line"].strip()

    item_pl = collections.defaultdict(set)
    for r in csv.DictReader(open(paths.DATA_ROOT / "Work_Phases.csv", encoding="latin1"), delimiter=";"):
        pl = wc.get(r["Work center no"].strip(), "")
        if pl:
            item_pl[r["Item"].strip()].add(pl)

    erp = ErpTruth()
    stem_pl = {}
    for d in paths.TEXT_PIPE.iterdir():
        if not d.is_dir():
            continue
        stem = d.name
        pls = set()
        for it in erp.stem_to_items(stem):
            pls |= item_pl.get(it, set())
        if pls:
            stem_pl[stem] = pls
    return stem_pl


# ---------------------------------------------------------------------------
# 2.  EXTRACT TEXT FROM SIGNAL (handles old + new schema)
# ---------------------------------------------------------------------------

def extract_text(sig: dict, mode: str = "default") -> str:
    """Pull text from signal JSON.

    mode="default"     — classified.units.text + unclassified.body.text (current behaviour)
    mode="body"        — unclassified.body only (raw prose notes, no structured fields)
    mode="fingerprint" — e1.fingerprint() string including synthetic tag tokens
    """
    if mode == "fingerprint":
        import e1
        return e1.fingerprint(sig)

    parts = []
    cl = sig.get("classified", {})
    un = sig.get("unclassified", {})

    if mode == "process-tags":
        # Only units where reason='tag' (vocab-tagged sentences with process/production signal)
        for u in cl.get("units", []):
            if u.get("reason") == "tag":
                t = u.get("text", "")
                if t and isinstance(t, str):
                    parts.append(t)
        return " ".join(parts).strip() or " "

    if mode == "default":
        for u in cl.get("units", []):
            t = u.get("text", "")
            if t and isinstance(t, str):
                parts.append(t)

    # body text — used by both modes
    for b in un.get("body", []):
        t = b.get("text", "") or ""
        if isinstance(t, str) and t:
            parts.append(t)

    # old / flat schema fallback (both modes)
    if not parts:
        for b in sig.get("body", []):
            t = b.get("text", "") if isinstance(b, dict) else str(b)
            if t:
                parts.append(t)
        tb = sig.get("title_block", {})
        for c in tb.get("cells", []):
            t = c.get("text", "") if isinstance(c, dict) else str(c)
            if t:
                parts.append(t)

    return " ".join(parts).strip() or " "


# ---------------------------------------------------------------------------
# 3a. TF-IDF ENCODER (baseline — no neural encoder)
# ---------------------------------------------------------------------------

def encode_tfidf(texts, tr_idx=None):
    """Fit on train indices only to avoid IDF leakage. Returns full matrix."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=2,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-_]+\b")
    if tr_idx is not None:
        tr_texts = [texts[i] for i in tr_idx]
        vec.fit(tr_texts)
        X = vec.transform(texts).toarray().astype(np.float32)
    else:
        X = vec.fit_transform(texts).toarray().astype(np.float32)
    print(f"  TF-IDF matrix: {X.shape}  (vocab={len(vec.vocabulary_)})", flush=True)
    return X


ENCODER_MODELS = {
    "e5":         ("intfloat/e5-base-v2",                  "passage: "),
    "minilm":     ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "scibert":    ("allenai/scibert_scivocab_uncased",       ""),
    "matscibert": ("m3rg-iitd/matscibert",                   ""),
}


# ---------------------------------------------------------------------------
# 3b. EMBED WITH FROZEN NEURAL ENCODER
# ---------------------------------------------------------------------------

def embed_texts(texts, model_key="e5", batch=32):
    model_name, prefix = ENCODER_MODELS[model_key]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(dev).eval()
    print(f"  {model_name} on {dev}", flush=True)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = [prefix + (t or " ") for t in texts[i:i + batch]]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=MAX_TOKENS, return_tensors="pt").to(dev)
            hid = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hid * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.cpu().numpy())
        if i % (batch * 10) == 0:
            print(f"  embedded {min(i+batch, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(out)


# ---------------------------------------------------------------------------
# 4.  MLP CLASSIFIER
# ---------------------------------------------------------------------------

class PhaseHead(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def train(X_tr, Y_tr, X_va, Y_va, n_cls, epochs, lr):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = PhaseHead(X_tr.shape[1], n_cls).to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    Yt = torch.tensor(Y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_va, dtype=torch.float32).to(dev)
    Yv = torch.tensor(Y_va, dtype=torch.float32).to(dev)

    best_f1, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            xb = Xt[idx[i:i+BATCH]].to(dev)
            yb = Yt[idx[i:i+BATCH]].to(dev)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        if ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                logits = model(Xv)
                preds  = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            f1 = f1_score(Yv.cpu().numpy(), preds, average="macro", zero_division=0)
            print(f"  epoch {ep:3d}/{epochs}  val macro-F1={f1:.3f}", flush=True)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# 5.  MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",    type=int,   default=60)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--min-count", type=int,   default=MIN_COUNT)
    ap.add_argument("--signal",    default="signal.json",
                    help="which signal file to read per stem")
    ap.add_argument("--text-mode", default="default",
                    choices=["default", "body", "fingerprint", "process-tags"],
                    help="default=fields+body | body=raw notes only | fingerprint=e1 synthetic tokens")
    ap.add_argument("--encoder", default="e5",
                    choices=["e5", "minilm", "scibert", "matscibert", "tfidf"],
                    help="neural encoder key or tfidf for bag-of-words baseline")
    ap.add_argument("--no-save-split", action="store_true",
                    help="skip writing phase_classify_split.json (for parallel runs)")
    args = ap.parse_args()

    print("=== Phase Classifier — ERP-supervised multi-label ===\n")

    # --- labels ---
    print("Building production-line labels...", flush=True)
    stem_pl = load_prod_lines()
    pl_counts = collections.Counter(pl for pls in stem_pl.values() for pl in pls)
    classes = sorted(pl for pl, c in pl_counts.items() if c >= args.min_count)
    cl_idx  = {pl: i for i, pl in enumerate(classes)}
    n_cls   = len(classes)
    print(f"  {len(stem_pl)} stems with labels  |  {n_cls} production-line classes (>={args.min_count} drawings)")
    print(f"  Classes: {classes}\n")

    # --- load signals ---
    print(f"Loading signals ({args.signal})...", flush=True)
    stems, texts, labels = [], [], []
    for d in sorted(paths.TEXT_PIPE.iterdir()):
        if not d.is_dir():
            continue
        stem = d.name
        if stem not in stem_pl:
            continue
        f = d / args.signal
        if not f.exists():
            continue
        try:
            sig = json.load(open(f))
        except Exception:
            continue
        pls = stem_pl[stem]
        y = np.zeros(n_cls, dtype=np.float32)
        for pl in pls:
            if pl in cl_idx:
                y[cl_idx[pl]] = 1.0
        stems.append(stem)
        texts.append(extract_text(sig, mode=args.text_mode))
        labels.append(y)

    X_raw = np.array(labels)
    print(f"  Loaded {len(stems)} stems\n")

    # --- stratified split by vendor ---
    vendors = ["ABB" if s.startswith("3A") else "KC" for s in stems]
    idx_all = np.arange(len(stems))
    tr_idx, te_idx = train_test_split(
        idx_all, test_size=TEST_SIZE, stratify=vendors, random_state=42
    )
    abb_tr = sum(1 for i in tr_idx if vendors[i] == "ABB")
    kc_tr  = sum(1 for i in tr_idx if vendors[i] == "KC")
    abb_te = sum(1 for i in te_idx if vendors[i] == "ABB")
    kc_te  = sum(1 for i in te_idx if vendors[i] == "KC")
    print(f"Split: train={len(tr_idx)} (ABB={abb_tr}, KC={kc_tr})  "
          f"test={len(te_idx)} (ABB={abb_te}, KC={kc_te})\n")

    # --- encode ---
    if args.encoder == "tfidf":
        print("Encoding with TF-IDF (fit on train only)...", flush=True)
        X_emb = encode_tfidf(texts, tr_idx=tr_idx)
    else:
        print(f"Embedding with frozen {args.encoder}...", flush=True)
        X_emb = embed_texts(texts, model_key=args.encoder)
    print(f"  Encoded matrix: {X_emb.shape}\n")

    X_tr, Y_tr = X_emb[tr_idx], X_raw[tr_idx]
    X_te, Y_te = X_emb[te_idx], X_raw[te_idx]

    # --- train ---
    print(f"Training MLP ({args.epochs} epochs, lr={args.lr})...", flush=True)
    model = train(X_tr, Y_tr, X_te, Y_te, n_cls, args.epochs, args.lr)

    # --- eval ---
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_te, dtype=torch.float32).to(dev))
        preds  = (torch.sigmoid(logits) > 0.5).cpu().numpy()

    print("\n=== TEST SET RESULTS ===\n")
    print(classification_report(Y_te, preds, target_names=classes,
                                zero_division=0, digits=3))
    macro = f1_score(Y_te, preds, average="macro",  zero_division=0)
    micro = f1_score(Y_te, preds, average="micro",  zero_division=0)
    print(f"Macro F1: {macro:.3f}   Micro F1: {micro:.3f}")

    print(f"\nencoder={args.encoder}  text-mode={args.text_mode}  signal={args.signal}")

    # save split for reproducibility
    if not args.no_save_split:
        split = {"train": [stems[i] for i in tr_idx],
                 "test":  [stems[i] for i in te_idx],
                 "classes": classes}
        out = ROOT / "allpages_review" / "phase_classify_split.json"
        json.dump(split, open(out, "w"), indent=1)
        print(f"Split saved -> {out}")


if __name__ == "__main__":
    main()
