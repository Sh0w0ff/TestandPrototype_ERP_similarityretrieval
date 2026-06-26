"""iam_pool.py — TRAINABLE IAM-lite over (frozen) per-view DINOv2 embeddings, trained with ArcFace
on part-type. Learns which views are DISCRIMINATIVE. Compares three aggregators:
  mean    : mean-pool + ArcFace (no attention)
  gated   : gated-attention pool (views scored independently)
  selfattn: self-attention over the view SET (cross-view interaction) -> gated pool  [the real IAM idea]
Improvements for better learning: fixed seed, mini-batch SGD, VIEW-DROPOUT regularisation, cosine LR.
Frozen features + tiny head -> CPU. Renders learned attention on a demo drawing.

Run: /opt/anaconda3/envs/fyp/bin/python scripts/part1/iam_pool.py [--viz STEM]
"""
import os, sys, json, collections
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import paths
from erp_truth import ErpTruth
torch.manual_seed(0); np.random.seed(0)
DEVICE = "cpu"; MAXV = 8
VIZ = sys.argv[sys.argv.index("--viz")+1] if "--viz" in sys.argv else "3AUA0000038918"


def load_index():
    E = np.load(paths.VISUAL_INDEX / "embeddings.npy").astype(np.float32)
    man = json.load(open(paths.VISUAL_INDEX / "manifest.json"))
    by = collections.defaultdict(list)
    for i, m in enumerate(man):
        by[m["stem"]].append((i, m.get("area_frac", 0.0), m.get("bbox_xywh"), m.get("view")))
    return E, by


class ArcHead(nn.Module):
    def __init__(self, d, C, s=20.0, m=0.4):
        super().__init__(); self.W = nn.Parameter(torch.empty(C, d)); nn.init.xavier_uniform_(self.W)
        self.s, self.m, self.C = s, m, C
    def logits(self, z): return z @ F.normalize(self.W, dim=1).T
    def loss(self, z, y):
        cos = self.logits(z).clamp(-1+1e-6, 1-1e-6); th = torch.acos(cos)
        oh = F.one_hot(y, self.C).float()
        return F.cross_entropy(self.s * (oh*torch.cos(th+self.m) + (1-oh)*cos), y)


class Pool(nn.Module):
    """kind: 'mean' | 'gated' | 'selfattn'. Returns (z, alpha)."""
    def __init__(self, kind, d=384, h=128):
        super().__init__(); self.kind = kind
        if kind == "selfattn":
            self.enc = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=512,
                                                  dropout=0.1, batch_first=True, norm_first=True)
        if kind in ("gated", "selfattn"):
            self.V = nn.Linear(d, h); self.U = nn.Linear(d, h); self.w = nn.Linear(h, 1)
    def forward(self, X, mask):
        if self.kind == "mean":
            z = (X * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            return F.normalize(z, dim=1), mask.float() / mask.sum(1, keepdim=True)
        h = self.enc(X, src_key_padding_mask=~mask) if self.kind == "selfattn" else X
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = a.masked_fill(~mask, -1e9); alpha = torch.softmax(a, 1)
        return F.normalize((alpha.unsqueeze(-1) * h).sum(1), dim=1), alpha


def build(by, E, classes_min=10):
    erp = ErpTruth(); pt = {s: erp.stem_to_parttype(s) for s in by}
    freq = collections.Counter(p for p in pt.values() if p)
    classes = sorted(c for c, n in freq.items() if n >= classes_min); c2i = {c: i for i, c in enumerate(classes)}
    items = []
    for s, lst in by.items():
        if pt[s] not in c2i: continue
        lst = sorted(lst, key=lambda t: -t[1])[:MAXV]
        items.append((s, np.stack([E[i] for i, *_ in lst]), c2i[pt[s]]))
    return items, classes


def pad(batch, drop=0.0):
    B = len(batch); d = batch[0][1].shape[1]
    X = np.zeros((B, MAXV, d), np.float32); mask = np.zeros((B, MAXV), bool); y = np.zeros(B, np.int64)
    for b, (_, v, lab) in enumerate(batch):
        nv = len(v); X[b, :nv] = v; mask[b, :nv] = True; y[b] = lab
        if drop > 0 and nv > 2:                      # view-dropout: randomly hide some views
            for j in range(nv):
                if np.random.rand() < drop and mask[b].sum() > 2: mask[b, j] = False
    return torch.tensor(X), torch.tensor(mask), torch.tensor(y)


def train_eval(kind, tr, te, classes, epochs=200, bs=64, drop=0.25):
    torch.manual_seed(0)
    pool = Pool(kind); head = ArcHead(384, len(classes))
    lr = 3e-4 if kind == "selfattn" else 1e-3        # transformer needs a gentler lr
    opt = torch.optim.Adam(list(pool.parameters())+list(head.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    params = list(pool.parameters())+list(head.parameters())
    for ep in range(epochs):
        pool.train(); head.train(); order = np.random.permutation(len(tr))
        for i in range(0, len(tr), bs):
            batch = [tr[j] for j in order[i:i+bs]]
            X, mask, y = pad(batch, drop=drop)
            opt.zero_grad(); z, _ = pool(X, mask); head.loss(z, y).backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sched.step()
    pool.eval(); head.eval()
    with torch.no_grad():
        X, mask, y = pad(te); z, _ = pool(X, mask)
        acc = float((head.logits(z).argmax(1) == y).float().mean())
    return pool, head, acc


def main():
    E, by = load_index(); items, classes = build(by, E)
    idx = np.random.permutation(len(items)); cut = int(0.7*len(items))
    tr = [items[i] for i in idx[:cut]]; te = [items[i] for i in idx[cut:]]
    print(f"{len(items)} drawings, {len(classes)} part-types, train={len(tr)} test={len(te)}\n")
    print(f"  {'aggregator':10s}  part-type acc")
    best = None
    for kind in ["mean", "gated", "selfattn"]:
        pool, head, acc = train_eval(kind, tr, te, classes)
        print(f"  {kind:10s}  {acc:.3f}")
        if kind == "selfattn": best = pool
    # viz learned attention (selfattn) on demo drawing
    vi = next((it for it in items if it[0] == VIZ), None)
    if vi and best:
        with torch.no_grad():
            X = torch.tensor(vi[1][None]); mask = torch.ones(1, len(vi[1]), dtype=torch.bool)
            _, alpha = best(X, mask)
        render(VIZ, sorted(by[VIZ], key=lambda t: -t[1])[:MAXV], alpha.squeeze(0).numpy())


def render(stem, lst, alpha):
    import cv2
    sd = paths.VISUAL_PIPE / stem
    H, W = json.loads((sd/"sam_views/sam_masks.json").read_text())["image_hw"]
    sup = cv2.imread(str(sd/"polygon_removed_super.png"))
    if sup.shape[:2] != (H, W): sup = cv2.resize(sup, (W, H))
    cell, pad_, n = 200, 40, len(lst)
    canvas = np.full((cell+pad_, cell*n, 3), 255, np.uint8)
    for i, (_, area, bbox, vi) in enumerate(lst):
        x, y, w, h = bbox; crop = sup[max(0,y):y+h, max(0,x):x+w]
        if crop.size: canvas[pad_:pad_+cell, i*cell:(i+1)*cell] = cv2.resize(crop, (cell, cell))
        cv2.putText(canvas, f"v{vi}: {alpha[i]*100:.0f}%", (i*cell+4, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,180), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (i*cell+4, pad_+cell-8),
                      (i*cell+4+int((cell-8)*alpha[i]/max(alpha.max(),1e-9)), pad_+cell-2), (0,140,200), -1)
    out = ROOT/"allpages_review"/f"iam_trained_{stem}.png"; cv2.imwrite(str(out), canvas)
    print(f"\n  selfattn learned weights {dict((f'v{lst[i][3]}', round(float(alpha[i]),3)) for i in range(n))}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
