"""retrieval_demo.py — show the visual encoder bringing out similar parts to a QUERY drawing.
Trains the self-attention multi-view pool (on part-type), embeds every drawing, then for a query
stem renders [query | top-K neighbours] as clean rendered sheets (polygon_removed_super.png) into
allpages_review/ for human eyeballing. Cosine similarity + part-type shown for reference only.

Run: /opt/anaconda3/envs/fyp/bin/python scripts/part1/retrieval_demo.py [QUERY_STEM] [K]
"""
import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import numpy as np, torch, torch.nn as nn, cv2
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "part1"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import iam_pool as IP
from erp_truth import ErpTruth

QUERY = sys.argv[1] if len(sys.argv) > 1 else "3AUA0000038918"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 6


CW, CH = 940, 600          # wide cells (engineering sheets are landscape) -> legible


def sheet_img(stem, cw=CW, ch=CH):
    p = ROOT / "visual_pipe" / stem / "polygon_removed_super.png"
    im = cv2.imread(str(p))
    canvas = np.full((ch, cw, 3), 255, np.uint8)
    if im is None: return canvas
    h, w = im.shape[:2]; s = min(cw / w, ch / h)
    im = cv2.resize(im, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    yh, xw = im.shape[:2]; canvas[(ch-yh)//2:(ch-yh)//2+yh, (cw-xw)//2:(cw-xw)//2+xw] = im
    return canvas


def main():
    E, by = IP.load_index(); items, classes = IP.build(by, E)
    erp = ErpTruth(); pt = {s: erp.stem_to_parttype(s) for s, _, _ in items}
    cache = ROOT / "visual_pipe" / "adapted" / "iam_selfattn_emb.npz"
    if cache.exists() and "--retrain" not in sys.argv:
        z = np.load(cache, allow_pickle=True); stems = list(z["stems"]); M = z["M"]
        emb = {s: M[i] for i, s in enumerate(stems)}
        print(f"loaded cached self-attention embeddings ({len(stems)} drawings)")
    else:
        print(f"training self-attention pool on {len(items)} drawings ...")
        torch.manual_seed(0); np.random.seed(0)
        pool = IP.Pool("selfattn"); head = IP.ArcHead(384, len(classes))
        params = list(pool.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200)
        for ep in range(200):
            pool.train(); head.train(); order = np.random.permutation(len(items))
            for i in range(0, len(items), 64):
                X, m, y = IP.pad([items[j] for j in order[i:i+64]], drop=0.25)
                opt.zero_grad(); zz, _ = pool(X, m); head.loss(zz, y).backward()
                nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            sched.step()
        pool.eval(); emb = {}
        with torch.no_grad():
            for s, vecs, _ in items:
                X = torch.tensor(vecs[None]); m = torch.ones(1, len(vecs), dtype=torch.bool)
                emb[s] = pool(X, m)[0].squeeze(0).numpy()
        np.savez(cache, stems=np.array(list(emb)), M=np.stack([emb[s] for s in emb]))
    if QUERY not in emb:
        print(f"{QUERY} not in pool (needs part-type freq>=10 + views). Pick another."); return

    stems = list(emb); M = np.stack([emb[s] for s in stems])
    sims = M @ emb[QUERY]; qi = stems.index(QUERY); sims[qi] = -2
    top = np.argsort(-sims)[:K]
    print(f"\nQUERY {QUERY}  (part-type {pt.get(QUERY)})")
    for r, j in enumerate(top, 1):
        print(f"  #{r}  cos={sims[j]:.3f}  {stems[j]}  (part-type {pt.get(stems[j])})")

    # render: big wide cells in a 2-column grid; query first (green border)
    order_stems = [QUERY] + [stems[j] for j in top]
    coss = [None] + [sims[j] for j in top]
    lab = 34; ncol = 2; nrow = (len(order_stems) + ncol - 1) // ncol
    canvas = np.full((nrow*(CH+lab), ncol*CW, 3), 255, np.uint8)
    for i, s in enumerate(order_stems):
        r, c = divmod(i, ncol); y0 = r*(CH+lab); x0 = c*CW
        canvas[y0+lab:y0+lab+CH, x0:x0+CW] = sheet_img(s)
        if i == 0:
            cv2.rectangle(canvas, (x0+2, y0+lab+2), (x0+CW-3, y0+lab+CH-3), (0,160,0), 5)
            cv2.putText(canvas, f"QUERY  ({s[:40]})  part-type {pt.get(s)}", (x0+6, y0+24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,120,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, f"#{i}  cos {coss[i]:.3f}  part-type {pt.get(s)}  ({s[:34]})",
                        (x0+6, y0+24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (160,0,0), 2, cv2.LINE_AA)
    out = ROOT / "allpages_review" / f"retrieval_{QUERY}.png"
    cv2.imwrite(str(out), canvas)
    print(f"\nwrote {out}  ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main()
