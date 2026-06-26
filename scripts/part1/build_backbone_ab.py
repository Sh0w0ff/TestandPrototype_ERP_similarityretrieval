"""build_backbone_ab.py — 3-way BLIND human A/B/C: does DINOv2 BACKBONE adaptation improve
HUMAN-perceived visual similarity? (The adapted backbones were only ever scored on automatic
metrics; never put in front of a human.)

For each query, pool the mean-pool top-5 neighbours from THREE backbones over the SAME corpus:
  frozen     visual_pipe/index            (frozen DINOv2 — the human-validated baseline)
  unsup      visual_pipe/index_unsup      (unsupervised same-drawing multi-view SSL adaptation)
  parttype   visual_pipe/index_parttype2  (part-type supervised-contrastive adaptation)
Union + de-identify (lettered, shuffled; no backbone/score shown). Reuses the score2 query set for
continuity. You mark which letters look similar; backbone_ab_eval.py maps letters->backbone and
computes HUMAN Success@5 / Recall@5 / P@1 per backbone.

Run: /opt/anaconda3/envs/fyp/bin/python scripts/part1/build_backbone_ab.py
"""
import os, sys, json, random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import numpy as np, cv2, collections
ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "allpages_review" / "score3"; OUTD.mkdir(parents=True, exist_ok=True)
CW, CH, LAB, NCOL = 720, 460, 30, 4
K = 5; MAXV = 8
random.seed(11)

BACKBONES = {
    "frozen":   ROOT / "visual_pipe" / "index",
    "unsup":    ROOT / "visual_pipe" / "index_unsup",
    "parttype": ROOT / "visual_pipe" / "index_parttype2",
}


def meanpool(index_dir):
    """per-stem mean-pool embedding (top-MAXV views by area, L2-normalised) from an index dir."""
    E = np.load(index_dir / "embeddings.npy").astype(np.float32)
    man = json.load(open(index_dir / "manifest.json"))
    by = collections.defaultdict(list)
    for i, m in enumerate(man):
        by[m["stem"]].append((i, m.get("area_frac", 0.0)))
    emb = {}
    for s, lst in by.items():
        idx = [i for i, _ in sorted(lst, key=lambda t: -t[1])[:MAXV]]
        v = E[idx].mean(0); emb[s] = v / (np.linalg.norm(v) + 1e-9)
    return emb


def sheet(stem):
    im = cv2.imread(str(ROOT / "visual_pipe" / stem / "polygon_removed_super.png"))
    c = np.full((CH, CW, 3), 255, np.uint8)
    if im is None: return c
    h, w = im.shape[:2]; s = min(CW/w, CH/h)
    im = cv2.resize(im, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    yh, xw = im.shape[:2]; c[(CH-yh)//2:(CH-yh)//2+yh, (CW-xw)//2:(CW-xw)//2+xw] = im
    return c


def topk(emb, q):
    stems = list(emb); M = np.stack([emb[s] for s in stems])
    sims = M @ emb[q]; qi = stems.index(q); sims[qi] = -2
    return [stems[j] for j in np.argsort(-sims)[:K]]


def main():
    embs = {name: meanpool(d) for name, d in BACKBONES.items()}
    print("loaded mean-pool for:", {k: len(v) for k, v in embs.items()})
    queries = [v["query"] for v in json.load(open(ROOT / "allpages_review" / "score2" / "_key.json")).values()]
    common = set.intersection(*[set(e) for e in embs.values()])
    queries = [q for q in queries if q in common]
    print(f"{len(queries)} queries (present in all 3 backbones)")

    key = {}; template = ["query_id,query_stem,candidate_letters,hit_letters(fill: e.g. A;C)"]
    for qn, q in enumerate(queries, 1):
        tops = {name: topk(embs[name], q) for name in BACKBONES}
        cands = list(dict.fromkeys([s for name in BACKBONES for s in tops[name]]))  # union, order-stable
        random.shuffle(cands)
        letters = {c: chr(65 + i) for i, c in enumerate(cands)}
        cells = [("QUERY", q)] + [(letters[c], c) for c in cands]
        rows = (len(cells) + NCOL - 1) // NCOL
        canvas = np.full((rows*(CH+LAB), NCOL*CW, 3), 255, np.uint8)
        for i, (tag, s) in enumerate(cells):
            r, cc = divmod(i, NCOL); y0 = r*(CH+LAB); x0 = cc*CW
            canvas[y0+LAB:y0+LAB+CH, x0:x0+CW] = sheet(s)
            col = (0,150,0) if tag == "QUERY" else (150,0,0)
            cv2.putText(canvas, tag, (x0+6, y0+24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
            if tag == "QUERY":
                cv2.rectangle(canvas, (x0+2, y0+LAB+2), (x0+CW-3, y0+LAB+CH-3), (0,150,0), 4)
        cv2.imwrite(str(OUTD / f"q{qn:02d}.png"), canvas)
        key[f"q{qn:02d}"] = {"query": q, "letters": {letters[c]: c for c in cands},
                             **{f"{name}_top5": [letters[c] for c in tops[name]] for name in BACKBONES}}
        template.append(f"q{qn:02d},{q[:40]},{''.join(sorted(letters.values()))},")
    (OUTD / "_key.json").write_text(json.dumps(key, indent=1))
    (OUTD / "_judgments.csv").write_text("\n".join(template) + "\n")
    print(f"wrote {len(queries)} blind sheets + _key.json + _judgments.csv to {OUTD}")
    print("→ open q01..png, fill hit_letters in _judgments.csv, then run backbone_ab_eval.py")


if __name__ == "__main__":
    main()
