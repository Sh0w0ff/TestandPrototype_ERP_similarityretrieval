"""build_score_set.py — generate a BLIND human-judgment retrieval set.

For N query drawings (stratified across part-types), pool the top-5 neighbours from the self-attention
pool AND the mean pool, shuffle + de-identify them (lettered A,B,...; no method/cosine/part-type shown),
and render a legible sheet per query into allpages_review/<out>/. You mark which letters are similar
to the query (in _judgments.csv); score_eval.py then maps letters->method and computes human P@k for
selfattn vs mean. Uses cached self-attention embeddings (run retrieval_demo.py once first).

Run: /opt/anaconda3/envs/fyp/bin/python scripts/part1/build_score_set.py [--n N] [--seed S]
                                  [--out DIRNAME] [--exclude path/to/prior/_key.json]
  --exclude  drop any stems used as queries in a prior key.json (fresh, non-overlapping confirmation set)
Default reproduces the original 12-query set in allpages_review/score/.
"""
import os, sys, json, random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "part1"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import iam_pool as IP
from erp_truth import ErpTruth


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


N        = int(_arg("--n", 12))
SEED     = int(_arg("--seed", 7))
OUT_NAME = _arg("--out", "score")
EXCLUDE  = _arg("--exclude", None)
OUTD = ROOT / "allpages_review" / OUT_NAME; OUTD.mkdir(parents=True, exist_ok=True)
CW, CH, LAB, NCOL = 760, 480, 30, 3
K = 5
random.seed(SEED)


def sheet(stem):
    im = cv2.imread(str(ROOT / "visual_pipe" / stem / "polygon_removed_super.png"))
    c = np.full((CH, CW, 3), 255, np.uint8)
    if im is None: return c
    h, w = im.shape[:2]; s = min(CW/w, CH/h); im = cv2.resize(im, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    yh, xw = im.shape[:2]; c[(CH-yh)//2:(CH-yh)//2+yh, (CW-xw)//2:(CW-xw)//2+xw] = im
    return c


def main():
    cache = ROOT / "visual_pipe" / "adapted" / "iam_selfattn_emb.npz"
    if not cache.exists():
        print("run scripts/part1/retrieval_demo.py once to build the embedding cache first."); return
    z = np.load(cache, allow_pickle=True); stems = list(z["stems"]); SA = z["M"]
    sidx = {s: i for i, s in enumerate(stems)}
    E, by = IP.load_index(); items, _ = IP.build(by, E)
    erp = ErpTruth(); pt = {s: erp.stem_to_parttype(s) for s in stems}
    # mean-pool embeddings for the SAME stems
    vecs = {s: v for s, v, _ in items}
    MN = np.stack([vecs[s].mean(0) / (np.linalg.norm(vecs[s].mean(0))+1e-9) for s in stems])
    SAn = SA / (np.linalg.norm(SA, axis=1, keepdims=True)+1e-9)

    # optionally exclude stems already used as queries in a prior set (fresh confirmation set)
    excl = set()
    if EXCLUDE:
        prior = json.load(open(EXCLUDE))
        excl = {v["query"] for v in prior.values()}
        print(f"excluding {len(excl)} prior query stems from {EXCLUDE}")

    # stratified round-robin across part-types: cover every type, then keep cycling to reach N.
    byclass = {}
    for s in stems:
        if pt[s] and s not in excl:
            byclass.setdefault(pt[s], []).append(s)
    pools = {c: random.sample(v, len(v)) for c, v in sorted(byclass.items())}  # shuffled per class
    queries, ptr = [], {c: 0 for c in pools}
    while len(queries) < N and any(ptr[c] < len(pools[c]) for c in pools):
        for c in sorted(pools):
            if ptr[c] < len(pools[c]):
                queries.append(pools[c][ptr[c]]); ptr[c] += 1
                if len(queries) >= N: break

    key = {}; template = ["query_id,query_stem,part_type,candidate_letters,hit_letters(fill: e.g. A;C)"]
    for qn, q in enumerate(queries, 1):
        qi = sidx[q]
        sa = SAn @ SAn[qi]; sa[qi] = -2; sa_top = [stems[j] for j in np.argsort(-sa)[:K]]
        mn = MN @ MN[qi]; mn[qi] = -2; mn_top = [stems[j] for j in np.argsort(-mn)[:K]]
        cands = list(dict.fromkeys(sa_top + mn_top))      # union, dedup, preserve order
        random.shuffle(cands)
        letters = {c: chr(65+i) for i, c in enumerate(cands)}   # A,B,...
        # render: query (green) + lettered candidates, BLIND
        rows = (len(cands) + 1 + NCOL - 1) // NCOL
        canvas = np.full((rows*(CH+LAB), NCOL*CW, 3), 255, np.uint8)
        cells = [("QUERY", q)] + [(letters[c], c) for c in cands]
        for i, (tag, s) in enumerate(cells):
            r, cc = divmod(i, NCOL); y0 = r*(CH+LAB); x0 = cc*CW
            canvas[y0+LAB:y0+LAB+CH, x0:x0+CW] = sheet(s)
            col = (0,150,0) if tag == "QUERY" else (150,0,0)
            cv2.putText(canvas, tag, (x0+6, y0+24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
            if tag == "QUERY":
                cv2.rectangle(canvas, (x0+2, y0+LAB+2), (x0+CW-3, y0+LAB+CH-3), (0,150,0), 4)
        cv2.imwrite(str(OUTD / f"q{qn:02d}.png"), canvas)
        key[f"q{qn:02d}"] = {"query": q, "part_type": pt[q], "letters": {letters[c]: c for c in cands},
                             "selfattn_top5": [letters[c] for c in sa_top],
                             "mean_top5": [letters[c] for c in mn_top]}
        template.append(f"q{qn:02d},{q[:40]},{pt[q]},{''.join(sorted(letters.values()))},")
    (OUTD / "_key.json").write_text(json.dumps(key, indent=1))
    (OUTD / "_judgments.csv").write_text("\n".join(template) + "\n")
    print(f"wrote {len(queries)} query sheets + _key.json + _judgments.csv to {OUTD}")
    print("→ open q01..qNN.png, then in _judgments.csv fill hit_letters (which candidates match the query).")


if __name__ == "__main__":
    main()
