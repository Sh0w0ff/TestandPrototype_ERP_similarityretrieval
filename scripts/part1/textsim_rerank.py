"""textsim_rerank.py — does an UNSUPERVISED text-similarity model rerank the visual candidate pool
so the human-marked similars rise? Same fixed pools + IR metrics as text_rerank_eval, but the text
scorers use the DECOUPLED cleaned corpus (textsim) and dedicated similarity models:
  tfidf_clean   TF-IDF cosine over the cleaned text (IDF fit on the whole cleaned corpus)
  minilm_base   all-MiniLM-L6-v2 zero-shot over cleaned text (the SimCSE base, un-adapted)
  simcse        our unsupervised SimCSE domain-adapted model (models/simcse_drawing)
plus the visual mean baseline and RRF fusions of mean+each text model.

Run: /opt/anaconda3/envs/fyp/bin/python scripts/part1/textsim_rerank.py [score_dir ...]
"""
import os, sys, json, re, csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "part1"))
import paths, iam_pool as IP, textsim

DIRS = [a for a in sys.argv[1:] if not a.startswith("-")] or ["score2"]
SIMCSE_DIR = ROOT / "models" / "simcse_drawing"


def load_dirs(dirs):
    queries = []
    for d in dirs:
        D = ROOT / "allpages_review" / d
        key = json.load(open(D / "_key.json")); hits = {}
        for row in csv.DictReader(open(D / "_judgments.csv")):
            h = set(re.findall(r"[A-Z]", (row.get("hit_letters(fill: e.g. A;C)") or row.get("hit_letters") or "").upper()))
            hits[row["query_id"]] = h
        for q, k in key.items():
            L = k["letters"]
            queries.append((f"{d}:{q}", k["query"], list(L.values()),
                            {L[h] for h in hits.get(q, set()) if h in L}))
    return queries


def ir(order, H):
    if not H: return None
    succ1 = 1.0 if order[0] in H else 0.0
    rec5 = len([s for s in order[:5] if s in H]) / len(H)
    nh = ap = 0
    for i, s in enumerate(order):
        if s in H: nh += 1; ap += nh / (i + 1)
    apv = ap / min(len(H), len(order))
    mrank = np.mean([order.index(s) + 1 for s in H if s in order])
    return succ1, rec5, apv, mrank


def rank_by_sim(simfn, q, cands):
    sc = np.array([simfn(q, c) for c in cands])
    return [cands[i] for i in np.argsort(-sc)]


def rrf(orders, k=60):
    items = orders[0]; s = {it: 0.0 for it in items}
    for o in orders:
        for r, it in enumerate(o): s[it] += 1.0 / (k + r + 1)
    return sorted(items, key=lambda it: -s[it])


def main():
    queries = load_dirs(DIRS)
    need = sorted({q for _, q, _, _ in queries} | {c for _, _, cs, _ in queries for c in cs})

    # visual mean baseline
    E, by = IP.load_index(); items, _ = IP.build(by, E); vecs = {s: v for s, v, _ in items}
    MN = {s: (vecs[s].mean(0) / (np.linalg.norm(vecs[s].mean(0)) + 1e-9)) for s in vecs}
    def s_mean(a, b):
        return float(MN[a] @ MN[b]) if a in MN and b in MN else -1.0

    # cleaned text
    corpus = textsim.build_corpus()
    txt = {s: corpus.get(s, "") or " " for s in need}

    # tfidf on cleaned corpus
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-_]+\b",
                          min_df=2, sublinear_tf=True).fit([corpus[s] for s in corpus])
    Xt = {s: vec.transform([txt[s]]) for s in need}
    def s_tfidf(a, b):
        return float(cosine_similarity(Xt[a], Xt[b])[0, 0])

    scorers = {"mean": s_mean, "tfidf_clean": s_tfidf}

    # dense models
    from sentence_transformers import SentenceTransformer
    def add_dense(name, model_path):
        m = SentenceTransformer(str(model_path), device="cpu")
        emb = {s: m.encode(txt[s], normalize_embeddings=True) for s in need}
        scorers[name] = (lambda a, b, emb=emb: float(emb[a] @ emb[b]))
    add_dense("minilm_base", "all-MiniLM-L6-v2")
    if SIMCSE_DIR.exists():
        add_dense("simcse", SIMCSE_DIR)
    else:
        print(f"[warn] {SIMCSE_DIR} missing — train_simcse.py not finished; skipping simcse")

    # evaluate
    cache = {}; rows = {n: [] for n in scorers}
    for qid, q, cands, H in queries:
        if not H: continue
        for n, fn in scorers.items():
            o = rank_by_sim(fn, q, cands); cache[(qid, n)] = o; rows[n].append(ir(o, H))
    fuses = {f"fuse_mean+{n}": ("mean", n) for n in scorers if n != "mean"}
    for fn, (a, b) in fuses.items():
        rows[fn] = []
        for qid, q, cands, H in queries:
            if not H: continue
            rows[fn].append(ir(rrf([cache[(qid, a)], cache[(qid, b)]]), H))

    n = sum(1 for _, _, _, H in queries if H)
    print(f"\nDECOUPLED text-sim rerank of the FIXED visual pool — {DIRS}  (n={n})\n")
    print(f"  {'scorer':18s}  Succ@1  Recall@5   MAP   mean-hit-rank")
    base = None
    order = ["mean", "tfidf_clean", "minilm_base"] + (["simcse"] if "simcse" in scorers else []) + list(fuses)
    for name in order:
        s1, r5, mp, mr = np.array(rows[name], float).mean(0)
        if name == "mean": base = mr
        tag = f"   (Δrank {mr - base:+.2f})" if (base is not None and name != "mean") else ""
        print(f"  {name:18s}  {s1:.3f}   {r5:.3f}    {mp:.3f}   {mr:.2f}{tag}")
    print("\n  lower mean-hit-rank = human picks sit higher after that ordering.")


if __name__ == "__main__":
    main()
