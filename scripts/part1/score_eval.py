"""score_eval.py — score the human judgments (allpages_review/<dir>/_judgments.csv + _key.json).
Maps the marked hit-letters back to each method's ranked candidates and computes HUMAN-judged
P@1 / P@5 / MRR for selfattn-pool vs mean-pool. Also reports human-vs-part-type agreement.

Run: /opt/anaconda3/envs/fyp/bin/python scripts/part1/score_eval.py [dir1 dir2 ...]
  Pass one or more score-dir names (default: score). Multiple dirs are POOLED into one combined
  evaluation (query_ids are prefixed per dir to avoid collisions) — use e.g. `score score2` for the
  firmest combined n.
"""
import sys, json, re, csv
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from erp_truth import ErpTruth

DIRS = [a for a in sys.argv[1:] if not a.startswith("-")] or ["score"]


def main():
    key, hits = {}, {}
    for dname in DIRS:
        D = ROOT / "allpages_review" / dname
        pfx = "" if len(DIRS) == 1 else f"{dname}:"
        k = json.load(open(D / "_key.json"))
        for q, v in k.items():
            key[pfx + q] = v
        for row in csv.DictReader(open(D / "_judgments.csv")):
            h = set(re.findall(r"[A-Z]", (row.get("hit_letters(fill: e.g. A;C)") or row.get("hit_letters") or "").upper()))
            hits[pfx + row["query_id"]] = h
    erp = ErpTruth()

    def metrics(method):
        p1, p5, mrr = [], [], []
        for q, k in key.items():
            H = hits.get(q, set()); top = k[method]              # ranked letters
            p1.append(1.0 if (top and top[0] in H) else 0.0)
            p5.append(len([t for t in top if t in H]) / len(top))
            rr = next((1/(i+1) for i, t in enumerate(top) if t in H), 0.0); mrr.append(rr)
        return np.mean(p1), np.mean(p5), np.mean(mrr)

    print(f"Human-judged retrieval over {len(key)} queries (you marked the hits):\n")
    print(f"  {'method':10s}  P@1    P@5    MRR")
    for m in ["selfattn_top5", "mean_top5"]:
        a, b, c = metrics(m)
        print(f"  {m.replace('_top5',''):10s}  {a:.3f}  {b:.3f}  {c:.3f}")

    # per-query detail + human-vs-part-type agreement
    print(f"\n  per-query (hits / selfattn P@5 / mean P@5 / query part-type):")
    agree_same = agree_tot = 0
    for q, k in key.items():
        H = hits.get(q, set()); qpt = k["part_type"]; L = k["letters"]
        sp5 = len([t for t in k["selfattn_top5"] if t in H]) / 5
        mp5 = len([t for t in k["mean_top5"] if t in H]) / 5
        for t in H:                                              # does a human-hit share the query part-type?
            agree_tot += 1; agree_same += int(erp.stem_to_parttype(L[t]) == qpt)
        print(f"    {q}: hits={sorted(H)}  SA@5={sp5:.1f}  mean@5={mp5:.1f}  ({qpt})")
    if agree_tot:
        print(f"\n  human-hit shares query part-type: {agree_same}/{agree_tot} = {100*agree_same/agree_tot:.0f}% "
              f"(how often 'looks similar' == same part-type label)")


if __name__ == "__main__":
    main()
