"""backbone_ab_eval.py — score the 3-way blind backbone judgement (allpages_review/score3).
Maps marked hit-letters back to each backbone's mean-pool top-5 and reports HUMAN
Success@5 / Recall@5 / P@1 per backbone (frozen vs unsup vs parttype). Answers: does DINOv2
BACKBONE adaptation improve human-perceived visual similarity?
Run: /opt/anaconda3/envs/fyp/bin/python scripts/part1/backbone_ab_eval.py
"""
import sys, json, re, csv
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "allpages_review" / "score3"
BACKBONES = ["frozen", "unsup", "parttype"]


def main():
    key = json.load(open(D / "_key.json"))
    hits = {}
    for row in csv.DictReader(open(D / "_judgments.csv")):
        h = set(re.findall(r"[A-Z]", (row.get("hit_letters(fill: e.g. A;C)") or row.get("hit_letters") or "").upper()))
        hits[row["query_id"]] = h
    njudged = sum(1 for q in key if hits.get(q))
    print(f"3-way blind backbone retrieval — {len(key)} queries ({njudged} with >=1 hit)\n")
    print(f"  {'backbone':10s}  Succ@1  Succ@5  Recall@5")
    for b in BACKBONES:
        s1 = s5 = rec = nq = 0
        for q, k in key.items():
            H = hits.get(q, set())
            if not H: continue
            nq += 1; top = k[f"{b}_top5"]
            s1 += top[0] in H
            s5 += any(t in H for t in top)
            rec += len([t for t in top if t in H]) / len(H)
        print(f"  {b:10s}  {s1/nq:.3f}   {s5/nq:.3f}   {rec/nq:.3f}")
    # overlap: how distinct are the three top-5 sets on average?
    import itertools
    ov = {f"{a}∩{c}": [] for a, c in itertools.combinations(BACKBONES, 2)}
    for q, k in key.items():
        for a, c in itertools.combinations(BACKBONES, 2):
            A, C = set(k[f"{a}_top5"]), set(k[f"{c}_top5"])
            ov[f"{a}∩{c}"].append(len(A & C) / 5)
    print("\n  mean top-5 overlap between backbones (1.0=identical, 0=disjoint):")
    for k, v in ov.items():
        print(f"    {k}: {np.mean(v):.2f}")


if __name__ == "__main__":
    main()
