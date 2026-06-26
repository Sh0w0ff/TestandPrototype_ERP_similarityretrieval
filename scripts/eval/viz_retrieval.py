"""viz_retrieval — make the e1/fusion retrieval LEGIBLE (look at it, don't just trust the numbers).

For a handful of query drawings, show the top-K neighbours each channel retrieves, side by side:
  row 1 = TEXT (TF-IDF cosine)   row 2 = VISUAL (frozen DINOv2 view-matcher)   row 3 = FUSED (a=0.5)
Each neighbour thumbnail is bordered:
  GREEN  = shares >=1 BOM component with the query  (a CORRECT neighbour — we'd copy good ERP from it)
  RED    = shares no BOM component                  (a wrong neighbour)
and labelled  "<stem-tail>  C<n>  P<m>"  = # shared BOM components / # shared work-phases.
The query thumbnail (blue) heads each sheet. One PNG per query + a stacked contact sheet.

This is the visual companion to fuse_eval.py: you can SEE text retrieve more green than visual, and
whether fusion mixes useful neighbours in or dilutes them.

Run:  python scripts/eval/viz_retrieval.py [--k 5] [--n 6] [--alpha 0.5] [--out review/viz_retrieval]
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path("/Users/sh0w0ff/FYP")
sys.path.insert(0, str(ROOT / "scripts"))
import paths
SIG = paths.TEXT_PIPE                          # <stem>/signal.json
PAGE = paths.PREPROCESS                        # <stem>/page1.png for every stem
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from erp_truth import ErpTruth
from e1 import fingerprint
from encoder_eval import matcher_score
from fuse_eval import minmax_offdiag

THUMB_W = 240
BLUE, GREEN, RED = (40, 90, 200), (30, 160, 60), (200, 50, 50)


def thumb(stem, w=THUMB_W):
    p = PAGE / stem / "page1.png"
    im = Image.open(p).convert("RGB") if p.exists() else Image.new("RGB", (w, w), (235, 235, 235))
    im.thumbnail((w, w))
    return im


def label_strip(width, text, color):
    h = 22
    s = Image.new("RGB", (width, h), color)
    d = ImageDraw.Draw(s)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    except Exception:
        f = ImageFont.load_default()
    d.text((4, 3), text, fill=(255, 255, 255), font=f)
    return s


def carded(stem, color, caption):
    """thumbnail + colored border + caption strip, all same width."""
    t = thumb(stem)
    t = ImageOps.expand(t, border=5, fill=color)
    strip = label_strip(t.width, caption, color)
    card = Image.new("RGB", (t.width, t.height + strip.height), (255, 255, 255))
    card.paste(t, (0, 0)); card.paste(strip, (0, t.height))
    return card


def row(cards, gap=8):
    if not cards:
        return Image.new("RGB", (THUMB_W, 10), (255, 255, 255))
    h = max(c.height for c in cards)
    w = sum(c.width for c in cards) + gap * (len(cards) - 1)
    r = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for c in cards:
        r.paste(c, (x, 0)); x += c.width + gap
    return r


def build_matrices(stems, docs, emb):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-_]+\b",
                          min_df=2, sublinear_tf=True)
    text_sim = cosine_similarity(vec.fit_transform(docs))
    n = len(stems)
    vis_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            vis_sim[i, j] = vis_sim[j, i] = matcher_score(emb[stems[i]], emb[stems[j]])
    return text_sim, vis_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n", type=int, default=6, help="number of query drawings to visualise")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--emb", default=str(paths.VISUAL_INDEX))
    ap.add_argument("--out", default=str(paths.total_images()))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    embdir = Path(args.emb)
    E = np.load(embdir / "embeddings.npy")
    manifest = json.loads((embdir / "manifest.json").read_text())
    vis_stems = sorted(set(m["stem"] for m in manifest))
    vidx = {s: [i for i, m in enumerate(manifest) if m["stem"] == s] for s in vis_stems}
    emb = {s: E[vidx[s]] for s in vis_stems}
    stems = [s for s in vis_stems if (SIG / s / "signal.json").exists()]
    docs = [fingerprint(json.load(open(SIG / s / "signal.json"))) for s in stems]

    erp = ErpTruth()
    comp = [erp.stem_to_components(s) for s in stems]
    phase = [erp.stem_to_phases(s) for s in stems]

    text_sim, vis_sim = build_matrices(stems, docs, emb)
    tN, vN = minmax_offdiag(text_sim), minmax_offdiag(vis_sim)
    fused = args.alpha * tN + (1 - args.alpha) * vN
    channels = [("TEXT", tN), ("VISUAL", vN), (f"FUSED a={args.alpha}", fused)]

    queries = [i for i in range(len(stems)) if comp[i]][: args.n]
    sheets = []
    for qi in queries:
        q = stems[qi]
        green_count = {}
        blocks = [carded(q, BLUE, f"QUERY {q[-10:]}  |C|={len(comp[qi])} |P|={len(phase[qi])}")]
        for cname, sim in channels:
            order = np.argsort(-sim[qi]); order = order[order != qi][: args.k]
            cards, ng = [], 0
            for j in order:
                shared_c = len(comp[qi] & comp[j])
                shared_p = len(phase[qi] & phase[j])
                color = GREEN if shared_c > 0 else RED
                if shared_c > 0:
                    ng += 1
                cards.append(carded(stems[j], color, f"{stems[j][-8:]}  C{shared_c} P{shared_p}"))
            green_count[cname] = ng
            head = label_strip(160, f"{cname}  ({ng}/{args.k} hit)", (60, 60, 60))
            r = row(cards)
            band = Image.new("RGB", (max(r.width, head.width), r.height + head.height + 4), (255, 255, 255))
            band.paste(head, (0, 0)); band.paste(r, (0, head.height + 4))
            blocks.append(band)
        W = max(b.width for b in blocks)
        H = sum(b.height for b in blocks) + 10 * len(blocks)
        sheet = Image.new("RGB", (W + 16, H + 16), (255, 255, 255))
        y = 8
        for b in blocks:
            sheet.paste(b, (8, y)); y += b.height + 10
        f = out / f"q_{q[-12:]}.png"
        sheet.save(f)
        sheets.append(sheet)
        print(f"{q}  hits text/vis/fused = "
              f"{green_count['TEXT']}/{green_count['VISUAL']}/{green_count[f'FUSED a={args.alpha}']}  -> {f.name}")

    if sheets:
        W = max(s.width for s in sheets)
        H = sum(s.height for s in sheets) + 20 * len(sheets)
        contact = Image.new("RGB", (W, H), (245, 245, 245))
        y = 0
        for s in sheets:
            contact.paste(s, (0, y)); y += s.height + 20
        contact.save(out / "contact.png")
        print(f"\ncontact sheet -> {out/'contact.png'}")


if __name__ == "__main__":
    main()
