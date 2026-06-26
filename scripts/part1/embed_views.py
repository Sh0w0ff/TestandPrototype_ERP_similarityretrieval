"""
embed_views.py — Visual-channel STAGE 2 go/no-go (2026-06-01).

Pipeline:
  1. For each sample stem, read the v5 view selection (sam_masks_filtered_v5.json)
     and crop each view from polygon_removed_super.png (resized to the SAM working
     space image_hw, the space the v5 bboxes live in).
  2. Line-art preprocess each crop: grayscale -> ink mask -> dilate by ~max(w,h)/224
     (so thin strokes survive the 224 downscale) -> pad to square (white) -> 224 ->
     3ch -> ImageNet norm.
  3. Embed each view with frozen DINOv2-small (CLS token, L2-normalized).
  4. Save crops (for eyeballing), per-view embeddings, and a manifest.
  5. Cross-drawing retrieval: POOLED baseline (mean view-vector cosine) vs PER-VIEW
     late-interaction MATCHER (one-to-one assignment, sum max(0,cos-tau), normalized
     by min set size). Print ranked retrieval per stem + dump matched-view montages.

This is the visual-channel GO/NO-GO: do geometrically/part-similar drawings surface
matching views and beat dissimilar ones? Decision is read BY EYE on the montages
first, then the numbers. See thesis_direction.md s9.8.15 + session resume.

Run:  /opt/anaconda3/envs/fyp/bin/python scripts/part1/embed_views.py
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]   # portable repo root (GPU-bundle friendly)
sys.path.insert(0, str(ROOT / "scripts")); import paths
SAMPLES = paths.VISUAL_PIPE                 # <stem>/sam_views, polygon_removed_super.png
OUT = paths.VISUAL_INDEX                    # embeddings.npy, manifest.json, crops/
IMG = paths.visual_images()                 # browsable match_*.png / contact sheets
OUT.mkdir(parents=True, exist_ok=True); IMG.mkdir(parents=True, exist_ok=True)
TAU = 0.5          # late-interaction floor: only count view pairs with cos > TAU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # GPU box uses CUDA; Mac falls back to CPU
torch.manual_seed(42)


# ---------- line-art preprocessing ----------
def preprocess(crop_rgb):
    """crop_rgb: HxWx3 uint8 (RGB). -> PIL 224x224 RGB, strokes dilated to survive downscale."""
    g = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    ink = (g < 200).astype(np.uint8)               # dark strokes on white
    k = max(1, int(round(max(g.shape) / 224)))     # dilate proportional to downscale factor
    if k > 1:
        ink = cv2.dilate(ink, np.ones((k, k), np.uint8))
    img = np.full(g.shape, 255, np.uint8)
    img[ink > 0] = 0                               # clean white bg, black ink
    h, w = img.shape
    s = max(h, w)
    sq = np.full((s, s), 255, np.uint8)            # pad to square, keep aspect
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    sq = cv2.resize(sq, (224, 224), interpolation=cv2.INTER_AREA)
    return Image.fromarray(np.stack([sq] * 3, -1))


# ---------- DINOv2-small encoder (frozen) ----------
def load_dino():
    from transformers import AutoModel
    from torchvision import transforms
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval().to(DEVICE)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, tf


@torch.no_grad()
def embed(model, tf, pil_img):
    x = tf(pil_img).unsqueeze(0).to(DEVICE)
    cls = model(pixel_values=x).last_hidden_state[:, 0, :].cpu().numpy().squeeze()
    return cls / (np.linalg.norm(cls) + 1e-9)


# ---------- step 1-4: embed all stems ----------
def build_embeddings():
    model, tf = load_dino()
    manifest, embs = [], []
    stems = sorted(p.name for p in SAMPLES.iterdir() if p.is_dir())
    for stem in stems:
        sd = SAMPLES / stem
        vf = sd / "sam_views/sam_masks_filtered_v5.json"
        mf = sd / "sam_views/sam_masks.json"
        sup_path = sd / "polygon_removed_super.png"
        if not (vf.exists() and mf.exists() and sup_path.exists()):
            print(f"  SKIP {stem} (missing inputs)"); continue
        H, W = json.loads(mf.read_text())["image_hw"]
        sup = cv2.imread(str(sup_path))
        if sup.shape[:2] != (H, W):
            sup = cv2.resize(sup, (W, H))
        sup_rgb = cv2.cvtColor(sup, cv2.COLOR_BGR2RGB)
        views = json.loads(vf.read_text())["views"]
        crop_dir = OUT / "crops" / stem[:30]
        crop_dir.mkdir(parents=True, exist_ok=True)
        for vi, v in enumerate(views):
            x, y, w, h = v["bbox_xywh"]
            crop = sup_rgb[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                continue
            pil = preprocess(crop)
            pil.save(crop_dir / f"v{vi}.png")
            embs.append(embed(model, tf, pil))
            manifest.append({"stem": stem, "view": vi, "bbox_xywh": [x, y, w, h],
                             "area_frac": v.get("bbox_area_frac"),
                             "crop": str(crop_dir / f"v{vi}.png")})
        print(f"  {stem[:36]:38s} {len(views)} views")
    E = np.stack(embs)
    np.save(OUT / "embeddings.npy", E)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nEmbedded {len(E)} views across {len(set(m['stem'] for m in manifest))} stems "
          f"-> {OUT}/embeddings.npy")
    return E, manifest


# ---------- step 5: retrieval ----------
def matcher_score(EA, EB, tau=TAU):
    """one-to-one view assignment; score = sum max(0,cos-tau) / min set size."""
    S = EA @ EB.T
    R = np.maximum(0.0, S - tau)
    ri, ci = linear_sum_assignment(-R)
    pairs = [(int(i), int(j), float(S[i, j])) for i, j in zip(ri, ci) if R[i, j] > 0]
    score = sum(p[2] - tau for p in pairs)
    return score / max(1, min(len(EA), len(EB))), pairs


def montage(paths, labels, out_path, cell=160):
    imgs = [cv2.resize(cv2.imread(p), (cell, cell)) for p in paths]
    if not imgs:
        return
    pad = 18
    canvas = np.full((cell + pad, cell * len(imgs), 3), 255, np.uint8)
    for i, im in enumerate(imgs):
        canvas[pad:pad + cell, i * cell:(i + 1) * cell] = im
        cv2.putText(canvas, labels[i][:20], (i * cell + 2, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)


def retrieve(E, manifest):
    stems = sorted(set(m["stem"] for m in manifest))
    idx = {s: [i for i, m in enumerate(manifest) if m["stem"] == s] for s in stems}
    emb = {s: E[idx[s]] for s in stems}
    pooled = {s: (emb[s].mean(0) / (np.linalg.norm(emb[s].mean(0)) + 1e-9)) for s in stems}

    print("\n================ PER-VIEW MATCHER (primary) ================")
    lines = []
    for q in stems:
        scores = []
        for c in stems:
            if c == q:
                continue
            sc, pairs = matcher_score(emb[q], emb[c])
            scores.append((sc, c, pairs))
        scores.sort(reverse=True)
        print(f"\nQUERY {q[:42]}  ({len(emb[q])} views)")
        for rank, (sc, c, pairs) in enumerate(scores[:3], 1):
            pv = ", ".join(f"q{i}~c{j}:{s:.2f}" for i, j, s in pairs[:4])
            print(f"   #{rank}  {sc:.3f}  {c[:40]:42s} [{pv}]")
        lines.append((q, scores))
        # montage of the top match's paired view crops
        if scores:
            top = scores[0]
            qpaths = [manifest[idx[q][i]]["crop"] for i, j, s in top[2]]
            cpaths = [manifest[idx[top[1]][j]]["crop"] for i, j, s in top[2]]
            labs = [f"q v{i}" for i, j, s in top[2]] + [f"c v{j}" for i, j, s in top[2]]
            montage(qpaths + cpaths, labs,
                    str(IMG / f"match_{q.split('_')[0][:18]}.png"))

    print("\n================ POOLED baseline (mean-vector cosine) ================")
    for q in stems:
        sims = sorted(((float(pooled[q] @ pooled[c]), c) for c in stems if c != q), reverse=True)
        print(f"\nQUERY {q[:42]}")
        for rank, (s, c) in enumerate(sims[:3], 1):
            print(f"   #{rank}  {s:.3f}  {c[:40]}")
    return lines


if __name__ == "__main__":
    print("=== embedding views (frozen DINOv2-small) ===")
    E, manifest = build_embeddings()
    retrieve(E, manifest)
    print(f"\nCrops/index in {OUT}/  ·  montages in {IMG}/  (eyeball match_*.png before trusting numbers)")
