"""embed_all.py — Embed all visual views with frozen DINOv2-small.

Writes:
  visual_pipe/index/embeddings.npy   — (N_views, 384) float32, L2-normalised
  visual_pipe/index/manifest.json    — [{stem, view, bbox_xywh, area_frac}]

Resumable: reads manifest.json to find already-done stems; skips them.

Run:
  /opt/anaconda3/envs/fyp/bin/python scripts/vispipe/embed_all.py
  /opt/anaconda3/envs/fyp/bin/python scripts/vispipe/embed_all.py --batch-size 64
"""
import argparse, json, os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import paths

OUT      = paths.VISUAL_INDEX
EMB_PATH = OUT / "embeddings.npy"
MAN_PATH = OUT / "manifest.json"
OUT.mkdir(parents=True, exist_ok=True)

def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _pick_device()


# ---------------------------------------------------------------------------
# Image preprocessing (same as embed_views.py)
# ---------------------------------------------------------------------------

def preprocess(crop_rgb: np.ndarray) -> Image.Image:
    """H×W×3 uint8 RGB → PIL 224×224 white-bg black-ink, strokes dilated."""
    g   = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    ink = (g < 200).astype(np.uint8)
    k   = max(1, int(round(max(g.shape) / 224)))
    if k > 1:
        ink = cv2.dilate(ink, np.ones((k, k), np.uint8))
    img = np.full(g.shape, 255, np.uint8)
    img[ink > 0] = 0
    h, w = img.shape
    s  = max(h, w)
    sq = np.full((s, s), 255, np.uint8)
    sq[(s-h)//2:(s-h)//2+h, (s-w)//2:(s-w)//2+w] = img
    sq = cv2.resize(sq, (224, 224), interpolation=cv2.INTER_AREA)
    return Image.fromarray(np.stack([sq]*3, -1))


# ---------------------------------------------------------------------------
# DINOv2-small encoder
# ---------------------------------------------------------------------------

def load_dino():
    from transformers import AutoModel
    from torchvision import transforms
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval().to(DEVICE)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    print(f"DINOv2-small loaded on {DEVICE}", flush=True)
    return model, tf


@torch.no_grad()
def embed_batch(model, tf, pil_imgs: list) -> np.ndarray:
    tensors = torch.stack([tf(p) for p in pil_imgs]).to(DEVICE)
    cls = model(pixel_values=tensors).last_hidden_state[:, 0, :].cpu().numpy()
    norms = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-9
    return (cls / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--reset", action="store_true",
                    help="Ignore existing manifest and re-embed everything")
    args = ap.parse_args()

    # Load existing progress
    existing_manifest = []
    existing_embs     = []
    done_stems        = set()

    if not args.reset and MAN_PATH.exists() and EMB_PATH.exists():
        existing_manifest = json.loads(MAN_PATH.read_text())
        existing_embs     = [np.load(EMB_PATH)]
        done_stems        = set(m["stem"] for m in existing_manifest)
        print(f"Resuming: {len(done_stems)} stems already embedded "
              f"({len(existing_manifest)} view vectors)", flush=True)

    # Collect stems to process
    all_stem_dirs = sorted(
        d for d in paths.VISUAL_PIPE.iterdir()
        if d.is_dir() and d.name not in ("images", "index")
    )
    todo = [d for d in all_stem_dirs if d.name not in done_stems]
    print(f"To embed: {len(todo)} stems  |  total corpus: {len(all_stem_dirs)}", flush=True)

    if not todo:
        print("Nothing to do — all stems already embedded.")
        return

    model, tf = load_dino()

    new_manifest = []
    new_embs     = []
    batch_imgs   = []
    batch_meta   = []

    def flush_batch():
        if not batch_imgs:
            return
        vecs = embed_batch(model, tf, batch_imgs)
        new_embs.extend(vecs)
        new_manifest.extend(batch_meta)
        batch_imgs.clear()
        batch_meta.clear()

    for si, sd in enumerate(todo, 1):
        stem    = sd.name
        vf      = sd / "sam_views" / "sam_masks_filtered_v5.json"
        mf      = sd / "sam_views" / "sam_masks.json"
        sup_png = sd / "polygon_removed_super.png"

        if not (vf.exists() and mf.exists() and sup_png.exists()):
            print(f"  SKIP {stem[:55]} (missing files)", flush=True)
            continue

        try:
            mask_meta = json.loads(mf.read_text())
            H, W      = mask_meta["image_hw"]
            v5_data   = json.loads(vf.read_text())
            views     = v5_data.get("views", [])
        except Exception as e:
            print(f"  SKIP {stem[:55]} (JSON error: {e})", flush=True)
            continue

        sup = cv2.imread(str(sup_png))
        if sup is None:
            print(f"  SKIP {stem[:55]} (can't read image)", flush=True)
            continue
        if sup.shape[:2] != (H, W):
            sup = cv2.resize(sup, (W, H))
        sup_rgb = cv2.cvtColor(sup, cv2.COLOR_BGR2RGB)

        stem_views = 0
        for vi, v in enumerate(views):
            x, y, w, h = v["bbox_xywh"]
            crop = sup_rgb[max(0, y):y+h, max(0, x):x+w]
            if crop.size == 0:
                continue
            batch_imgs.append(preprocess(crop))
            batch_meta.append({
                "stem":       stem,
                "view":       vi,
                "bbox_xywh":  [x, y, w, h],
                "area_frac":  v.get("bbox_area_frac"),
            })
            stem_views += 1
            if len(batch_imgs) >= args.batch_size:
                flush_batch()

        print(f"  [{si:4d}/{len(todo)}] {stem[:55]:57s} {stem_views} views", flush=True)

        # Checkpoint every 200 stems
        if si % 200 == 0:
            flush_batch()
            _save(existing_embs, existing_manifest, new_embs, new_manifest)
            print(f"  [checkpoint] saved {len(new_manifest)} new vectors", flush=True)

    flush_batch()
    _save(existing_embs, existing_manifest, new_embs, new_manifest)
    total = len(existing_manifest) + len(new_manifest)
    print(f"\nDone — {total} view vectors across "
          f"{len(done_stems) + len(set(m['stem'] for m in new_manifest))} stems", flush=True)
    print(f"  {EMB_PATH}", flush=True)
    print(f"  {MAN_PATH}", flush=True)


def _save(existing_embs, existing_manifest, new_embs, new_manifest):
    if not new_embs:
        return
    all_embs = np.vstack(
        existing_embs + ([np.stack(new_embs)] if new_embs else [])
    ).astype(np.float32)
    all_man  = existing_manifest + new_manifest
    np.save(EMB_PATH, all_embs)
    MAN_PATH.write_text(json.dumps(all_man, indent=1))


if __name__ == "__main__":
    main()
