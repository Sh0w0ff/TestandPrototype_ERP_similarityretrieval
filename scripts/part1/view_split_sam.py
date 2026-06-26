"""
SAM zero-shot view segmentation spike (2026-05-26).

Runs SamAutomaticMaskGenerator ("everything" mode, no prompts) on each sample's
polygon_removed_super.png and writes a colored-mask overlay + per-mask metadata
JSON for visual inspection.

Question this spike answers: does SAM's unprompted "everything" mode produce
masks that correspond to engineering-drawing views on the polygon-removed
super image, without any prompt engineering or fine-tuning?

If yes -> view-seg solved at zero label cost.
If no  -> escalate to supervised (Xiao FCN / U-Net / V2 Zhang GCN / Khan YOLO).
"""
import base64
import json
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def encode_seg(seg: np.ndarray) -> str:
    """Pack 2D bool array into zlib-compressed base64 string."""
    packed = np.packbits(seg.astype(np.uint8).flatten(order="C"), bitorder="big")
    return base64.b64encode(zlib.compress(packed.tobytes(), level=6)).decode("ascii")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); import paths
SAMPLES_DIR = paths.VISUAL_PIPE
WEIGHTS = ROOT / "models" / "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
OUT_SUBDIR = "sam_views"

# Resize longest side to this BEFORE padding+SAM. SAM's ViT-H resizes to 1024
# longest side internally for the encoder anyway, but NMS + mask postprocessing
# happen at input resolution -- so downscaling big stems gives big speedups
# without losing view-detection quality. 2000 px keeps thin features > 10 px.
RESIZE_LONGEST = 2000

# Whitespace border added (on the resized image) before SAM, to pull edge-flush
# views away from the image boundary. SAM's stability_score degrades when a
# proposed mask is clipped by the input edge, causing AutomaticMaskGenerator to
# discard views whose true bbox sits flush with the page border (observed on
# 52262640 side view + 52284552 right view). Translated back to ORIGINAL-image
# coords in saved JSON (subtract pad, then divide by resize scale).
PAD_BORDER_PX = 150

# 2026-05-30: the thin channel-section view (38918) IS seeded at pps=32 -- it was
# only ever rejected by the 0.95 stability threshold (it scores ~0.87). Dropping
# stability_score_thresh to 0.85 recovers it at NO extra cost (41s, same as the
# old basic run). crop_n_layers was tried and rejected: 758s-8189s on CPU and it
# degraded results. Resolution floor is 2000 (1600/1400 lose the section).
GEN_KWARGS = dict(
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.85,
    min_mask_region_area=0,
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # MPS skipped: SamAutomaticMaskGenerator builds float64 point grids,
    # which MPS doesn't support. CPU is the reliable fallback on Mac.
    return "cpu"


def colorize_masks(image_rgb: np.ndarray, masks: list[dict]) -> np.ndarray:
    """Overlay each mask in a distinct random color on a faded background."""
    rng = np.random.default_rng(0)
    overlay = (image_rgb.astype(np.float32) * 0.35).astype(np.uint8)
    # Sort largest-first so small masks paint on top
    for m in sorted(masks, key=lambda d: -d["area"]):
        seg = m["segmentation"]
        color = rng.integers(60, 256, size=3, dtype=np.int32).tolist()
        overlay[seg] = (0.45 * overlay[seg] + 0.55 * np.array(color)).astype(np.uint8)
    # Draw bbox outlines + index labels
    for i, m in enumerate(sorted(masks, key=lambda d: -d["area"])):
        x, y, w, h = [int(v) for v in m["bbox"]]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(overlay, str(i), (x + 4, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, str(i), (x + 4, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return overlay


def run_one(generator: SamAutomaticMaskGenerator, src_png: Path, out_dir: Path) -> dict | None:
    cache_file = out_dir / "sam_masks.json"
    if cache_file.exists():
        return None  # already done
    out_dir.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(src_png))
    if bgr is None:
        raise RuntimeError(f"could not read {src_png}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb.shape[:2]

    # 1) Downscale longest side to RESIZE_LONGEST (skip if already smaller).
    scale = min(1.0, RESIZE_LONGEST / max(orig_h, orig_w))
    if scale < 1.0:
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        rgb_scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        rgb_scaled = rgb
        new_h, new_w = orig_h, orig_w
    inv_scale = 1.0 / scale if scale > 0 else 1.0

    # 2) Pad with whitespace.
    pad = PAD_BORDER_PX
    if pad > 0:
        rgb_in = cv2.copyMakeBorder(rgb_scaled, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
    else:
        rgb_in = rgb_scaled

    t0 = time.time()
    masks = generator.generate(rgb_in)
    elapsed = time.time() - t0

    # 3) Translate bboxes (subtract pad, scale up) + crop+upsample segmentations
    # back to original image size. Overlay/JSON live in original coords.
    for m in masks:
        # bbox: padded-scaled -> scaled -> original
        x, y, w, h = m["bbox"]
        x -= pad; y -= pad
        x0 = int(round(max(0, x) * inv_scale))
        y0 = int(round(max(0, y) * inv_scale))
        x1 = int(round(min(new_w, x + w) * inv_scale))
        y1 = int(round(min(new_h, y + h) * inv_scale))
        x1 = min(orig_w, x1); y1 = min(orig_h, y1)
        m["bbox"] = [x0, y0, max(0, x1 - x0), max(0, y1 - y0)]
        # segmentation: crop pad region, then upsample to original size
        seg = m["segmentation"]
        if pad > 0:
            seg = seg[pad:pad + new_h, pad:pad + new_w]
        if scale < 1.0:
            seg = cv2.resize(seg.astype(np.uint8), (orig_w, orig_h),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        m["segmentation"] = seg
        m["area"] = int(seg.sum())

    overlay_rgb = colorize_masks(rgb, masks)
    cv2.imwrite(str(out_dir / "sam_overlay.png"),
                cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    meta = {
        "source": str(src_png.relative_to(ROOT)),
        "image_hw": [orig_h, orig_w],
        "resize_longest": RESIZE_LONGEST,
        "scale_used": scale,
        "scaled_hw": [new_h, new_w],
        "pad_border_px": pad,
        "n_masks": len(masks),
        "elapsed_s": round(elapsed, 2),
        "generator_kwargs": GEN_KWARGS,
        "masks": [
            {
                "i": i,
                "bbox_xywh": [int(v) for v in m["bbox"]],
                "area": int(m["area"]),
                "predicted_iou": float(m["predicted_iou"]),
                "stability_score": float(m["stability_score"]),
                "rle": encode_seg(m["segmentation"]),
            }
            for i, m in enumerate(sorted(masks, key=lambda d: -d["area"]))
        ],
    }
    (out_dir / "sam_masks.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(stems: list[str] | None = None) -> None:
    if not WEIGHTS.exists():
        sys.exit(f"weights not found at {WEIGHTS}")

    device = pick_device()
    print(f"[sam] loading {MODEL_TYPE} from {WEIGHTS.name} on {device}", flush=True)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(WEIGHTS))
    sam.to(device=device)
    generator = SamAutomaticMaskGenerator(sam, **GEN_KWARGS)

    if stems:
        targets = [SAMPLES_DIR / s for s in stems]
    else:
        targets = sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir())

    summary = []
    for stem_dir in targets:
        src = stem_dir / "polygon_removed_super.png"
        if not src.exists():
            print(f"[sam] SKIP {stem_dir.name} (no polygon_removed_super.png)", flush=True)
            continue
        out_dir = stem_dir / OUT_SUBDIR
        print(f"[sam] {stem_dir.name} ...", flush=True)
        meta = run_one(generator, src, out_dir)
        if meta is None:
            print(f"       SKIP (cached)", flush=True)
            continue
        print(f"       n_masks={meta['n_masks']}  t={meta['elapsed_s']}s  -> {out_dir.name}/", flush=True)
        summary.append({"stem": stem_dir.name, "n_masks": meta["n_masks"], "elapsed_s": meta["elapsed_s"]})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = SAMPLES_DIR / "sam_views_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[sam] summary -> {summary_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
