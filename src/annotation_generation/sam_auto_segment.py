"""
Segments RealWaste images directly from RAW images using SAM's
AUTOMATIC mask generator (no YOLO bounding box prompt involved).

SAM proposes many candidate masks per image; we pick the "best" one using:
    1. Discard masks that touch the image border on all 4 sides (usually background)
    2. Discard masks smaller than a minimum area fraction (noise / small artifacts)
    3. Among remaining candidates, keep the one whose centroid is closest to the
       image center AND has the largest area (waste items are typically the
       single, roughly-centered subject of RealWaste photos)

This same pass is now also the SOLE source of bounding boxes for the
`bbox_cropped` variant: the picked mask's own bbox (from SAM, in xywh) is
converted to xyxy and written out in the same JSON schema that used to come
from detect_bboxes_yolo.py. YOLO is no longer used anywhere in the pipeline.
If SAM finds no valid candidate mask, we fall back to a heuristic box (full
image minus a small margin), same idea as the old YOLO fallback.

Saves:
    data/annotations_generated/realwaste/sam_masks/<image_id>.png    (binary mask)
    data/annotations_generated/realwaste/bboxes/<image_id>.json      (bbox, SAM-derived)
    data/annotations_generated/realwaste/annotation_qc/<image_id>.jpg (overlay, every Nth image)

Usage:
    python -m src.annotation_generation.sam_auto_segment
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import load_config, resolve_path, get_logger

logger = get_logger("sam_auto_segment")


def fallback_bbox(w, h, margin_ratio):
    mx, my = int(w * margin_ratio), int(h * margin_ratio)
    return [mx, my, w - mx, h - my]


def pick_best_mask(masks, img_w, img_h, min_area_frac, border_margin_px=3):
    """masks: list of dicts from SamAutomaticMaskGenerator.generate()
    Each dict has 'segmentation' (bool array), 'area', 'bbox' (xywh)."""
    img_area = img_w * img_h
    img_cx, img_cy = img_w / 2.0, img_h / 2.0

    candidates = []
    for m in masks:
        area = m["area"]
        if area < min_area_frac * img_area:
            continue

        x, y, w, h = m["bbox"]
        touches_left = x <= border_margin_px
        touches_top = y <= border_margin_px
        touches_right = (x + w) >= (img_w - border_margin_px)
        touches_bottom = (y + h) >= (img_h - border_margin_px)
        # Reject masks that span (almost) the entire frame - these are usually
        # background / table surface rather than the waste object itself
        if touches_left and touches_top and touches_right and touches_bottom:
            continue
        if area > 0.92 * img_area:
            continue

        cx, cy = x + w / 2.0, y + h / 2.0
        dist_from_center = np.hypot(cx - img_cx, cy - img_cy) / np.hypot(img_cx, img_cy)

        # Score: prefer large area, prefer close to center
        score = (area / img_area) - 0.5 * dist_from_center
        candidates.append((score, m))

    if not candidates:
        # Fallback: just take the largest mask overall (even if border-touching)
        if masks:
            return max(masks, key=lambda m: m["area"])
        return None

    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def main():
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    splits_dir = resolve_path(cfg["paths"]["splits_dir"])
    ann_dir = resolve_path(cfg["paths"]["annotations_dir"])
    mask_dir = ann_dir / "sam_masks"
    qc_dir = ann_dir / "annotation_qc"
    bbox_dir = ann_dir / "bboxes"
    mask_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir.mkdir(parents=True, exist_ok=True)

    split_csv = splits_dir / f"{cfg['dataset']['name']}_full_split.csv"
    if not split_csv.exists():
        logger.error("%s not found. Run clean_and_split.py first.", split_csv)
        sys.exit(1)
    df = pd.read_csv(split_csv)

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    ckpt_path = resolve_path(cfg["sam"]["checkpoint_path"])
    if not ckpt_path.exists():
        logger.error("SAM checkpoint not found at %s. "
                      "Run: python -m src.annotation_generation.download_sam_checkpoint", ckpt_path)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading SAM (%s) on %s for AUTOMATIC mask generation", cfg["sam"]["model_type"], device)
    sam = sam_model_registry[cfg["sam"]["model_type"]](checkpoint=str(ckpt_path))
    sam.to(device)

    amg_cfg = cfg["sam"]["automatic_mask_generator"]
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=amg_cfg["points_per_side"],
        pred_iou_thresh=amg_cfg["pred_iou_thresh"],
        stability_score_thresh=amg_cfg["stability_score_thresh"],
        min_mask_region_area=amg_cfg["min_mask_region_area"],
    )

    min_area_frac = amg_cfg["min_area_fraction"]
    margin_ratio = cfg["sam"].get("bbox_fallback_margin_ratio", 0.08)
    qc_every_n = 25
    n_ok, n_no_candidate, n_bbox_sam, n_bbox_fallback = 0, 0, 0, 0

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="SAM auto-segmentation")):
        image_id, cls = row["image_id"], row["class"]
        mask_out = mask_dir / (Path(image_id).stem + ".png")
        bbox_out = bbox_dir / (Path(image_id).stem + ".json")
        if mask_out.exists() and bbox_out.exists():
            continue

        img_path = raw_dir / cls / image_id
        if not img_path.exists():
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            logger.warning("Could not read %s", img_path)
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        masks = mask_generator.generate(image_rgb)
        best = pick_best_mask(masks, w, h, min_area_frac)

        if best is None:
            n_no_candidate += 1
            bbox = fallback_bbox(w, h, margin_ratio)
            confidence = 0.0
            bbox_source = "fallback"
            n_bbox_fallback += 1
        else:
            mask_uint8 = (best["segmentation"].astype(np.uint8)) * 255
            cv2.imwrite(str(mask_out), mask_uint8)
            n_ok += 1

            x, y, bw, bh = best["bbox"]
            bbox = [int(x), int(y), int(x + bw), int(y + bh)]
            confidence = round(float(best.get("predicted_iou", 0.0)), 4)
            bbox_source = "sam"
            n_bbox_sam += 1

            if i % qc_every_n == 0:
                overlay = image_bgr.copy()
                colored = np.zeros_like(overlay)
                colored[:, :, 1] = mask_uint8
                overlay = cv2.addWeighted(overlay, 0.7, colored, 0.3, 0)
                cv2.rectangle(overlay, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 0, 255), 2)
                cv2.imwrite(str(qc_dir / (Path(image_id).stem + ".jpg")), overlay)

        with open(bbox_out, "w") as f:
            json.dump({
                "image_id": image_id,
                "class": cls,
                "width": w, "height": h,
                "bbox": bbox,
                "confidence": confidence,
                "source": bbox_source,
            }, f, indent=2)

    logger.info("SAM automatic segmentation complete. Masks saved: %d | No valid mask candidate: %d",
                n_ok, n_no_candidate)
    logger.info("Bbox generation (SAM-derived, feeds bbox_cropped): SAM boxes: %d | Fallback boxes: %d",
                n_bbox_sam, n_bbox_fallback)
    logger.info("QC overlay samples saved to %s (spot-check these!)", qc_dir)


if __name__ == "__main__":
    main()
