"""
Builds the three parallel, aligned datasets used for the comparative study:

  data/processed/realwaste/original/<class>/<image_id>        - untouched copy
  data/processed/realwaste/bbox_cropped/<class>/<image_id>    - cropped to a SAM-derived bbox
                                                                  (the picked mask's own bbox from
                                                                  SAM's automatic mask generator,
                                                                  or a heuristic fallback box;
                                                                  see src/annotation_generation/
                                                                  sam_auto_segment.py). YOLO is no
                                                                  longer used anywhere.
  data/processed/realwaste/sam_segmented/<class>/<image_id>   - SAM foreground mask (generated
                                                                  directly from the RAW image via
                                                                  SAM's automatic mask generator),
                                                                  background filled (white/black),
                                                                  then cropped to a box derived
                                                                  from the mask itself
                                                                  (cv2.boundingRect, tighter/padded
                                                                  vs. SAM's raw candidate bbox)

All three variants share identical filenames so any of them can be swapped in
via config without touching training code. bbox_cropped and sam_segmented both
come from the SAME single SAM automatic-segmentation pass now (one box, one
mask, per image) - they differ only in whether pixels outside the mask are
kept (bbox_cropped: yes, just cropped to the box) or blanked out
(sam_segmented: masked then cropped).

Usage:
    python -m src.preprocessing.generate_variants
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import load_config, resolve_path, get_logger

logger = get_logger("generate_variants")


def apply_sam_mask(image_bgr, mask, bg_color=(255, 255, 255)):
    mask_bool = mask.astype(bool)
    out = np.full_like(image_bgr, bg_color, dtype=np.uint8)
    out[mask_bool] = image_bgr[mask_bool]
    return out


def crop_to_bbox(image, bbox):
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    h, w = image.shape[:2]
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return image  # degenerate box safeguard
    return image[y1:y2, x1:x2]


def bbox_from_mask(mask, pad_ratio=0.05):
    """Derives a (padded) bounding box directly from a binary mask via
    cv2.boundingRect - used by the sam_segmented variant. Note this is a
    tighter, padded box recomputed from the mask pixels; bbox_cropped instead
    uses SAM's own raw candidate bbox (see sam_auto_segment.py)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape[:2]
        return [0, 0, w, h]  # no foreground found - fall back to full frame
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    mh, mw = mask.shape[:2]
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(mw, x2 + pad_x), min(mh, y2 + pad_y)
    return [int(x1), int(y1), int(x2), int(y2)]


def main():
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    splits_dir = resolve_path(cfg["paths"]["splits_dir"])
    ann_dir = resolve_path(cfg["paths"]["annotations_dir"])
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    bg_color = (255, 255, 255) if cfg["sam"]["mask_background"] == "white" else (0, 0, 0)

    df = pd.read_csv(splits_dir / f"{cfg['dataset']['name']}_full_split.csv")

    for variant in cfg["variants"]:
        (processed_dir / variant).mkdir(parents=True, exist_ok=True)

    n_ok = {v: 0 for v in cfg["variants"]}
    n_skip = {v: 0 for v in cfg["variants"]}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating variants"):
        image_id, cls = row["image_id"], row["class"]
        src_path = raw_dir / cls / image_id
        stem = Path(image_id).stem
        bbox_json = ann_dir / "bboxes" / (stem + ".json")
        mask_png = ann_dir / "sam_masks" / (stem + ".png")

        for variant in cfg["variants"]:
            out_dir = processed_dir / variant / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / image_id
            if out_path.exists():
                n_skip[variant] += 1
                continue

            if variant == "original":
                if src_path.exists():
                    shutil.copy2(src_path, out_path)
                    n_ok[variant] += 1
                continue

            if not src_path.exists():
                continue
            image = cv2.imread(str(src_path))
            if image is None:
                continue

            if variant == "bbox_cropped":
                # Uses ONLY the SAM-derived bbox JSON (written by
                # sam_auto_segment.py) - no YOLO involved.
                if not bbox_json.exists():
                    continue
                with open(bbox_json) as f:
                    ann = json.load(f)
                cropped = crop_to_bbox(image, ann["bbox"])
                cv2.imwrite(str(out_path), cropped)
                n_ok[variant] += 1

            elif variant == "sam_segmented":
                # Independent pipeline: uses ONLY the SAM mask (generated
                # directly from the raw image via automatic mask generation).
                # No YOLO bbox is involved anywhere in this branch.
                if not mask_png.exists():
                    continue
                mask = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape[:2] != image.shape[:2]:
                    continue
                masked = apply_sam_mask(image, mask, bg_color)
                mask_bbox = bbox_from_mask(mask)
                cropped = crop_to_bbox(masked, mask_bbox)
                cv2.imwrite(str(out_path), cropped)
                n_ok[variant] += 1

    for v in cfg["variants"]:
        logger.info("[%s] generated: %d | already existed / skipped: %d",
                    v, n_ok[v], n_skip[v])


if __name__ == "__main__":
    main()
