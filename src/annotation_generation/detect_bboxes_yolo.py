"""
RealWaste has NO ground-truth annotations, so we generate bounding boxes
ourselves using a pretrained YOLOv8 detector.

Since waste items (cardboard, textile trash, vegetation, ...) are mostly NOT
COCO classes, we do not filter by class label. Instead we treat YOLOv8 as a
generic salient-object detector: we take the highest-confidence box from ANY
class above a low confidence threshold. If YOLO detects nothing (common for
categories like "Miscellaneous Trash"), we fall back to a heuristic box
(full image minus a small margin), which is a reasonable prior since RealWaste
images are single-object, roughly centred photographs.

Output: one JSON per image in data/annotations_generated/realwaste/bboxes/
    {"image_id": ..., "bbox": [x1, y1, x2, y2], "source": "yolo" | "fallback",
     "confidence": float}

Usage:
    python -m src.annotation_generation.detect_bboxes_yolo
"""
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import load_config, resolve_path, get_logger

logger = get_logger("detect_bboxes_yolo")


def fallback_bbox(w, h, margin_ratio):
    mx, my = int(w * margin_ratio), int(h * margin_ratio)
    return [mx, my, w - mx, h - my]


def main():
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    splits_dir = resolve_path(cfg["paths"]["splits_dir"])
    ann_dir = resolve_path(cfg["paths"]["annotations_dir"]) / "bboxes"
    ann_dir.mkdir(parents=True, exist_ok=True)

    split_csv = splits_dir / f"{cfg['dataset']['name']}_full_split.csv"
    if not split_csv.exists():
        logger.error("%s not found. Run clean_and_split.py first.", split_csv)
        sys.exit(1)
    df = pd.read_csv(split_csv)

    from ultralytics import YOLO
    logger.info("Loading YOLOv8 weights: %s (auto-downloads on first run)",
                cfg["annotation"]["yolo_weights"])
    model = YOLO(cfg["annotation"]["yolo_weights"])

    conf_thresh = cfg["annotation"]["yolo_conf_threshold"]
    margin_ratio = cfg["annotation"]["fallback_margin_ratio"]

    n_yolo, n_fallback = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="YOLO bbox generation"):
        img_path = raw_dir / row["class"] / row["image_id"]
        out_path = ann_dir / (Path(row["image_id"]).stem + ".json")
        if out_path.exists():
            continue
        if not img_path.exists():
            logger.warning("Missing image: %s", img_path)
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        result = model.predict(source=str(img_path), conf=conf_thresh, verbose=False)[0]

        bbox, conf, source = None, 0.0, "fallback"
        if len(result.boxes) > 0:
            # take the single highest-confidence detection, any class
            best_idx = result.boxes.conf.argmax().item()
            xyxy = result.boxes.xyxy[best_idx].tolist()
            conf = float(result.boxes.conf[best_idx].item())
            bbox = [int(v) for v in xyxy]
            source = "yolo"
            n_yolo += 1
        else:
            bbox = fallback_bbox(w, h, margin_ratio)
            n_fallback += 1

        with open(out_path, "w") as f:
            json.dump({
                "image_id": row["image_id"],
                "class": row["class"],
                "width": w, "height": h,
                "bbox": bbox,
                "confidence": round(conf, 4),
                "source": source,
            }, f, indent=2)

    logger.info("Bbox generation complete. YOLO detections: %d | Fallback boxes: %d",
                n_yolo, n_fallback)
    logger.info("Fallback rate: %.1f%%", 100 * n_fallback / max(1, n_yolo + n_fallback))


if __name__ == "__main__":
    main()
