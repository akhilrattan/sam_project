"""
Creates stratified train/val/test splits from data/raw/realwaste and writes
CSVs (filename, class, split) to data/splits/. These CSVs are the single
source of truth used by every downstream stage (annotation, preprocessing,
training) so that all three preprocessing variants share IDENTICAL splits.

Usage:
    python -m src.data_prep.clean_and_split
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import load_config, resolve_path, get_logger
from src.utils.seed import set_seed

logger = get_logger("clean_and_split")


def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    splits_dir = resolve_path(cfg["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_r = cfg["dataset"]["train_ratio"]
    val_r = cfg["dataset"]["val_ratio"]
    test_r = cfg["dataset"]["test_ratio"]
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "split ratios must sum to 1.0"

    rows = []
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append({"image_id": img_path.stem + img_path.suffix,
                             "class": class_dir.name})

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("No images found in %s. Run download_dataset.py first.", raw_dir)
        return

    # First split off test set, then split remaining into train/val
    train_val_df, test_df = train_test_split(
        df, test_size=test_r, stratify=df["class"], random_state=cfg["project"]["seed"]
    )
    val_relative = val_r / (train_r + val_r)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative, stratify=train_val_df["class"],
        random_state=cfg["project"]["seed"]
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    ds_name = cfg["dataset"]["name"]
    full_df = pd.concat([train_df, val_df, test_df]).sort_values(["class", "image_id"])
    full_df.to_csv(splits_dir / f"{ds_name}_full_split.csv", index=False)

    # Also save per-split CSVs for convenience
    train_df.to_csv(splits_dir / f"{ds_name}_train.csv", index=False)
    val_df.to_csv(splits_dir / f"{ds_name}_val.csv", index=False)
    test_df.to_csv(splits_dir / f"{ds_name}_test.csv", index=False)

    logger.info("Split sizes -> train: %d, val: %d, test: %d (total %d)",
                len(train_df), len(val_df), len(test_df), len(full_df))
    logger.info("Per-class train counts:\n%s", train_df['class'].value_counts().to_string())


if __name__ == "__main__":
    main()
