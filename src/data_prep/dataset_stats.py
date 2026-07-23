"""
Exploratory data analysis for the configured dataset (RealWaste, TrashNet, ...):
- class distribution
- image resolution stats
- rough file-size / background-complexity proxy

Outputs a report + bar chart to results/figures/ and results/tables/
Usage:
    python -m src.data_prep.dataset_stats
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import load_config, resolve_path, get_logger

logger = get_logger("dataset_stats")


def main():
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)

    records = []
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                records.append({
                    "class": class_dir.name,
                    "path": str(img_path),
                    "width": w,
                    "height": h,
                    "aspect_ratio": round(w / h, 3),
                    "file_size_kb": round(img_path.stat().st_size / 1024, 1),
                })
            except Exception as e:
                logger.warning("Could not read %s: %s", img_path, e)

    df = pd.DataFrame(records)
    if df.empty:
        logger.error("No images found under %s. Run download_dataset.py first.", raw_dir)
        return

    df.to_csv(results_dir / "tables" / "raw_image_inventory.csv", index=False)

    # ---- Class distribution ----
    dist = df["class"].value_counts().sort_index()
    dist.to_csv(results_dir / "tables" / "class_distribution.csv")

    plt.figure(figsize=(10, 5))
    dist.plot(kind="bar", color="#4C72B0")
    plt.title(f"{cfg['dataset']['name'].capitalize()} - Class Distribution")
    plt.ylabel("Number of images")
    plt.xlabel("Class")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / "class_distribution.png", dpi=150)
    plt.close()

    # ---- Resolution stats ----
    res_summary = df[["width", "height", "aspect_ratio", "file_size_kb"]].describe()
    res_summary.to_csv(results_dir / "tables" / "resolution_summary.csv")

    logger.info("Total images: %d across %d classes", len(df), df['class'].nunique())
    logger.info("Class distribution:\n%s", dist.to_string())
    logger.info("Imbalance ratio (max/min class count): %.2f",
                dist.max() / dist.min())
    logger.info("Saved EDA report to results/tables/ and results/figures/")


if __name__ == "__main__":
    main()
