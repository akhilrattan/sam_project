"""
Plots training vs validation loss & accuracy curves from history.csv
(saved automatically during train.py).

Usage:
    python -m src.evaluation.training_curves --model mobilenet_v3 --variant original
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import load_config, resolve_path, get_logger

logger = get_logger("training_curves")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mobilenet_v3", "efficientnet_b0"])
    parser.add_argument("--variant", required=True,
                        choices=["original", "bbox_cropped", "sam_segmented", "fused", "early_fused"])
    args = parser.parse_args()

    cfg = load_config()
    run_name = f"{args.model}_{args.variant}"
    run_dir = resolve_path(cfg["paths"]["experiments_dir"]) / run_name
    hist_csv = run_dir / "history.csv"
    if not hist_csv.exists():
        logger.error("%s not found. Train this run first.", hist_csv)
        sys.exit(1)

    df = pd.read_csv(hist_csv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o", ms=3)
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker="o", ms=3)
    axes[0].set_title(f"{run_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["train_acc"], label="Train Acc", marker="o", ms=3)
    axes[1].plot(df["epoch"], df["val_acc"], label="Val Acc", marker="o", ms=3)
    axes[1].set_title(f"{run_name} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(run_dir / "training_val_curves.png", dpi=150)
    plt.close()

    logger.info("Saved training/validation curves to %s",
                run_dir / "training_val_curves.png")


if __name__ == "__main__":
    main()
