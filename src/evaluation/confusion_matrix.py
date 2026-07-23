"""
Builds and saves the confusion matrix (raw counts CSV + heatmap PNG) for a
trained run, using the predictions.csv produced by evaluate.py.

Usage:
    python -m src.evaluation.confusion_matrix --model mobilenet_v3 --variant original
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import load_config, resolve_path, get_logger

logger = get_logger("confusion_matrix")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mobilenet_v3", "efficientnet_b0"])
    parser.add_argument("--variant", required=True,
                        choices=["original", "bbox_cropped", "sam_segmented", "fused", "early_fused"])
    args = parser.parse_args()

    cfg = load_config()
    run_name = f"{args.model}_{args.variant}"
    run_dir = resolve_path(cfg["paths"]["experiments_dir"]) / run_name
    pred_csv = run_dir / "predictions.csv"
    if not pred_csv.exists():
        logger.error("%s not found. Run evaluate.py first.", pred_csv)
        sys.exit(1)

    df = pd.read_csv(pred_csv)
    classes = sorted(cfg["dataset"]["classes"])

    cm = confusion_matrix(df["true_class"], df["pred_class"], labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(run_dir / "confusion_matrix.csv")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", xticklabels=classes,
                yticklabels=classes, cbar_kws={"label": "Normalized proportion"})
    plt.title(f"Confusion Matrix - {run_name}")
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    logger.info("Confusion matrix saved to %s", run_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
