"""
Saves example CORRECT and INCORRECT classification prediction images
(with true vs predicted labels overlaid) for qualitative inspection.

Output:
    experiments/runs/<run>/prediction_samples/correct/*.jpg
    experiments/runs/<run>/prediction_samples/incorrect/*.jpg
    experiments/runs/<run>/prediction_samples/grid_correct.png
    experiments/runs/<run>/prediction_samples/grid_incorrect.png

Usage:
    python -m src.evaluation.prediction_visualizer --model mobilenet_v3 --variant original
"""
import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import load_config, resolve_path, get_logger

logger = get_logger("prediction_visualizer")


def save_grid(rows, processed_dir, out_path, title):
    n = len(rows)
    if n == 0:
        return
    cols = min(6, n)
    grid_rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 2.4, grid_rows * 2.6))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (_, row) in zip(axes, rows.iterrows()):
        img_path = processed_dir / row["true_class"] / row["image_id"]
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception:
            pass
        color = "green" if row["correct"] else "red"
        ax.set_title(f"T:{row['true_class'][:10]}\nP:{row['pred_class'][:10]}\n{row['confidence']:.2f}",
                     fontsize=7, color=color)
        ax.axis("off")

    for ax in axes[len(rows):]:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


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

    # `fused` and `early_fused` have no dedicated processed_dir folder of their
    # own (they reuse original + bbox_cropped at train/eval time) - show the
    # `original` crop here since it's the more informative one to eyeball.
    image_variant = "original" if args.variant in ("fused", "early_fused") else args.variant
    processed_dir = resolve_path(cfg["paths"]["processed_dir"]) / image_variant
    n_samples = cfg["evaluation"]["n_prediction_samples"]

    df = pd.read_csv(pred_csv)
    correct_df = df[df["correct"]].sample(min(n_samples, (df["correct"]).sum()),
                                           random_state=cfg["project"]["seed"])
    incorrect_df = df[~df["correct"]].sample(
        min(n_samples, (~df["correct"]).sum()), random_state=cfg["project"]["seed"]
    ) if (~df["correct"]).sum() > 0 else df.iloc[0:0]

    out_dir = run_dir / "prediction_samples"
    (out_dir / "correct").mkdir(parents=True, exist_ok=True)
    (out_dir / "incorrect").mkdir(parents=True, exist_ok=True)

    for _, row in correct_df.iterrows():
        src = processed_dir / row["true_class"] / row["image_id"]
        if src.exists():
            shutil.copy2(src, out_dir / "correct" / row["image_id"])
    for _, row in incorrect_df.iterrows():
        src = processed_dir / row["true_class"] / row["image_id"]
        if src.exists():
            shutil.copy2(src, out_dir / "incorrect" / row["image_id"])

    save_grid(correct_df, processed_dir, out_dir / "grid_correct.png",
              f"{run_name} - Correct predictions")
    save_grid(incorrect_df, processed_dir, out_dir / "grid_incorrect.png",
              f"{run_name} - Incorrect predictions")

    logger.info("Saved %d correct / %d incorrect prediction samples to %s",
                len(correct_df), len(incorrect_df), out_dir)


if __name__ == "__main__":
    main()
