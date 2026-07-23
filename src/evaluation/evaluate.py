"""
Evaluates a trained (model, variant) checkpoint on the TEST split.
Saves:
    experiments/runs/<run>/overall_metrics.json
    experiments/runs/<run>/per_class_metrics.csv
    experiments/runs/<run>/predictions.csv        (raw per-image predictions, used by
                                                     confusion_matrix.py / prediction_visualizer.py)

Usage:
    python -m src.evaluation.evaluate --model mobilenet_v3 --variant original
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              precision_score, recall_score, f1_score)
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import load_config, resolve_path, get_logger
from src.augmentation.augment import build_transforms
from src.models.model_factory import build_model, build_fused_model, build_early_fused_model
from src.training.dataset_loader import WasteDataset, FusedWasteDataset, EarlyFusedWasteDataset

logger = get_logger("evaluate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mobilenet_v3", "efficientnet_b0"])
    parser.add_argument("--variant", required=True,
                        choices=["original", "bbox_cropped", "sam_segmented", "fused", "early_fused"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config()
    run_name = f"{args.model}_{args.variant}"
    run_dir = resolve_path(cfg["paths"]["experiments_dir"]) / run_name
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s. Train this run first.", ckpt_path)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    num_classes = len(classes)

    if args.variant == "fused":
        model = build_fused_model(args.model, num_classes)
    elif args.variant == "early_fused":
        model = build_early_fused_model(args.model, num_classes)
    else:
        model = build_model(args.model, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    eval_tf = build_transforms(cfg, "test")
    if args.variant == "fused":
        ds = FusedWasteDataset(cfg, args.split, transform=eval_tf)
    elif args.variant == "early_fused":
        ds = EarlyFusedWasteDataset(cfg, args.split, transform=eval_tf)
    else:
        ds = WasteDataset(cfg, args.variant, args.split, transform=eval_tf)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                        num_workers=cfg["training"]["num_workers"])

    all_preds, all_labels, all_ids, all_probs = [], [], [], []
    with torch.no_grad():
        for images, labels, ids in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_ids.extend(ids)
            all_probs.extend(probs.max(dim=1).values.cpu().numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)

    # ---- Overall metrics ----
    acc = accuracy_score(all_labels, all_preds)
    macro_p = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_r = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_p = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    weighted_r = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    overall = {
        "run_name": run_name, "model": args.model, "variant": args.variant, "split": args.split,
        "n_samples": len(all_labels),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "best_val_acc_during_training": round(float(ckpt["val_acc"]), 4),
        "best_epoch": int(ckpt["epoch"]),
    }
    with open(run_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)

    # ---- Per-class metrics ----
    p, r, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(num_classes)), zero_division=0
    )
    per_class_df = pd.DataFrame({
        "class": classes, "precision": p.round(4), "recall": r.round(4),
        "f1_score": f1.round(4), "support": support,
    })
    per_class_df.to_csv(run_dir / "per_class_metrics.csv", index=False)

    # ---- Raw predictions (feeds confusion matrix + prediction visualizer) ----
    pred_df = pd.DataFrame({
        "image_id": all_ids,
        "true_class": [classes[i] for i in all_labels],
        "pred_class": [classes[i] for i in all_preds],
        "confidence": np.round(all_probs, 4),
        "correct": all_labels == all_preds,
    })
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    logger.info("[%s] Test accuracy: %.4f | macro-F1: %.4f | weighted-F1: %.4f",
                run_name, acc, macro_f1, weighted_f1)
    logger.info("Per-class metrics:\n%s", per_class_df.to_string(index=False))


if __name__ == "__main__":
    main()
