"""
Trains ONE (model, preprocessing-variant) combination and saves:
    experiments/runs/<model>_<variant>/checkpoints/best.pt
    experiments/runs/<model>_<variant>/history.csv           (per-epoch train/val loss & acc)
    experiments/runs/<model>_<variant>/run_log.txt

Usage:
    python -m src.training.train --model mobilenet_v3 --variant original
    python -m src.training.train --model efficientnet_b0 --variant sam_segmented
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import load_config, resolve_path, get_logger
from src.utils.seed import set_seed
from src.augmentation.augment import build_transforms
from src.models.model_factory import build_model, build_fused_model, build_early_fused_model
from src.training.dataset_loader import WasteDataset, FusedWasteDataset, EarlyFusedWasteDataset


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mobilenet_v3", "efficientnet_b0"])
    parser.add_argument("--variant", required=True,
                        choices=["original", "bbox_cropped", "sam_segmented", "fused", "early_fused"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    run_name = f"{args.model}_{args.variant}"
    run_dir = resolve_path(cfg["paths"]["experiments_dir"]) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    logger = get_logger(run_name, log_file=str(run_dir / "run_log.txt"))

    device = torch.device("cuda" if torch.cuda.is_available() and
                           cfg["training"]["device"] == "cuda" else "cpu")
    logger.info("Run: %s | Device: %s", run_name, device)

    classes = sorted(cfg["dataset"]["classes"])
    num_classes = len(classes)

    train_tf = build_transforms(cfg, "train")
    eval_tf = build_transforms(cfg, "val")

    if args.variant == "fused":
        train_ds = FusedWasteDataset(cfg, "train", transform=train_tf)
        val_ds = FusedWasteDataset(cfg, "val", transform=eval_tf)
    elif args.variant == "early_fused":
        train_ds = EarlyFusedWasteDataset(cfg, "train", transform=train_tf)
        val_ds = EarlyFusedWasteDataset(cfg, "val", transform=eval_tf)
    else:
        train_ds = WasteDataset(cfg, args.variant, "train", transform=train_tf)
        val_ds = WasteDataset(cfg, args.variant, "val", transform=eval_tf)
    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                            pin_memory=(device.type == "cuda"))

    if args.variant == "fused":
        model = build_fused_model(args.model, num_classes).to(device)
    elif args.variant == "early_fused":
        model = build_early_fused_model(args.model, num_classes).to(device)
    else:
        model = build_model(args.model, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                                   weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )

    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    best_val_acc, epochs_no_improve = 0.0, 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        dt = time.time() - t0

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "lr": scheduler.get_last_lr()[0],
                        "epoch_time_s": round(dt, 2)})
        logger.info("Epoch %02d/%d | train_loss %.4f acc %.4f | val_loss %.4f acc %.4f | %.1fs",
                    epoch, epochs, train_loss, train_acc, val_loss, val_acc, dt)

        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_name": args.model,
                "variant": args.variant,
                "classes": classes,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
            }, run_dir / "checkpoints" / "best.pt")
            logger.info("  -> New best val_acc %.4f, checkpoint saved.", val_acc)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs).",
                            epoch, patience)
                break

    logger.info("Training complete. Best val_acc: %.4f", best_val_acc)


if __name__ == "__main__":
    main()
