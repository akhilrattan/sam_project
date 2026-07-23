"""
PyTorch Dataset for a single preprocessing variant (original / bbox_cropped /
sam_segmented). Reads file lists from the shared split CSVs so every variant
and every model sees EXACTLY the same train/val/test images.

Also provides FusedWasteDataset for the `fused` two-stream variant, which
reads BOTH `original` and `bbox_cropped` for each sample (no new
preprocessing step - it reuses the folders generate_variants.py already
built) and stacks them into one (2, C, H, W) tensor per sample, so the
existing training/eval loops (`for images, labels, ids in loader`) don't
need to change at all.

Also provides EarlyFusedWasteDataset for the `early_fused` variant, which
reads the same `original` + `bbox_cropped` pair but channel-concatenates
them into a single (6, C, H, W)-collapsed -> (6, H, W) tensor per sample,
for single-backbone early-fusion models (see build_early_fused_model in
src/models/model_factory.py).
"""
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import resolve_path


class WasteDataset(Dataset):
    def __init__(self, cfg, variant: str, split: str, transform=None):
        self.cfg = cfg
        self.variant = variant
        self.split = split
        self.transform = transform

        processed_dir = resolve_path(cfg["paths"]["processed_dir"]) / variant
        splits_dir = resolve_path(cfg["paths"]["splits_dir"])
        split_csv = splits_dir / f"{cfg['dataset']['name']}_{split}.csv"
        self.df = pd.read_csv(split_csv).reset_index(drop=True)

        self.classes = sorted(cfg["dataset"]["classes"])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.processed_dir = processed_dir

        # Filter out any rows whose processed file doesn't exist yet (safety net)
        exists_mask = self.df.apply(
            lambda r: (processed_dir / r["class"] / r["image_id"]).exists(), axis=1
        )
        missing = (~exists_mask).sum()
        if missing:
            print(f"[WARN] {missing} images missing for variant '{variant}' split '{split}' "
                  f"- run generate_variants.py. Skipping them for now.")
        self.df = self.df[exists_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.processed_dir / row["class"] / row["image_id"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[row["class"]]
        return image, label, row["image_id"]


class _TwoImageWasteDatasetBase(Dataset):
    """Shared loading logic for variants that need BOTH `original` and
    `bbox_cropped` for each sample (no new preprocessing step - reuses the
    folders generate_variants.py already built). Subclasses only differ in
    how the two loaded/transformed images are combined into one tensor
    (see __getitem__ / _combine in FusedWasteDataset vs
    EarlyFusedWasteDataset)."""

    def __init__(self, cfg, split: str, transform=None):
        self.cfg = cfg
        self.split = split
        self.transform = transform

        processed_dir = resolve_path(cfg["paths"]["processed_dir"])
        self.dir_original = processed_dir / "original"
        self.dir_bbox = processed_dir / "bbox_cropped"

        splits_dir = resolve_path(cfg["paths"]["splits_dir"])
        split_csv = splits_dir / f"{cfg['dataset']['name']}_{split}.csv"
        self.df = pd.read_csv(split_csv).reset_index(drop=True)

        self.classes = sorted(cfg["dataset"]["classes"])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        exists_mask = self.df.apply(
            lambda r: (self.dir_original / r["class"] / r["image_id"]).exists()
            and (self.dir_bbox / r["class"] / r["image_id"]).exists(),
            axis=1,
        )
        missing = (~exists_mask).sum()
        if missing:
            print(f"[WARN] {missing} images missing an original/bbox_cropped pair for "
                  f"split '{split}' - run generate_variants.py. Skipping them for now.")
        self.df = self.df[exists_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_pair(self, row):
        img_original = Image.open(self.dir_original / row["class"] / row["image_id"]).convert("RGB")
        img_bbox = Image.open(self.dir_bbox / row["class"] / row["image_id"]).convert("RGB")
        if self.transform:
            img_original = self.transform(img_original)
            img_bbox = self.transform(img_bbox)
        return img_original, img_bbox

    def _combine(self, img_original, img_bbox):
        raise NotImplementedError

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_original, img_bbox = self._load_pair(row)
        combined = self._combine(img_original, img_bbox)
        label = self.class_to_idx[row["class"]]
        return combined, label, row["image_id"]


class FusedWasteDataset(_TwoImageWasteDatasetBase):
    """Two-stream dataset for the `fused` (late-fusion) variant: stacks
    `original` and `bbox_cropped` into a (2, C, H, W) tensor, one image per
    stream - see FusedTwoStreamModel in src/models/model_factory.py."""

    def _combine(self, img_original, img_bbox):
        return torch.stack([img_original, img_bbox], dim=0)   # (2, C, H, W)


class EarlyFusedWasteDataset(_TwoImageWasteDatasetBase):
    """Single-backbone dataset for the `early_fused` (early/pixel-fusion)
    variant: channel-concatenates `original` and `bbox_cropped` into one
    (6, H, W) tensor (channels 0-2 = original, 3-5 = bbox_cropped) - see
    build_early_fused_model in src/models/model_factory.py, which inflates
    the backbone's first conv layer to accept 6 input channels."""

    def _combine(self, img_original, img_bbox):
        return torch.cat([img_original, img_bbox], dim=0)   # (6, H, W)
