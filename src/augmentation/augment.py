"""
Data augmentation (training split only, works for any configured dataset).

Applied on-the-fly inside the training DataLoader (standard, memory-efficient
practice) rather than pre-generating augmented image files. This keeps the
val/test sets and the three preprocessing variants (original / bbox_cropped /
sam_segmented) perfectly comparable, while the training set benefits from
richer, randomized augmentation every epoch.
"""
from torchvision import transforms


def build_transforms(cfg, split: str):
    img_size = cfg["dataset"]["image_size"]
    aug_cfg = cfg["augmentation"]

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats (pretrained backbones)
        std=[0.229, 0.224, 0.225],
    )

    if split == "train" and aug_cfg.get("enabled", True):
        return transforms.Compose([
            transforms.RandomResizedCrop(
                img_size, scale=tuple(aug_cfg["random_resized_crop_scale"])
            ),
            transforms.RandomHorizontalFlip(p=aug_cfg["random_horizontal_flip"]),
            transforms.RandomVerticalFlip(p=aug_cfg["random_vertical_flip"]),
            transforms.RandomRotation(aug_cfg["random_rotation_deg"]),
            transforms.ColorJitter(**aug_cfg["color_jitter"]),
            transforms.ToTensor(),
            normalize,
        ])

    # val / test: deterministic, no augmentation
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
