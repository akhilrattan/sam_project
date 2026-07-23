"""
Downloads the RealWaste dataset (no bundled annotations) and organizes it into
data/raw/realwaste/<ClassName>/*.jpg

Primary method : kagglehub (dataset: joebeachcapital/realwaste)
Fallback       : prints manual download instructions if kagglehub/Kaggle auth is unavailable.

Usage:
    python -m src.data_prep.download_dataset
"""
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import load_config, resolve_path, get_logger

logger = get_logger("download_dataset")


def flatten_into_raw_dir(src_root: Path, raw_dir: Path):
    """Kaggle download can nest folders (e.g. RealWaste/realwaste-main/<class>/*.jpg).
    This walks the tree and copies class-folders directly under raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png"}
    class_dirs_found = []

    for path in src_root.rglob("*"):
        if path.is_dir():
            imgs = [p for p in path.iterdir() if p.suffix.lower() in image_exts]
            if imgs:
                class_dirs_found.append(path)

    if not class_dirs_found:
        logger.error("No class-folders with images found under %s", src_root)
        return 0

    total = 0
    for class_dir in class_dirs_found:
        class_name = class_dir.name
        dest = raw_dir / class_name
        dest.mkdir(parents=True, exist_ok=True)
        for img in class_dir.iterdir():
            if img.suffix.lower() in image_exts:
                shutil.copy2(img, dest / img.name)
                total += 1
        logger.info("Copied %d images -> %s", len(list(dest.iterdir())), dest)

    return total


def main():
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    slug = cfg["dataset"]["kaggle_slug"]

    if any(raw_dir.glob("*/*.jpg")) or any(raw_dir.glob("*/*.png")):
        logger.info("Images already present in %s - skipping download.", raw_dir)
        return

    try:
        import kagglehub
        logger.info("Downloading RealWaste dataset via kagglehub (%s) ...", slug)
        download_path = kagglehub.dataset_download(slug)
        download_path = Path(download_path)
        logger.info("Downloaded to %s. Organizing into %s ...", download_path, raw_dir)
        n = flatten_into_raw_dir(download_path, raw_dir)
        logger.info("Done. %d images organized into %d class folders.",
                     n, len(list(raw_dir.iterdir())))
    except Exception as e:
        logger.error("Automatic download failed: %s", e)
        logger.error(
            "\nMANUAL FALLBACK:\n"
            "1. Go to https://www.kaggle.com/datasets/joebeachcapital/realwaste\n"
            "   (or the UCI mirror: https://archive.ics.uci.edu/dataset/908/realwaste)\n"
            "2. Download and unzip the dataset.\n"
            f"3. Place the class folders (Cardboard/, Glass/, Metal/, ...) directly under:\n"
            f"   {raw_dir}\n"
            "4. Re-run this script - it will detect the images and skip downloading again."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
