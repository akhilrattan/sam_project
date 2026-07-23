"""
Downloads the SAM ViT-B checkpoint (~375 MB) if not already present.

Usage:
    python -m src.annotation_generation.download_sam_checkpoint
"""
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logger import load_config, resolve_path, get_logger

logger = get_logger("download_sam_checkpoint")


def main():
    cfg = load_config()
    ckpt_path = resolve_path(cfg["sam"]["checkpoint_path"])
    url = cfg["sam"]["checkpoint_url"]

    if ckpt_path.exists():
        logger.info("SAM checkpoint already present at %s", ckpt_path)
        return

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading SAM checkpoint from %s ...", url)

    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(ckpt_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info("Saved SAM checkpoint to %s", ckpt_path)


if __name__ == "__main__":
    main()
