# make_empty_masks.py
from pathlib import Path
import os
import cv2
import numpy as np

# =========================
# CONFIG - EDIT THESE PATHS FOR YOUR SETUP
# =========================
ROOT = Path(os.getenv("FMVS_TRAIN_DATA_DIR", "./data/dl_train"))

IMAGES_DIR = ROOT / "images"
MASKS_DIR  = ROOT / "masks"

# mask pixel values:
# 0   = background (no crack)
# 255 = crack (you'll draw these later for crack images)
BACKGROUND_VALUE = 0


def main():
    if not IMAGES_DIR.exists():
        raise RuntimeError(
            f"Images directory not found: {IMAGES_DIR}\n"
            f"Please create it or set FMVS_TRAIN_DATA_DIR environment variable."
        )
    
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in IMAGES_DIR.glob("*.png")])
    if not img_paths:
        raise RuntimeError(f"No .png images found in {IMAGES_DIR}")

    created = 0
    skipped = 0

    for ip in img_paths:
        mp = MASKS_DIR / ip.name
        if mp.exists():
            skipped += 1
            continue

        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not read: {ip}")
            continue

        h, w = img.shape[:2]
        mask = np.full((h, w), BACKGROUND_VALUE, dtype=np.uint8)
        ok = cv2.imwrite(str(mp), mask)
        if not ok:
            raise RuntimeError(f"Failed to write mask: {mp}")
        created += 1

    print(f"[DONE] created={created}, skipped(existing)={skipped}, total_images={len(img_paths)}")
    print(f"Masks folder: {MASKS_DIR}")


if __name__ == "__main__":
    main()
