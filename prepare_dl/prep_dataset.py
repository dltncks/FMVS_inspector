# prep_dataset.py
from __future__ import annotations

from pathlib import Path
import random
import os
import cv2
import numpy as np

# =========================
# CONFIG - EDIT THESE PATHS FOR YOUR SETUP
# =========================
ROOT = Path(os.getenv("FMVS_TRAIN_DATA_DIR", "./data/dl_train"))

# Verify paths
if not ROOT.exists():
    raise RuntimeError(
        f"Training data directory not found: {ROOT}\n"
        f"Please create it or set FMVS_TRAIN_DATA_DIR environment variable."
    )

IMAGES_DIR = ROOT / "images"
MASKS_DIR  = ROOT / "masks"
SPLITS_DIR = ROOT / "splits"

VAL_RATIO = 0.10
RANDOM_SEED = 42

# If you have too few crack masks, enforce at least this many in val
MIN_VAL_POS = 10


def _read_mask_any(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    return m


def _mask_to_binary_u8(m: np.ndarray) -> np.ndarray:
    """
    Convert any CVAT mask (palette/grayscale/RGB) to 1-channel binary uint8 {0,255}.
    """
    if m.ndim == 3:
        # RGB or RGBA -> any nonzero pixel becomes crack
        if m.shape[2] == 4:
            m = m[:, :, :3]
        gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        binm = (gray > 0).astype(np.uint8) * 255
        return binm

    # already single channel
    binm = (m > 0).astype(np.uint8) * 255
    return binm


def _make_empty_mask_like(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]
    return np.zeros((h, w), dtype=np.uint8)


def main():
    if not IMAGES_DIR.exists():
        raise RuntimeError(f"Missing images dir: {IMAGES_DIR}")

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(IMAGES_DIR.glob("*.png"))
    if not img_paths:
        raise RuntimeError(f"No PNG images found in {IMAGES_DIR}")

    # 1) Ensure every image has a mask; if missing, create empty
    missing = 0
    for ip in img_paths:
        mp = MASKS_DIR / ip.name
        if not mp.exists():
            empty = _make_empty_mask_like(ip)
            cv2.imwrite(str(mp), empty)
            missing += 1
    print(f"[STEP1] Missing masks created: {missing}")

    # 2) Normalize all masks to binary {0,255} and count positives
    pos_files = []
    neg_files = []
    weird_values_count = 0

    for ip in img_paths:
        mp = MASKS_DIR / ip.name
        m = _read_mask_any(mp)

        # detect weird values (for info only)
        uniq = np.unique(m)
        if len(uniq) > 3 or (m.ndim == 2 and not set(uniq.tolist()).issubset({0, 1, 2, 255})):
            weird_values_count += 1

        mb = _mask_to_binary_u8(m)
        # overwrite normalized mask
        cv2.imwrite(str(mp), mb)

        if int(np.count_nonzero(mb)) > 0:
            pos_files.append(ip.name)
        else:
            neg_files.append(ip.name)

    print(f"[STEP2] Masks normalized to binary. Weird masks seen: {weird_values_count}")
    print(f"[INFO] Positive (has crack pixels): {len(pos_files)}")
    print(f"[INFO] Negative (empty mask): {len(neg_files)}")

    if len(pos_files) == 0:
        raise RuntimeError("No positive masks found after normalization. Check your exported masks.")

    # 3) Create stratified train/val split by file name
    random.seed(RANDOM_SEED)
    random.shuffle(pos_files)
    random.shuffle(neg_files)

    val_pos_n = max(MIN_VAL_POS, int(round(len(pos_files) * VAL_RATIO)))
    val_pos_n = min(val_pos_n, max(1, len(pos_files) - 1))  # keep at least 1 pos for train
    val_neg_n = int(round(len(neg_files) * VAL_RATIO))

    val_files = pos_files[:val_pos_n] + neg_files[:val_neg_n]
    train_files = pos_files[val_pos_n:] + neg_files[val_neg_n:]

    random.shuffle(train_files)
    random.shuffle(val_files)

    (SPLITS_DIR / "train.txt").write_text("\n".join(train_files) + "\n", encoding="utf-8")
    (SPLITS_DIR / "val.txt").write_text("\n".join(val_files) + "\n", encoding="utf-8")

    print(f"[STEP3] Wrote splits:")
    print(f"  train: {len(train_files)}")
    print(f"  val  : {len(val_files)}")
    print(f"  val positives: {sum(1 for f in val_files if (MASKS_DIR / f).exists() and np.count_nonzero(cv2.imread(str(MASKS_DIR / f), 0))>0)}")


if __name__ == "__main__":
    main()
