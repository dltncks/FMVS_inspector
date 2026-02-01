# train_unet.py
from __future__ import annotations

from pathlib import Path
import time
import random
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# =========================
# CONFIG - EDIT THESE PATHS FOR YOUR SETUP
# =========================
# Use relative paths from repo root, or set via environment variables
ROOT = Path(os.getenv("FMVS_TRAIN_DATA_DIR", "./data/dl_train"))
OUT_DIR = Path(os.getenv("FMVS_MODEL_OUTPUT_DIR", "./dl_models"))

# Verify paths exist before proceeding
if not ROOT.exists():
    raise RuntimeError(
        f"Training data directory not found: {ROOT}\n"
        f"Please create it or set FMVS_TRAIN_DATA_DIR environment variable.\n"
        f"Expected structure:\n"
        f"  {ROOT}/images/\n"
        f"  {ROOT}/masks/\n"
        f"  {ROOT}/splits/"
    )

IMAGES_DIR = ROOT / "images"
MASKS_DIR  = ROOT / "masks"
SPLITS_DIR = ROOT / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# With RTX 4090 laptop, you can usually raise these.
# If you get CUDA OOM, lower BATCH_SIZE first.
INPUT_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3

# Windows: 2~8 is usually fine. Start at 4.
NUM_WORKERS = 4

# Mixed precision = huge speedup on 4090
USE_AMP = True

# Augmentation strengths (tune later)
AUG_FLIP = True
AUG_BRIGHTNESS = 0.20
AUG_CONTRAST = 0.20
AUG_BLUR_PROB = 0.10


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resize_pair(img: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    img_rs = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mask_rs = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img_rs, mask_rs


def augment(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if AUG_FLIP and random.random() < 0.5:
        img = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()
    if AUG_FLIP and random.random() < 0.5:
        img = np.flipud(img).copy()
        mask = np.flipud(mask).copy()

    if AUG_BRIGHTNESS > 0 or AUG_CONTRAST > 0:
        b = 1.0 + random.uniform(-AUG_BRIGHTNESS, AUG_BRIGHTNESS)
        c = 1.0 + random.uniform(-AUG_CONTRAST, AUG_CONTRAST)
        f = img.astype(np.float32)
        f = f * c + (b - 1.0) * 128.0
        img = np.clip(f, 0, 255).astype(np.uint8)

    if random.random() < AUG_BLUR_PROB:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img, mask


class CrackSegDataset(Dataset):
    def __init__(self, filenames: list[str], train: bool):
        self.filenames = filenames
        self.train = train

        self.is_pos = []
        for fn in self.filenames:
            m = cv2.imread(str(MASKS_DIR / fn), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise RuntimeError(f"Missing mask: {MASKS_DIR / fn}")
            self.is_pos.append(int(np.count_nonzero(m) > 0))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fn = self.filenames[idx]
        ip = IMAGES_DIR / fn
        mp = MASKS_DIR / fn

        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {ip}")
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mp}")

        mask = (mask > 0).astype(np.uint8) * 255

        img, mask = resize_pair(img, mask, INPUT_SIZE)

        if self.train:
            img, mask = augment(img, mask)

        # tensors
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)      # (1,H,W)
        y = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)       # (1,H,W)
        return x, y, fn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(base*4, base*8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base*8, base*16)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = conv_block(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = conv_block(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_block(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = conv_block(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.up4(b); d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4); d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3); d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return 1.0 - (num / den).mean()


@torch.no_grad()
def dice_iou_metrics(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    pred = (probs > thr).float()
    inter = (pred * targets).sum(dim=(2, 3))
    union = (pred + targets - pred * targets).sum(dim=(2, 3)) + eps
    iou = (inter + eps) / union
    dice = (2*inter + eps) / (pred.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps)
    return float(dice.mean().cpu()), float(iou.mean().cpu())


def main():
    set_seed(42)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch:", torch.__version__)
    print("cuda :", torch.cuda.is_available())
    print("gpu  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
    print("device:", device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = bool(USE_AMP and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_list = read_list(SPLITS_DIR / "train.txt")
    val_list = read_list(SPLITS_DIR / "val.txt")

    train_ds = CrackSegDataset(train_list, train=True)
    val_ds   = CrackSegDataset(val_list, train=False)

    pos_count = sum(train_ds.is_pos)
    neg_count = len(train_ds) - pos_count
    if pos_count == 0:
        raise RuntimeError("No positive samples in train split.")

    pos_w = neg_count / max(1, pos_count)
    weights = [pos_w if p == 1 else 1.0 for p in train_ds.is_pos]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    model = UNet(in_ch=1, out_ch=1, base=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()

    best_val_dice = -1.0
    ckpt_path = OUT_DIR / "unet_crack_best.pt"
    print(f"[INFO] saving best checkpoint to: {ckpt_path}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0

        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = bce(logits, y) + dice_loss(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += float(loss.item()) * x.size(0)

        tr_loss /= max(1, len(train_ds))

        model.eval()
        va_loss = 0.0
        dices, ious = [], []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    loss = bce(logits, y) + dice_loss(logits, y)
                va_loss += float(loss.item()) * x.size(0)

                d, i = dice_iou_metrics(logits, y, thr=0.5)
                dices.append(d); ious.append(i)

        va_loss /= max(1, len(val_ds))
        va_dice = float(np.mean(dices)) if dices else 0.0
        va_iou  = float(np.mean(ious)) if ious else 0.0

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{EPOCHS} | tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} va_dice={va_dice:.4f} va_iou={va_iou:.4f} time={dt:.1f}s")

        if va_dice > best_val_dice:
            best_val_dice = va_dice
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": best_val_dice}, str(ckpt_path))
            print(f"  [SAVE] best updated: val_dice={best_val_dice:.4f}")

    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()
