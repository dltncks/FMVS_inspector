# fmvs_inspector/detectors/dl_unet.py
"""
DL U-Net + predictor.

This is essentially your previous dl_infer.py, moved into detectors/.
Keep model architecture + preprocessing consistent with training.
"""
from __future__ import annotations

from typing import Optional, Tuple


from fmvs_inspector.config.types import DLConfig
import cv2
import numpy as np
import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)  # logits



class CrackSegPredictor:
    """
    Predicts a crack probability map / binary mask on a grayscale ROI crop.
    Training used single-channel 0..1 input and resized to (input_size,input_size).
    """

    def __init__(self, cfg: DLConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = UNet(in_ch=1, out_ch=1, base=cfg.base_channels).to(self.device)
        self.model.eval()

        ckpt = torch.load(cfg.ckpt_path, map_location=self.device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)

        # speed knobs
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        self._open_k = self._kernel(cfg.morph_open)
        self._close_k = self._kernel(cfg.morph_close)

    @staticmethod
    def _kernel(ksize: Tuple[int, int]) -> Optional[np.ndarray]:
        kx, ky = int(ksize[0]), int(ksize[1])
        if kx <= 1 and ky <= 1:
            return None
        kx = kx if (kx % 2 == 1) else (kx + 1)
        ky = ky if (ky % 2 == 1) else (ky + 1)
        return cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))

    @torch.inference_mode()
    def predict_prob(self, gray_crop_u8: np.ndarray) -> np.ndarray:
        """
        gray_crop_u8: uint8 grayscale (H,W)
        returns prob float32 (H,W) in [0,1] (same size as input crop)
        """
        H, W = gray_crop_u8.shape[:2]
        s = int(self.cfg.input_size)

        img_rs = cv2.resize(gray_crop_u8, (s, s), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img_rs.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)  # (1,1,s,s)
        x = x.to(self.device, non_blocking=True)

        if self.device.startswith("cuda") and self.cfg.use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                logits = self.model(x)
        else:
            logits = self.model(x)

        prob = torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy()  # (s,s)
        prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return prob

    def prob_to_mask(self, prob: np.ndarray) -> np.ndarray:
        """
        prob float32 (H,W) -> mask uint8 {0,255}
        """
        m = (prob >= float(self.cfg.mask_thr)).astype(np.uint8) * 255

        # optional cleanup (keep kernels small for thin lines)
        if self._close_k is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._close_k)
        if self._open_k is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._open_k)

        return m
