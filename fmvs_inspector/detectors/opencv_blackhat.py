# fmvs_inspector/detectors/opencv_blackhat.py
"""
OpenCV crack detector.

This module contains ONLY the OpenCV crack pipeline:
- ROI masking
- optional CLAHE
- multi-kernel BlackHat
- percentile threshold
- morphology close
- history OR vote
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Any

import cv2
import numpy as np

from fmvs_inspector.config.types import OpenCVConfig


class OpenCVBlackhatCrackDetector:
    def __init__(self, cfg: OpenCVConfig):
        self.cfg = cfg
        self.hist = deque(maxlen=max(1, int(cfg.history)))
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if cfg.use_clahe else None

    def reset(self) -> None:
        self.hist.clear()

    def infer(self, gray_full: np.ndarray, roi_mask_full: np.ndarray) -> Dict[str, Any]:
        roi = cv2.bitwise_and(gray_full, gray_full, mask=roi_mask_full)

        if self.clahe is not None:
            roi = self.clahe.apply(roi)
            roi = cv2.bitwise_and(roi, roi, mask=roi_mask_full)

        bh = np.zeros_like(roi, dtype=np.uint8)
        for k in self.cfg.kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(k), int(k)))
            bh_k = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, kernel)
            bh = cv2.max(bh, bh_k)

        vals = bh[roi_mask_full == 255]
        t = np.percentile(vals, float(self.cfg.percentile)) if vals.size > 0 else 30

        _, thresh = cv2.threshold(bh, t, 255, cv2.THRESH_BINARY)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.cfg.close_ksize)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask_full)

        self.hist.append(thresh)

        thresh_vote = self.hist[0].copy()
        for i in range(1, len(self.hist)):
            thresh_vote = cv2.bitwise_or(thresh_vote, self.hist[i])
        thresh_vote = cv2.bitwise_and(thresh_vote, thresh_vote, mask=roi_mask_full)

        return {
            "vote_mask": thresh_vote,
            "debug": {
                "bh": bh,
                "thresh_vote": thresh_vote,
                "roi": roi,
                "thresh": thresh,
            },
        }
