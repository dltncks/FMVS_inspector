# fmvs_inspector/detectors/dl_crack.py
"""
DL crack detector wrapper around CrackSegPredictor.

Responsibilities:
- crop to ROI bbox (matches your training/inference style)
- apply ROI mask
- run predictor -> prob -> binary mask
- quick min-pixels filter
- history OR vote
- return full-frame vote mask + debug images compatible with viz.grid.build_debug_grid
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Any, Tuple

import cv2
import numpy as np

from fmvs_inspector.config.types import DLRunConfig
from fmvs_inspector.detectors.dl_unet import CrackSegPredictor


class DLCrackDetector:
    def __init__(self, cfg: DLRunConfig):
        self.cfg = cfg
        self.predictor = CrackSegPredictor(cfg.model)
        self.hist = deque(maxlen=max(1, int(cfg.history)))

    def reset(self) -> None:
        self.hist.clear()

    def infer(
        self,
        gray_full: np.ndarray,
        roi_mask_full: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
    ) -> Dict[str, Any]:
        x1, y1, x2, y2 = roi_bbox
        gray_crop = gray_full[y1:y2, x1:x2]
        roi_mask_crop = roi_mask_full[y1:y2, x1:x2]

        # mask outside ROI to reduce distractions
        gray_crop_masked = cv2.bitwise_and(gray_crop, gray_crop, mask=roi_mask_crop)

        prob = self.predictor.predict_prob(gray_crop_masked)
        pred_crop = self.predictor.prob_to_mask(prob)
        pred_crop = cv2.bitwise_and(pred_crop, pred_crop, mask=roi_mask_crop)

        # quick “any crack?” filter
        if int(np.count_nonzero(pred_crop)) >= int(self.cfg.min_pixels):
            self.hist.append(pred_crop)
        else:
            self.hist.append(np.zeros_like(pred_crop, dtype=np.uint8))

        vote_crop = self.hist[0].copy()
        for i in range(1, len(self.hist)):
            vote_crop = cv2.bitwise_or(vote_crop, self.hist[i])

        # place back to full-frame mask
        vote_full = np.zeros_like(gray_full, dtype=np.uint8)
        vote_full[y1:y2, x1:x2] = vote_crop

        # debug images (match viz grid interface)
        prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        bh_like = np.zeros_like(gray_full, dtype=np.uint8)
        bh_like[y1:y2, x1:x2] = prob_u8

        thresh_like = np.zeros_like(gray_full, dtype=np.uint8)
        thresh_like[y1:y2, x1:x2] = pred_crop

        roi_like = cv2.bitwise_and(gray_full, gray_full, mask=roi_mask_full)

        return {
            "vote_mask": vote_full,
            "debug": {
                "bh": bh_like,
                "thresh_vote": vote_full,
                "roi": roi_like,
                "thresh": thresh_like,
            },
        }
