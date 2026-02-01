# fmvs_inspector/roi/manual.py
"""
Manual/static ROI scheduler.

This keeps the same practical contract as your current code:
update(gray, ms) -> (roi_poly, roi_mask, dbg)
"""
from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np

from fmvs_inspector.utils.masks import polygon_to_mask


class StaticROIScheduler:
    def __init__(self, frame_shape_hw: Tuple[int, int], roi_poly_init: np.ndarray):
        self.h, self.w = frame_shape_hw
        self.roi_poly = roi_poly_init.astype(np.int32).copy()

    def current_mask(self) -> np.ndarray:
        return polygon_to_mask((self.h, self.w), self.roi_poly)

    def update(self, gray_full: np.ndarray, video_time_ms: int):
        roi_mask = self.current_mask()
        dbg: Dict[str, Any] = {"skipped": True, "updated": False}
        return self.roi_poly.copy(), roi_mask, dbg
