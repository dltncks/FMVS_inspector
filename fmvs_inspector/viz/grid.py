# fmvs_inspector/viz/grid.py
"""
Diagnostic visualization grid.

Shows (cropped around ROI bbox):
- BH (or DL prob map) [normalized for view]
- THRESH vote
- ROI image [normalized for view]
- THRESH (single-frame threshold / DL mask)
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_roi_bbox_from_mask(mask: np.ndarray, pad: int = 20) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape[:2]
    if xs.size == 0 or ys.size == 0:
        return 0, 0, w, h

    x1 = max(int(xs.min()) - pad, 0)
    y1 = max(int(ys.min()) - pad, 0)
    x2 = min(int(xs.max()) + 1 + pad, w)
    y2 = min(int(ys.max()) + 1 + pad, h)
    return x1, y1, x2, y2


def _crop_resize_for_debug(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    tile_size: tuple[int, int],
    is_mask: bool,
    normalize_view: bool,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        crop = np.zeros((tile_size[1], tile_size[0]), dtype=np.uint8)
        return cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    if normalize_view and not is_mask:
        crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    crop_rs = cv2.resize(crop, tile_size, interpolation=interp)

    if crop_rs.ndim == 2:
        return cv2.cvtColor(crop_rs, cv2.COLOR_GRAY2BGR)
    return crop_rs


def build_debug_grid(
    bh: np.ndarray,
    thresh_vote: np.ndarray,
    roi: np.ndarray,
    thresh: np.ndarray,
    bbox: tuple[int, int, int, int],
    tile_size: tuple[int, int] = (420, 420),
) -> np.ndarray:
    tile_bh = _crop_resize_for_debug(bh, bbox, tile_size, is_mask=False, normalize_view=True)
    tile_vote = _crop_resize_for_debug(thresh_vote, bbox, tile_size, is_mask=True, normalize_view=False)
    tile_roi = _crop_resize_for_debug(roi, bbox, tile_size, is_mask=False, normalize_view=True)
    tile_thresh = _crop_resize_for_debug(thresh, bbox, tile_size, is_mask=True, normalize_view=False)

    cv2.putText(tile_bh, "BH / PROB (normalized for view)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(tile_vote, "THRESH_VOTE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(tile_roi, "ROI (normalized for view)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(tile_thresh, "THRESH / PRED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    top = np.hstack([tile_bh, tile_vote])
    bot = np.hstack([tile_roi, tile_thresh])
    grid = np.vstack([top, bot])

    cv2.putText(grid, "DEBUG GRID", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return grid
