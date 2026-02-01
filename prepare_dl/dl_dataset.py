# dl_dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np


def ms_to_hhmmssmmm(ms: int) -> str:
    """Convert milliseconds to 'HHMMSSmmm' (e.g., 02:45:50.156 -> '024550156')."""
    if ms < 0:
        ms = 0
    total_sec = ms // 1000
    mmm = ms % 1000
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60
    return f"{hh:02d}{mm:02d}{ss:02d}{mmm:03d}"


def compute_bbox_from_mask(mask_u8: np.ndarray, pad: int = 0) -> Tuple[int, int, int, int]:
    """Return (x1,y1,x2,y2) slice bbox around nonzero mask, with pad."""
    ys, xs = np.where(mask_u8 > 0)
    h, w = mask_u8.shape[:2]
    if xs.size == 0 or ys.size == 0:
        return 0, 0, w, h
    x1 = max(int(xs.min()) - pad, 0)
    y1 = max(int(ys.min()) - pad, 0)
    x2 = min(int(xs.max()) + 1 + pad, w)
    y2 = min(int(ys.max()) + 1 + pad, h)
    return x1, y1, x2, y2


@dataclass
class DatasetExportConfig:
    base_dir: str
    video_stem: str

    crop_pad_px: int = 10
    save_masked_gray: bool = False  # image pixels outside ROI -> 0
    save_raw_gray: bool = True    # optionally save unmasked gray crop too
    save_preview_bgr: bool = True  # overlay outline for quick review


class DatasetWriter:
    """
    Writes training samples:
      - images/ : masked gray ROI crop (recommended input for segmentation)
      - roi_masks/ : ROI polygon mask crop (same size as image crop)
      - raw_gray/ : optional gray crop without masking
      - previews/ : optional BGR preview with outlines
      - meta/ : json metadata (timestamp, bbox, polys, stats)
    """

    def __init__(self, cfg: DatasetExportConfig):
        self.cfg = cfg
        self.root = Path(cfg.base_dir) / cfg.video_stem
        self.dir_images = self.root / "images"
        self.dir_masks = self.root / "roi_masks"
        self.dir_raw = self.root / "raw_gray"
        self.dir_prev = self.root / "previews"
        self.dir_meta = self.root / "meta"

        self.dir_images.mkdir(parents=True, exist_ok=True)
        self.dir_masks.mkdir(parents=True, exist_ok=True)
        if cfg.save_raw_gray:
            self.dir_raw.mkdir(parents=True, exist_ok=True)
        if cfg.save_preview_bgr:
            self.dir_prev.mkdir(parents=True, exist_ok=True)
        self.dir_meta.mkdir(parents=True, exist_ok=True)

    def _unique_base(self, base: str) -> str:
        # Avoid collisions when CAP_PROP_POS_MSEC repeats
        p_img = self.dir_images / f"{base}.png"
        if not p_img.exists():
            return base
        for i in range(1, 10000):
            b2 = f"{base}_{i:04d}"
            if not (self.dir_images / f"{b2}.png").exists():
                return b2
        raise RuntimeError("Too many duplicate filenames in dataset export.")

    def save_sample(
        self,
        frame_bgr: np.ndarray,
        gray: np.ndarray,
        roi_mask_full: np.ndarray,
        roi_poly_full: np.ndarray,
        pos_ms: int,
        frame_index: int,
        label_tag: str = "unk",
        extra_meta: Optional[Dict[str, Any]] = None,
        preview_bgr: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """
        Saves one sample and returns dict of paths.
        """
        x1, y1, x2, y2 = compute_bbox_from_mask(roi_mask_full, pad=self.cfg.crop_pad_px)
        gray_crop = gray[y1:y2, x1:x2]
        mask_crop = roi_mask_full[y1:y2, x1:x2]

        if self.cfg.save_masked_gray:
            img_crop = cv2.bitwise_and(gray_crop, gray_crop, mask=mask_crop)
        else:
            img_crop = gray_crop

        ts = ms_to_hhmmssmmm(int(pos_ms))
        base0 = f"{ts}_f{int(frame_index):07d}_{label_tag}"
        base = self._unique_base(base0)

        p_img = self.dir_images / f"{base}.png"
        p_mask = self.dir_masks / f"{base}_roi.png"
        cv2.imwrite(str(p_img), img_crop)
        cv2.imwrite(str(p_mask), mask_crop)

        out: Dict[str, str] = {"image": str(p_img), "roi_mask": str(p_mask)}

        if self.cfg.save_raw_gray:
            p_raw = self.dir_raw / f"{base}_raw.png"
            cv2.imwrite(str(p_raw), gray_crop)
            out["raw_gray"] = str(p_raw)

        if self.cfg.save_preview_bgr:
            if preview_bgr is None:
                # simple preview: crop BGR + draw ROI outline in crop coords
                prev = frame_bgr[y1:y2, x1:x2].copy()
                poly_crop = (roi_poly_full.astype(np.int32) - np.array([[x1, y1]], dtype=np.int32))
                cv2.polylines(prev, [poly_crop], True, (0, 255, 255), 2)
            else:
                prev = preview_bgr[y1:y2, x1:x2].copy()

            p_prev = self.dir_prev / f"{base}_prev.png"
            cv2.imwrite(str(p_prev), prev)
            out["preview"] = str(p_prev)

        meta = {
            "pos_ms": int(pos_ms),
            "frame_index": int(frame_index),
            "label_tag": label_tag,
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "roi_poly_full": roi_poly_full.astype(int).tolist(),
        }
        if extra_meta:
            meta.update(extra_meta)

        p_meta = self.dir_meta / f"{base}.json"
        p_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        out["meta"] = str(p_meta)

        return out
