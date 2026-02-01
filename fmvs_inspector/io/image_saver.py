# fmvs_inspector/io/image_saver.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2

from fmvs_inspector.utils.time import ms_to_hhmmssmmm_compact
from fmvs_inspector.utils.paths import safe_video_stem


class InspectionImageSaver:
    """Save detection frames (orig + inspected) using video time in the filename.

    Output folder:
      <img_dir>/<video_stem>/

    Filename contract:
      <video_stem>_HHMMSSmmm_orig.<ext>
      <video_stem>_HHMMSSmmm_insp.<ext>

    If the same HHMMSSmmm repeats (rare), suffix is added:
      <video_stem>_HHMMSSmmm_orig_01.<ext>
      <video_stem>_HHMMSSmmm_insp_01.<ext>
    """

    def __init__(self, video_path: str, base_dir: str, ext: str = "png"):
        self.video_path = video_path
        self.video_stem = safe_video_stem(video_path)

        self.base_dir = Path(base_dir)
        self.out_dir = self.base_dir / self.video_stem
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.ext = ext.lstrip(".") or "png"

    def _pick_unique_pair_paths(self, base_time: str) -> Tuple[Path, Path]:
        # Try without suffix first, then _01 ... _9999
        for n in range(0, 10_000):
            suffix = "" if n == 0 else f"_{n:02d}"
            raw = self.out_dir / f"{self.video_stem}_{base_time}_orig{suffix}.{self.ext}"
            insp = self.out_dir / f"{self.video_stem}_{base_time}_insp{suffix}.{self.ext}"
            if not raw.exists() and not insp.exists():
                return raw, insp
        raise RuntimeError("Could not allocate a unique image filename (too many collisions).")

    def save_pair(
        self,
        frame_raw,
        frame_annotated,
        video_time_ms: int,
        frame_index: int,
    ) -> Tuple[str, str]:
        """Write the raw and annotated frames to disk."""
        _ = frame_index  # kept for possible future naming, but currently unused by design
        base_time = ms_to_hhmmssmmm_compact(int(video_time_ms))
        raw_path, insp_path = self._pick_unique_pair_paths(base_time)

        ok1 = cv2.imwrite(str(raw_path), frame_raw)
        ok2 = cv2.imwrite(str(insp_path), frame_annotated)
        if not ok1 or not ok2:
            raise RuntimeError(f"Failed to write images: {raw_path} / {insp_path}")

        return str(raw_path), str(insp_path)
