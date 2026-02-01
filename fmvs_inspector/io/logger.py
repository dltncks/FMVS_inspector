# fmvs_inspector/io/logger.py
"""Detection logger.

Writes a simple text log per video to:
  <log_dir>/<video_stem>/<video_stem>_detections.txt

Each line contains the **video time** (HH:MM:SS.mmm) so the detection can be located again.
"""
from __future__ import annotations

from pathlib import Path

from fmvs_inspector.utils.time import ms_to_hhmmssmmm
from fmvs_inspector.utils.paths import safe_video_stem


class InspectionLogger:
    def __init__(self, video_path: str, base_dir: str):
        self.video_path = video_path
        self.video_stem = safe_video_stem(video_path)

        self.base_dir = Path(base_dir)
        self.out_dir = self.base_dir / self.video_stem
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.out_path = self.out_dir / f"{self.video_stem}_detections.txt"
        self._fh = self.out_path.open("a", encoding="utf-8")

    def log_detection(self, frame_index: int, video_time_ms: int, det_count: int) -> None:
        t_str = ms_to_hhmmssmmm(int(video_time_ms))
        self._fh.write(f"{t_str}  (frame={int(frame_index):07d}, count={int(det_count)})\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()

    def __enter__(self) -> "InspectionLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
