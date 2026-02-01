# fmvs_inspector/roi/ui_select.py
"""
Manual polygon ROI selection UI using OpenCV mouse callbacks.

Controls:
- Left click: add point
- Right click: finish polygon
- 'c': clear
- 'z': undo last point
- 'q': cancel
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def select_roi_polygon(window_name: str, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    pts: list[tuple[int, int]] = []
    is_done = False

    def draw_roi(event, x, y, flags, param):
        nonlocal pts, is_done
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            is_done = True

    cv2.setMouseCallback(window_name, draw_roi)

    while True:
        img_draw = frame_bgr.copy()
        for p in pts:
            cv2.circle(img_draw, p, 3, (0, 255, 255), -1)
        if len(pts) > 1:
            cv2.polylines(img_draw, [np.array(pts, dtype=np.int32)], True, (0, 255, 255), 2)

        cv2.putText(
            img_draw,
            "ROI(YELLOW): LClick=add | RClick=finish | c=clear | z=undo | q=cancel",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, img_draw)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.setMouseCallback(window_name, lambda *args: None)
            return None
        if key == ord("c"):
            pts.clear()
            is_done = False
        if key == ord("z") and pts:
            pts.pop()
            is_done = False

        if is_done:
            if len(pts) < 3:
                is_done = False
                continue
            roi_poly = np.array(pts, dtype=np.int32)
            cv2.setMouseCallback(window_name, lambda *args: None)
            return roi_poly
