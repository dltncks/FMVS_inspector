import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional, Dict, Any

from prepare_dl.dl_dataset import DatasetExportConfig, DatasetWriter


# =========================
# VIDEO CONFIG - EDIT THESE PATHS FOR YOUR SETUP
# =========================
VIDEO_PATH = os.getenv("FMVS_VIDEO_PATH", "./videos/sample.mp4")
START_SEC = 0  # Start time in seconds
END_SEC = None  # None for full video

# Where to save dataset
DATASET_DIR = os.getenv("FMVS_DATASET_DIR", "./data/dl_dataset")

# Display toggles
SHOW_AUTO_ROI_OVERLAY = False  # Disabled AutoROI Overlay
SHOW_IGNORE_STATUS = False  # Disabled Ignore Status

# =========================
# Dataset export config
# =========================
EXPORT_CFG = dict(
    crop_pad_px=10,          # padding around ROI bbox in the saved crop
    save_masked_gray=False,   # recommended for training (outside ROI -> 0)
    save_raw_gray=True,     # turn on if you want also unmasked crop
    save_preview_bgr=True,   # preview crop with outlines
)


def polygon_to_mask(shape_hw, poly):
    """Convert polygon to binary mask.
    
    Args:
        shape_hw: (height, width) tuple
        poly: Numpy array of polygon points
        
    Returns:
        Binary mask (uint8) with polygon filled as 255
    """
    m = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    return m


def ms_to_hhmmss(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS.mmm format."""
    if ms < 0:
        ms = 0
    total_sec = ms // 1000
    mmm = ms % 1000
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{mmm:03d}"


def select_roi_polygon(window_name: str, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Interactive polygon ROI selection.
    
    Controls:
        - Left click: Add point
        - Right click: Finish polygon
        - 'c': Clear points
        - 'z': Undo last point
        - 'q': Cancel
    """
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


def _safe_tint(frame_bgr: np.ndarray, mask_u8: np.ndarray, add_bgr: tuple[int, int, int]) -> None:
    """Tint image regions defined by mask (in-place)."""
    if mask_u8 is None or not np.any(mask_u8 > 0):
        return
    add = np.array(add_bgr, dtype=np.int16)
    region = frame_bgr[mask_u8 > 0].astype(np.int16)
    region += add
    np.clip(region, 0, 255, out=region)
    frame_bgr[mask_u8 > 0] = region.astype(np.uint8)


def main():
    # Verify video exists
    if not Path(VIDEO_PATH).exists():
        raise RuntimeError(
            f"Video not found: {VIDEO_PATH}\n"
            f"Please set FMVS_VIDEO_PATH environment variable or edit VIDEO_PATH in this script."
        )
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    target_start_ms = int(START_SEC * 1000)
    cap.set(cv2.CAP_PROP_POS_MSEC, target_start_ms)
    end_ms = int(END_SEC * 1000) if END_SEC is not None else None

    ret, setup_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read first frame. Check video path/codec/seek ability.")

    # Stable absolute time mapping across seeks
    raw0 = float(cap.get(cv2.CAP_PROP_POS_MSEC))
    time_offset_ms = float(target_start_ms - raw0)
    setup_pos_ms = int(raw0 + time_offset_ms)
    expected_abs_after_seek_ms: Optional[int] = None

    cv2.namedWindow("Collector", cv2.WINDOW_NORMAL)

    print("[INFO] Select initial ROI (yellow) on the setup frame.")
    roi_poly = select_roi_polygon("Collector", setup_frame)
    if roi_poly is None:
        cap.release()
        cv2.destroyAllWindows()
        return

    # Init gray
    setup_gray = cv2.cvtColor(setup_frame, cv2.COLOR_BGR2GRAY)
    setup_gray = cv2.GaussianBlur(setup_gray, (5, 5), 0)

    writer = DatasetWriter(
        DatasetExportConfig(
            base_dir=DATASET_DIR,
            video_stem=Path(VIDEO_PATH).stem or "video",
            **EXPORT_CFG,
        )
    )

    paused = False
    saved_count = 0

    help_lines = [
        "Keys:",
        "  SPACE : pause/resume",
        "  s     : save sample now",
        "  q     : quit",
    ]

    last_frame_vis = setup_frame.copy()

    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                raw_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))

                # Process frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                vis = frame.copy()
                cv2.putText(vis, f"t={ms_to_hhmmss(raw_ms)}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.polylines(vis, [roi_poly], True, (0, 255, 255), 2)

                last_frame_vis = vis

                cv2.imshow("Collector", last_frame_vis)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == 32:  # SPACE
                    paused = not paused

                if key == ord("s"):
                    # Save now
                    out = writer.save_sample(
                        frame_bgr=frame,
                        gray=gray,
                        roi_mask_full=polygon_to_mask(gray.shape[:2], roi_poly),
                        roi_poly_full=roi_poly,
                        pos_ms=raw_ms,
                        frame_index=saved_count,
                    )
                    saved_count += 1
                    cv2.putText(last_frame_vis, f"SAVED #{saved_count}", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[DONE] Saved {saved_count} samples to {DATASET_DIR}")


if __name__ == "__main__":
    main()
