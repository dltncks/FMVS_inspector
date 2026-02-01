# fmvs_inspector/inspection/run_inspection.py
"""
Main video inspection loop.

Responsibilities:
- Open video, seek to start
- Ask user to select manual polygon ROI
- Run detector (OpenCV or DL) inside ROI
- Apply shape filter, draw results
- Save images + log detections
- Optional debug visualization grid
"""
from __future__ import annotations

from typing import Optional

import cv2

from fmvs_inspector.config.types import RunConfig
from fmvs_inspector.io.logger import InspectionLogger
from fmvs_inspector.io.image_saver import InspectionImageSaver
from fmvs_inspector.roi.ui_select import select_roi_polygon
from fmvs_inspector.roi.manual import StaticROIScheduler
from fmvs_inspector.utils.time import ms_to_hhmmss
from fmvs_inspector.utils.masks import contours_from_mask
from fmvs_inspector.viz.grid import compute_roi_bbox_from_mask, build_debug_grid

from fmvs_inspector.detectors.opencv_blackhat import OpenCVBlackhatCrackDetector
from fmvs_inspector.detectors.dl_crack import DLCrackDetector


def run_inspection(cfg: RunConfig) -> None:
    if cfg.mode not in ("opencv", "dl"):
        raise ValueError("cfg.mode must be 'opencv' or 'dl'")

    # Build detector
    opencv_det: Optional[OpenCVBlackhatCrackDetector] = None
    dl_det: Optional[DLCrackDetector] = None
    if cfg.mode == "opencv":
        opencv_det = OpenCVBlackhatCrackDetector(cfg.opencv)
    else:
        dl_det = DLCrackDetector(cfg.dl)

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {cfg.video_path}")

    # Seek to start time (ms)
    target_start_ms = int(cfg.start_sec * 1000)
    cap.set(cv2.CAP_PROP_POS_MSEC, target_start_ms)
    end_ms = int(cfg.end_sec * 1000) if cfg.end_sec is not None else None

    # Read first frame after seek
    ret, setup_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read first frame. Check video path/codec/seek ability.")

    # Robust absolute time mapping across the initial seek
    raw0 = float(cap.get(cv2.CAP_PROP_POS_MSEC))
    time_offset_ms = float(target_start_ms - raw0)
    setup_pos_ms = int(raw0 + time_offset_ms)

    # Windows
    cv2.namedWindow("Inspection", cv2.WINDOW_NORMAL)
    if cfg.debug.show_debug:
        cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)

    # Manual ROI selection
    roi_poly = select_roi_polygon("Inspection", setup_frame)
    if roi_poly is None:
        cap.release()
        cv2.destroyAllWindows()
        return

    # Setup gray
    setup_gray = cv2.cvtColor(setup_frame, cv2.COLOR_BGR2GRAY)
    setup_gray = cv2.GaussianBlur(setup_gray, (5, 5), 0)

    # Manual ROI scheduler (static)
    tracker = StaticROIScheduler(setup_gray.shape[:2], roi_poly)

    # Init ROI bbox
    roi_mask_init = tracker.current_mask()
    roi_bbox = compute_roi_bbox_from_mask(roi_mask_init, pad=cfg.debug.debug_pad)

    logger = InspectionLogger(cfg.video_path, cfg.log_dir)
    saver = InspectionImageSaver(cfg.video_path, cfg.img_dir)

    frame_counter = 0
    paused = False

    last_frame_raw = setup_frame.copy()
    last_frame_vis = setup_frame.copy()
    last_pos_ms = setup_pos_ms
    last_debug_grid = None

    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_counter += 1

                raw_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                pos_ms = int(raw_ms + time_offset_ms)

                if end_ms is not None and pos_ms > end_ms:
                    break

                frame_raw = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                frame_vis = frame.copy()
                cv2.putText(
                    frame_vis,
                    f"Video Time: {ms_to_hhmmss(pos_ms)}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # --- ROI update (static/manual) ---
                roi_poly_dyn, roi_mask_dyn, _dbg = tracker.update(gray, pos_ms)
                roi_bbox = compute_roi_bbox_from_mask(roi_mask_dyn, pad=cfg.debug.debug_pad)
                cv2.polylines(frame_vis, [roi_poly_dyn], True, (0, 255, 255), 2)

                # --- Detector inference ---
                if cfg.mode == "opencv":
                    assert opencv_det is not None
                    out = opencv_det.infer(gray_full=gray, roi_mask_full=roi_mask_dyn)
                else:
                    assert dl_det is not None
                    out = dl_det.infer(gray_full=gray, roi_mask_full=roi_mask_dyn, roi_bbox=roi_bbox)

                vote_mask = out["vote_mask"]
                debug_imgs = out["debug"]

                # --- Debug grid (optional) ---
                if cfg.debug.show_debug:
                    last_debug_grid = build_debug_grid(
                        bh=debug_imgs["bh"],
                        thresh_vote=debug_imgs["thresh_vote"],
                        roi=debug_imgs["roi"],
                        thresh=debug_imgs["thresh"],
                        bbox=roi_bbox,
                        tile_size=cfg.debug.debug_tile_size,
                    )

                # --- Contours + shape filtering ---
                detected = False
                det_count = 0
                contours = contours_from_mask(vote_mask)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < cfg.shape.min_area:
                        continue

                    rect = cv2.minAreaRect(cnt)
                    (w, h) = rect[1]
                    if w < 1 or h < 1:
                        continue

                    long_side = max(w, h)
                    short_side = min(w, h)
                    aspect = (long_side / short_side) if short_side > 0 else 0.0

                    if long_side > cfg.shape.min_long_side and aspect > cfg.shape.min_aspect:
                        box = cv2.boxPoints(rect).astype(int)
                        cv2.drawContours(frame_vis, [box], 0, (0, 0, 255), 2)
                        cv2.putText(
                            frame_vis,
                            "CRACK",
                            tuple(box[0]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                        detected = True
                        det_count += 1

                # --- Save + log when detected ---
                if detected:
                    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if frame_index <= 0:
                        frame_index = frame_counter

                    saver.save_pair(
                        frame_raw=frame_raw,
                        frame_annotated=frame_vis,
                        frame_index=frame_index,
                        video_time_ms=pos_ms,
                    )
                    logger.log_detection(
                        frame_index=frame_index,
                        video_time_ms=pos_ms,
                        det_count=det_count,
                    )

                if cfg.pause_on_detect and detected:
                    paused = True

                last_frame_raw = frame_raw
                last_frame_vis = frame_vis
                last_pos_ms = pos_ms

            # --- Display ---
            frame_show = last_frame_vis.copy()
            if paused:
                cv2.putText(
                    frame_show,
                    "PAUSED: SPACE=resume | r=reselect ROI | q=quit",
                    (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Inspection", frame_show)

            if cfg.debug.show_debug and last_debug_grid is not None:
                cv2.imshow("Debug", last_debug_grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:  # SPACE
                paused = not paused

            # Reselect ROI (paused)
            if paused and key == ord("r"):
                new_poly = select_roi_polygon("Inspection", last_frame_raw)
                if new_poly is not None and len(new_poly) >= 3:
                    # reset ROI scheduler
                    new_gray = cv2.cvtColor(last_frame_raw, cv2.COLOR_BGR2GRAY)
                    new_gray = cv2.GaussianBlur(new_gray, (5, 5), 0)

                    tracker = StaticROIScheduler(new_gray.shape[:2], new_poly)

                    # reset detector history
                    if opencv_det is not None:
                        opencv_det.reset()
                    if dl_det is not None:
                        dl_det.reset()

                    last_frame_vis = last_frame_raw.copy()
                    cv2.putText(
                        last_frame_vis,
                        f"Video Time: {ms_to_hhmmss(last_pos_ms)}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        last_frame_vis,
                        "ROI updated.",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

    finally:
        logger.close()
        cap.release()
        cv2.destroyAllWindows()
