# fmvs_inspector/inspection/config.py
"""Default runtime configuration.

This module provides a *committed* DEFAULT_CONFIG that uses **relative paths** so it
works for other machines when the repo is cloned.

Local override pattern (recommended for your own machine):
- Create: fmvs_inspector/inspection/config_local.py
- Define: DEFAULT_CONFIG = RunConfig(...)
- That file is ignored by .gitignore and will override this DEFAULT_CONFIG at import time.
"""
from __future__ import annotations

from fmvs_inspector.config.types import (
    RunConfig,
    DebugConfig,
    ShapeFilterConfig,
    OpenCVConfig,
    DLConfig,
    DLRunConfig,
)


DEFAULT_CONFIG = RunConfig(
    # Use repo-relative paths (run from repo root)
    video_path="FMVS_videos/sample.mp4",
    log_dir="logs",
    img_dir="images",

    # Example time window (seconds)
    start_sec=2 * 3600 + 45 * 60 + 54,
    end_sec=None,

    pause_on_detect=False,
    mode="opencv",  # "opencv" or "dl"

    opencv=OpenCVConfig(
        kernel_sizes=(7, 15, 25),
        percentile=99.6,
        close_ksize=(5, 5),
        history=3,
        use_clahe=True,
    ),

    dl=DLRunConfig(
        model=DLConfig(
            ckpt_path="dl_models/unet_crack_best.pt",
            input_size=512,
            mask_thr=0.45,
            base_channels=32,
            use_amp=True,
            morph_open=(1, 1),
            morph_close=(3, 3),
        ),
        min_pixels=20,
        history=2,
    ),

    shape=ShapeFilterConfig(
        min_area=50,
        min_long_side=40,
        min_aspect=2.0,
    ),

    debug=DebugConfig(
        show_debug=False,
        debug_pad=20,
        debug_tile_size=(420, 420),
    ),
)


# ----------------------------
# Local override (optional)
# ----------------------------
try:
    from .config_local import DEFAULT_CONFIG as _LOCAL_DEFAULT_CONFIG  # type: ignore
except ModuleNotFoundError:
    _LOCAL_DEFAULT_CONFIG = None

if _LOCAL_DEFAULT_CONFIG is not None:
    DEFAULT_CONFIG = _LOCAL_DEFAULT_CONFIG
