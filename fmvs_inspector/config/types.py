# fmvs_inspector/config/types.py
"""Shared configuration dataclasses.

Why this module exists:
- Detectors should NOT import from inspection-specific modules (like inspection/config.py).
- The inspection runner, CLI, and detectors can all depend on these lightweight types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal


InspectionMode = Literal["opencv", "dl"]


@dataclass(frozen=True)
class DebugConfig:
    """Debug visualization configuration."""
    show_debug: bool = False
    debug_pad: int = 20
    debug_tile_size: Tuple[int, int] = (420, 420)


@dataclass(frozen=True)
class ShapeFilterConfig:
    """Post-detection shape filter (applies to both OpenCV and DL masks)."""
    min_area: int = 50
    min_long_side: int = 40
    min_aspect: float = 2.0


@dataclass(frozen=True)
class OpenCVConfig:
    """OpenCV-based crack detection configuration."""
    kernel_sizes: Tuple[int, ...] = (7, 15, 25)
    percentile: float = 99.6
    close_ksize: Tuple[int, int] = (5, 5)
    history: int = 3
    use_clahe: bool = True


@dataclass(frozen=True)
class DLConfig:
    """U-Net inference configuration."""
    ckpt_path: str
    input_size: int = 512
    mask_thr: float = 0.5
    base_channels: int = 32
    use_amp: bool = True

    # Morphological cleanup for predicted mask (keep small)
    morph_open: Tuple[int, int] = (1, 1)
    morph_close: Tuple[int, int] = (3, 3)


@dataclass(frozen=True)
class DLRunConfig:
    """DL runtime params (history vote + quick pixel filter)."""
    model: DLConfig
    min_pixels: int = 20
    history: int = 2


@dataclass(frozen=True)
class RunConfig:
    """Top-level runtime configuration for a single inspection run."""
    video_path: str
    log_dir: str
    img_dir: str

    start_sec: int = 0
    end_sec: Optional[int] = None

    pause_on_detect: bool = False
    mode: InspectionMode = "dl"

    opencv: OpenCVConfig = field(default_factory=OpenCVConfig)
    dl: DLRunConfig = field(default_factory=lambda: DLRunConfig(model=DLConfig(ckpt_path="")))

    shape: ShapeFilterConfig = field(default_factory=ShapeFilterConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
