# fmvs_inspector/utils/time.py
"""Time formatting utilities.

Used for:
- on-frame overlay text
- log timestamps
- filename timestamps

All functions take *milliseconds* as input.
"""
from __future__ import annotations


def _clamp_ms(ms: int) -> int:
    return int(ms) if int(ms) > 0 else 0


def ms_to_hhmmssmmm(ms: int) -> str:
    """Milliseconds -> 'HH:MM:SS.mmm'."""
    ms = _clamp_ms(ms)
    total_sec = ms // 1000
    mmm = ms % 1000
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{mmm:03d}"


def ms_to_hhmmssmmm_compact(ms: int) -> str:
    """Milliseconds -> 'HHMMSSmmm' (no separators), for filenames."""
    ms = _clamp_ms(ms)
    total_sec = ms // 1000
    mmm = ms % 1000
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60
    return f"{hh:02d}{mm:02d}{ss:02d}{mmm:03d}"


# Backward-compatible name used in the inspection overlay.
# (Historically it included milliseconds as well.)
def ms_to_hhmmss(ms: int) -> str:
    return ms_to_hhmmssmmm(ms)
