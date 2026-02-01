# fmvs_inspector/utils/paths.py
"""Path manipulation utilities."""
from pathlib import Path


def safe_video_stem(video_path: str) -> str:
    """Extract safe filename stem from video path.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Filename stem or "video" if extraction fails
        
    Examples:
        >>> safe_video_stem("/path/to/video.mp4")
        'video'
        >>> safe_video_stem("myfile.avi")
        'myfile'
    """
    return Path(video_path).stem or "video"
