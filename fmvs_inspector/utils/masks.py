# fmvs_inspector/utils/masks.py
"""Mask utilities for polygon-to-mask conversion and contour extraction."""
from __future__ import annotations

from typing import Tuple, List

import cv2
import numpy as np


def polygon_to_mask(shape_hw: Tuple[int, int], poly: np.ndarray) -> np.ndarray:
    """Convert polygon to binary mask.
    
    Args:
        shape_hw: (height, width) tuple for output mask shape
        poly: Numpy array of polygon points (N, 2)
        
    Returns:
        Binary mask (uint8) with 255 inside polygon, 0 outside
    """
    m = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    return m


def contours_from_mask(mask_u8: np.ndarray) -> List[np.ndarray]:
    """Extract contours from binary mask.
    
    Args:
        mask_u8: Binary mask (uint8)
        
    Returns:
        List of contours (each is a numpy array)
    """
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
