"""Depth visualization helpers using NumPy and OpenCV."""

import cv2
import numpy as np


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize one depth map to uint8, ignoring non-finite values."""
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        raise ValueError("depth must be a two-dimensional NumPy array")
    finite = np.isfinite(depth)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if not finite.any():
        return normalized
    minimum = float(depth[finite].min())
    maximum = float(depth[finite].max())
    if maximum > minimum:
        scaled = (depth[finite] - minimum) / (maximum - minimum)
        normalized[finite] = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    return normalized


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Colorize relative depth as a BGR image."""
    return cv2.applyColorMap(normalize_depth(depth), cv2.COLORMAP_INFERNO)


def make_side_by_side(rgb_bgr: np.ndarray, depth_bgr: np.ndarray) -> np.ndarray:
    """Place a BGR source frame and depth visualization side-by-side."""
    if rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
        raise ValueError("rgb_bgr must be an HxWx3 image")
    if depth_bgr.ndim != 3 or depth_bgr.shape[2] != 3:
        raise ValueError("depth_bgr must be an HxWx3 image")
    if depth_bgr.shape[:2] != rgb_bgr.shape[:2]:
        depth_bgr = cv2.resize(
            depth_bgr,
            (rgb_bgr.shape[1], rgb_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.hstack((rgb_bgr, depth_bgr))
