"""Pinhole backprojection from pixels and relative depth to camera-frame 3D."""

from __future__ import annotations

import numpy as np


def validate_camera_matrix(camera_matrix: np.ndarray) -> np.ndarray:
    """Return a validated float64 zero-skew pinhole camera matrix."""
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera intrinsics K must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("camera intrinsics must contain positive fx and fy")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0]):
        raise ValueError("camera intrinsics must have final row [0, 0, 1]")
    if not np.isclose(matrix[0, 1], 0.0) or not np.isclose(matrix[1, 0], 0.0):
        raise ValueError("backprojection currently requires zero-skew intrinsics")
    return matrix


def backproject_pixels(
    pixels: np.ndarray, depths: np.ndarray, camera_matrix: np.ndarray
) -> np.ndarray:
    """Backproject Nx2 pixels into Nx3 points in relative depth units.

    The resulting coordinates are in the input depth map's relative units, not
    metres. Invalid depths are rejected rather than replaced.
    """
    pixel_array = np.asarray(pixels, dtype=np.float64)
    if pixel_array.ndim != 2 or pixel_array.shape[1:] != (2,):
        raise ValueError("pixels must be an Nx2 array")
    if not np.isfinite(pixel_array).all():
        raise ValueError("pixels must contain only finite values")

    depth_array = np.asarray(depths, dtype=np.float64)
    if depth_array.ndim == 2 and depth_array.shape[1:] == (1,):
        depth_array = depth_array[:, 0]
    if depth_array.ndim != 1:
        raise ValueError("depths must be an N or Nx1 array")
    if depth_array.shape[0] != pixel_array.shape[0]:
        raise ValueError("pixels and depths must contain the same number of samples")
    if not np.isfinite(depth_array).all() or np.any(depth_array <= 0):
        raise ValueError("depths must contain only finite, positive values")

    intrinsics = validate_camera_matrix(camera_matrix)
    if pixel_array.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = depth_array
    x = (pixel_array[:, 0] - cx) * z / fx
    y = (pixel_array[:, 1] - cy) * z / fy
    return np.column_stack((x, y, z)).astype(np.float64, copy=False)
