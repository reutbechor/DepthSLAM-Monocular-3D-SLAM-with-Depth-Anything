"""Dense colored point-cloud generation from RGB and explicit camera Z."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backprojection import backproject_pixels, validate_camera_matrix
from .depth_types import CameraDepth


@dataclass(frozen=True)
class PointCloudResult:
    """A single-camera colored point cloud in relative depth units."""

    points: np.ndarray
    colors: np.ndarray
    valid_pixel_coordinates: np.ndarray
    sampled_pixel_count: int
    valid_point_count: int
    stride: int
    coordinate_frame: str = "camera"
    coordinate_units: str = "relative_depth_units"
    depth_type: str = "relative"
    depth_representation: str = "relative_camera_z_proxy"


def generate_colored_point_cloud(
    image_rgb: np.ndarray,
    camera_depth: CameraDepth,
    camera_matrix: np.ndarray,
    stride: int = 1,
) -> PointCloudResult:
    """Backproject typed camera Z while preserving uint8 RGB colors."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image_rgb must be a non-empty HxWx3 image")
    if image.dtype != np.uint8:
        raise ValueError("image_rgb must use uint8 RGB values")

    if not isinstance(camera_depth, CameraDepth):
        raise TypeError(
            "camera_depth must be a CameraDepth; raw model predictions are not Z"
        )
    depth = np.asarray(camera_depth.values)
    if depth.ndim != 2 or depth.size == 0 or not np.issubdtype(depth.dtype, np.number):
        raise ValueError("camera_depth values must be a non-empty numeric HxW array")
    if depth.shape != image.shape[:2]:
        raise ValueError("image_rgb and camera_depth must have matching dimensions")
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer")
    intrinsics = validate_camera_matrix(camera_matrix)

    height, width = depth.shape
    sampled_y = np.arange(0, height, int(stride), dtype=np.int64)
    sampled_x = np.arange(0, width, int(stride), dtype=np.int64)
    grid_x, grid_y = np.meshgrid(sampled_x, sampled_y)
    pixels = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    sampled_depths = depth[grid_y, grid_x].reshape(-1).astype(np.float64, copy=False)
    sampled_colors = image[grid_y, grid_x].reshape(-1, 3)

    valid = np.isfinite(sampled_depths) & (sampled_depths > 0.0)
    valid_pixels = pixels[valid].astype(np.float64, copy=False)
    valid_depths = sampled_depths[valid]
    points = backproject_pixels(valid_pixels, valid_depths, intrinsics)
    colors = sampled_colors[valid].astype(np.uint8, copy=False)
    return PointCloudResult(
        points=points,
        colors=colors,
        valid_pixel_coordinates=valid_pixels,
        sampled_pixel_count=pixels.shape[0],
        valid_point_count=points.shape[0],
        stride=int(stride),
        coordinate_units=camera_depth.coordinate_units,
        depth_type=camera_depth.depth_type,
        depth_representation=camera_depth.representation,
    )
