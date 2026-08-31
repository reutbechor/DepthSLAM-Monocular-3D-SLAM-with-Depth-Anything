"""Lightweight deterministic fusion of colored relative point clouds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_cloud(points: np.ndarray, colors_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1:] != (3,):
        raise ValueError("points must be an Nx3 array")
    if not np.isfinite(point_array).all():
        raise ValueError("points must contain only finite values")
    color_array = np.asarray(colors_rgb)
    if color_array.ndim != 2 or color_array.shape[1:] != (3,):
        raise ValueError("colors_rgb must be an Nx3 array")
    if color_array.shape[0] != point_array.shape[0]:
        raise ValueError("points and colors_rgb must have equal length")
    if not np.issubdtype(color_array.dtype, np.integer):
        raise ValueError("colors_rgb must contain integer values")
    if color_array.size and (color_array.min() < 0 or color_array.max() > 255):
        raise ValueError("colors_rgb values must be in the range 0..255")
    return point_array, color_array.astype(np.uint8, copy=False)


@dataclass(frozen=True)
class FusedPointCloud:
    points: np.ndarray
    colors: np.ndarray
    input_point_count: int
    output_point_count: int
    voxel_size: float
    coordinate_units: str = "relative_map_units"


def voxel_downsample(
    points: np.ndarray, colors_rgb: np.ndarray, voxel_size: float
) -> FusedPointCloud:
    """Average coordinates and RGB colors in deterministic relative-unit voxels."""
    point_array, color_array = _validate_cloud(points, colors_rgb)
    size = float(voxel_size)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("voxel_size must be finite and positive")
    if point_array.shape[0] == 0:
        return FusedPointCloud(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8),
            0,
            0,
            size,
        )

    voxel_indices = np.floor(point_array / size).astype(np.int64)
    _, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    voxel_count = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=voxel_count).astype(np.float64)
    point_sums = np.zeros((voxel_count, 3), dtype=np.float64)
    color_sums = np.zeros((voxel_count, 3), dtype=np.float64)
    np.add.at(point_sums, inverse, point_array)
    np.add.at(color_sums, inverse, color_array)
    averaged_points = point_sums / counts[:, None]
    averaged_colors = np.clip(
        np.rint(color_sums / counts[:, None]), 0, 255
    ).astype(np.uint8)
    return FusedPointCloud(
        points=averaged_points,
        colors=averaged_colors,
        input_point_count=point_array.shape[0],
        output_point_count=averaged_points.shape[0],
        voxel_size=size,
    )


class RelativeMapFusion:
    """Collect relative world-frame clouds and downsample them once at the end."""

    def __init__(self, voxel_size: float) -> None:
        size = float(voxel_size)
        if not np.isfinite(size) or size <= 0.0:
            raise ValueError("voxel_size must be finite and positive")
        self.voxel_size = size
        self._points: list[np.ndarray] = []
        self._colors: list[np.ndarray] = []

    def add(self, points: np.ndarray, colors_rgb: np.ndarray) -> None:
        point_array, color_array = _validate_cloud(points, colors_rgb)
        if point_array.shape[0]:
            self._points.append(point_array.copy())
            self._colors.append(color_array.copy())

    @property
    def raw_point_count(self) -> int:
        return sum(points.shape[0] for points in self._points)

    def finalize(self) -> FusedPointCloud:
        if not self._points:
            return voxel_downsample(
                np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), self.voxel_size
            )
        return voxel_downsample(
            np.concatenate(self._points, axis=0),
            np.concatenate(self._colors, axis=0),
            self.voxel_size,
        )
