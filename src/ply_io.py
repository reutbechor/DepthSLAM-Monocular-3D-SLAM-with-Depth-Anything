"""Minimal ASCII PLY export for colored relative point clouds."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ascii_ply(
    path: str | Path, points: np.ndarray, colors_rgb: np.ndarray
) -> Path:
    """Write Nx3 points and uint8 RGB colors to an ASCII PLY file."""
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
    color_array = color_array.astype(np.uint8, copy=False)

    output_path = Path(path)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        "comment coordinate_units relative_depth_units\n"
        "comment point_cloud_is_metric false\n"
        f"element vertex {point_array.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with output_path.open("w", encoding="ascii", newline="\n") as file:
        file.write(header)
        if point_array.shape[0]:
            vertices = np.column_stack((point_array, color_array))
            np.savetxt(
                file,
                vertices,
                fmt=("%.9g", "%.9g", "%.9g", "%d", "%d", "%d"),
            )
    return output_path
