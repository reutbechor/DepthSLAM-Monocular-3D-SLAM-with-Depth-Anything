"""Sample relative depth at pose inliers and create relative 3D geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backprojection import backproject_pixels, validate_camera_matrix


@dataclass(frozen=True)
class DepthGeometryResult:
    """Filtered two-view features and their Frame 1 relative 3D points."""

    original_match_count: int
    pose_inlier_count: int
    valid_depth_sample_count: int
    valid_pixel_coordinates: np.ndarray
    sampled_relative_depths: np.ndarray
    points_3d_relative: np.ndarray
    valid_match_indices: np.ndarray
    valid_match_mask: np.ndarray
    sampling_method: str


class DepthGeometryProcessor:
    """Convert Frame 1 pose inliers into relative camera-frame 3D points."""

    def __init__(self, sampling_method: str = "bilinear") -> None:
        if sampling_method not in {"bilinear", "nearest"}:
            raise ValueError("sampling_method must be 'bilinear' or 'nearest'")
        self.sampling_method = sampling_method

    @staticmethod
    def _validate_inputs(
        matched_points: np.ndarray,
        pose_inlier_mask: np.ndarray,
        relative_depth_map: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(matched_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (2,):
            raise ValueError("matched_points must be an Nx2 array")
        mask = np.asarray(pose_inlier_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != points.shape[0]:
            raise ValueError("pose_inlier_mask length must equal matched point count")
        depth = np.asarray(relative_depth_map)
        if depth.ndim != 2 or depth.size == 0 or not np.issubdtype(depth.dtype, np.number):
            raise ValueError("relative_depth_map must be a non-empty numeric HxW array")
        return points, mask, depth.astype(np.float64, copy=False)

    @staticmethod
    def _sample_nearest(
        depth: np.ndarray, pixels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = depth.shape
        inside = (
            np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] <= width - 1)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] <= height - 1)
        )
        samples = np.full(pixels.shape[0], np.nan, dtype=np.float64)
        if inside.any():
            x = np.floor(pixels[inside, 0] + 0.5).astype(int)
            y = np.floor(pixels[inside, 1] + 0.5).astype(int)
            samples[inside] = depth[y, x]
        return samples, inside & np.isfinite(samples) & (samples > 0.0)

    @staticmethod
    def _sample_bilinear(
        depth: np.ndarray, pixels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = depth.shape
        inside = (
            np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] <= width - 1)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] <= height - 1)
        )
        samples = np.full(pixels.shape[0], np.nan, dtype=np.float64)
        if not inside.any():
            return samples, inside

        selected = pixels[inside]
        x0 = np.floor(selected[:, 0]).astype(int)
        y0 = np.floor(selected[:, 1]).astype(int)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        dx = selected[:, 0] - x0
        dy = selected[:, 1] - y0
        neighbors = np.column_stack(
            (depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1])
        )
        weights = np.column_stack(
            ((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy)
        )
        neighbor_valid = np.isfinite(neighbors) & (neighbors > 0.0)
        required_valid = np.all((weights == 0.0) | neighbor_valid, axis=1)
        safe_neighbors = np.where(neighbor_valid, neighbors, 0.0)
        interpolated = np.sum(weights * safe_neighbors, axis=1)
        selected_samples = np.where(required_valid, interpolated, np.nan)
        samples[inside] = selected_samples
        valid = inside & np.isfinite(samples) & (samples > 0.0)
        return samples, valid

    def process(
        self,
        matched_points: np.ndarray,
        pose_inlier_mask: np.ndarray,
        relative_depth_map: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> DepthGeometryResult:
        """Filter, sample, and backproject Frame 1 matches in relative units."""
        points, pose_mask, depth = self._validate_inputs(
            matched_points, pose_inlier_mask, relative_depth_map
        )
        intrinsics = validate_camera_matrix(camera_matrix)
        pose_indices = np.flatnonzero(pose_mask)
        pose_points = points[pose_indices]
        sampler = (
            self._sample_bilinear
            if self.sampling_method == "bilinear"
            else self._sample_nearest
        )
        sampled, valid_sample_mask = sampler(depth, pose_points)
        valid_indices = pose_indices[valid_sample_mask]
        valid_pixels = pose_points[valid_sample_mask]
        valid_depths = sampled[valid_sample_mask]
        points_3d = backproject_pixels(valid_pixels, valid_depths, intrinsics)
        original_mask = np.zeros(points.shape[0], dtype=bool)
        original_mask[valid_indices] = True
        return DepthGeometryResult(
            original_match_count=points.shape[0],
            pose_inlier_count=int(np.count_nonzero(pose_mask)),
            valid_depth_sample_count=valid_depths.shape[0],
            valid_pixel_coordinates=valid_pixels,
            sampled_relative_depths=valid_depths,
            points_3d_relative=points_3d,
            valid_match_indices=valid_indices,
            valid_match_mask=original_mask,
            sampling_method=self.sampling_method,
        )
