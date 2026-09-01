"""Experimental pairwise normalization of aligned relative camera-Z maps."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .backprojection import backproject_pixels, validate_camera_matrix
from .depth_geometry import DepthGeometryProcessor
from .depth_types import CameraDepth
from .transforms import transform_points


@dataclass(frozen=True)
class TemporalDepthNormalizationConfig:
    enabled: bool = False
    minimum_correspondences: int = 200
    minimum_inliers: int = 150
    minimum_inlier_ratio: float = 0.50
    minimum_scale: float = 0.70
    maximum_scale: float = 1.30
    maximum_log_mad: float = 0.25
    minimum_cumulative_scale: float = 0.50
    maximum_cumulative_scale: float = 2.00
    log_mad_outlier_multiplier: float = 3.5

    def __post_init__(self) -> None:
        if self.minimum_correspondences < 1 or self.minimum_inliers < 1:
            raise ValueError("correspondence and inlier minima must be positive")
        if not 0.0 <= self.minimum_inlier_ratio <= 1.0:
            raise ValueError("minimum_inlier_ratio must be in the range 0..1")
        positive = (
            "minimum_scale", "maximum_scale", "maximum_log_mad",
            "minimum_cumulative_scale", "maximum_cumulative_scale",
            "log_mad_outlier_multiplier",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_scale > self.maximum_scale:
            raise ValueError("minimum_scale cannot exceed maximum_scale")
        if self.minimum_cumulative_scale > self.maximum_cumulative_scale:
            raise ValueError(
                "minimum_cumulative_scale cannot exceed maximum_cumulative_scale"
            )


@dataclass(frozen=True)
class TemporalScaleEstimate:
    attempted: bool
    accepted: bool
    reason: str
    raw_correspondence_count: int
    valid_ratio_count: int
    inlier_count: int
    inlier_ratio: float
    raw_ratio_median: float | None
    temporal_scale_pairwise: float | None
    temporal_scale_cumulative: float
    temporal_scale_cumulative_candidate: float | None
    log_ratio_median: float | None
    log_ratio_mad: float | None
    ratio_p05: float | None
    ratio_p50: float | None
    ratio_p95: float | None
    valid_input_mask: np.ndarray
    retained_input_mask: np.ndarray


@dataclass(frozen=True)
class TemporalDepthNormalizationDiagnostics:
    temporal_depth_normalization_attempted: bool
    temporal_depth_normalization_accepted: bool
    temporal_depth_normalization_reason: str
    temporal_depth_correspondence_count: int
    temporal_depth_valid_ratio_count: int
    temporal_depth_inlier_count: int
    temporal_depth_inlier_ratio: float
    temporal_depth_raw_ratio_median: float | None
    temporal_depth_ratio_p05: float | None
    temporal_depth_ratio_p50: float | None
    temporal_depth_ratio_p95: float | None
    temporal_depth_scale_pairwise: float | None
    temporal_depth_scale_cumulative: float
    temporal_depth_scale_cumulative_candidate: float | None
    temporal_depth_log_ratio_median: float | None
    temporal_depth_log_ratio_mad: float | None
    original_z_median: float | None
    normalized_z_median: float | None
    original_z_p95: float | None
    normalized_z_p95: float | None
    original_z_p99: float | None
    normalized_z_p99: float | None
    temporal_original_alignment_a: float | None
    temporal_original_alignment_b: float | None
    temporal_residual_correspondence_count: int = 0
    temporal_residual_before_median: float | None = None
    temporal_residual_before_rmse: float | None = None
    temporal_residual_after_median: float | None = None
    temporal_residual_after_rmse: float | None = None


@dataclass(frozen=True)
class TemporalDepthNormalizationResult:
    normalized_depth: CameraDepth
    diagnostics: TemporalDepthNormalizationDiagnostics
    retained_match_mask: np.ndarray


def reference_temporal_depth_result(
    camera_depth: CameraDepth,
) -> TemporalDepthNormalizationResult:
    """Return identity diagnostics for the fixed first accepted keyframe."""

    median, p95, p99 = _depth_statistics(camera_depth.values)
    diagnostics = TemporalDepthNormalizationDiagnostics(
        temporal_depth_normalization_attempted=False,
        temporal_depth_normalization_accepted=False,
        temporal_depth_normalization_reason="temporal_depth_reference_identity",
        temporal_depth_correspondence_count=0,
        temporal_depth_valid_ratio_count=0,
        temporal_depth_inlier_count=0,
        temporal_depth_inlier_ratio=0.0,
        temporal_depth_raw_ratio_median=1.0,
        temporal_depth_ratio_p05=1.0,
        temporal_depth_ratio_p50=1.0,
        temporal_depth_ratio_p95=1.0,
        temporal_depth_scale_pairwise=1.0,
        temporal_depth_scale_cumulative=1.0,
        temporal_depth_scale_cumulative_candidate=1.0,
        temporal_depth_log_ratio_median=0.0,
        temporal_depth_log_ratio_mad=0.0,
        original_z_median=median,
        normalized_z_median=median,
        original_z_p95=p95,
        normalized_z_p95=p95,
        original_z_p99=p99,
        normalized_z_p99=p99,
        temporal_original_alignment_a=camera_depth.disparity_scale,
        temporal_original_alignment_b=camera_depth.disparity_shift,
    )
    return TemporalDepthNormalizationResult(
        camera_depth, diagnostics, np.empty(0, dtype=bool)
    )


def _depth_statistics(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    array = np.asarray(values, dtype=np.float64)
    valid = array[np.isfinite(array) & (array > 0.0)]
    if valid.size == 0:
        return None, None, None
    median, p95, p99 = np.percentile(valid, (50.0, 95.0, 99.0))
    return float(median), float(p95), float(p99)


def estimate_temporal_scale(
    previous_depth_samples: np.ndarray,
    current_depth_samples: np.ndarray,
    config: TemporalDepthNormalizationConfig,
    *,
    raw_correspondence_count: int | None = None,
    previous_cumulative_scale: float = 1.0,
) -> TemporalScaleEstimate:
    """Estimate a robust symmetric scale from log(Z_prev)-log(Z_curr)."""

    previous = np.asarray(previous_depth_samples, dtype=np.float64).reshape(-1)
    current = np.asarray(current_depth_samples, dtype=np.float64).reshape(-1)
    if previous.shape != current.shape:
        raise ValueError("previous and current samples must have equal length")
    count = previous.size if raw_correspondence_count is None else int(raw_correspondence_count)
    if count < previous.size or count < 0:
        raise ValueError("raw_correspondence_count cannot be smaller than sample count")
    if not np.isfinite(previous_cumulative_scale) or previous_cumulative_scale <= 0:
        raise ValueError("previous_cumulative_scale must be finite and positive")
    empty_mask = np.zeros(previous.shape, dtype=bool)
    if not config.enabled:
        return TemporalScaleEstimate(
            False, False, "temporal_depth_normalization_disabled", count,
            0, 0, 0.0, None, None, float(previous_cumulative_scale), None,
            None, None, None, None, None, empty_mask, empty_mask.copy(),
        )

    valid = (
        np.isfinite(previous) & (previous > 0.0)
        & np.isfinite(current) & (current > 0.0)
    )
    valid_indices = np.flatnonzero(valid)
    log_ratios = np.log(previous[valid]) - np.log(current[valid])
    ratios = np.exp(log_ratios)
    valid_count = int(ratios.size)
    if valid_count:
        ratio_p05, ratio_p50, ratio_p95 = (
            float(value) for value in np.percentile(ratios, (5.0, 50.0, 95.0))
        )
        log_center = float(np.median(log_ratios))
        log_mad = float(np.median(np.abs(log_ratios - log_center)))
        threshold = max(
            config.log_mad_outlier_multiplier * 1.4826 * log_mad,
            1e-9,
        )
        retained_local = np.abs(log_ratios - log_center) <= threshold
        retained_logs = log_ratios[retained_local]
    else:
        ratio_p05 = ratio_p50 = ratio_p95 = None
        log_center = log_mad = None
        retained_local = np.zeros(0, dtype=bool)
        retained_logs = np.empty(0, dtype=np.float64)
    retained = np.zeros(previous.shape, dtype=bool)
    if valid_indices.size:
        retained[valid_indices[retained_local]] = True
    inlier_count = int(retained_logs.size)
    inlier_ratio = float(inlier_count / valid_count) if valid_count else 0.0
    scale = float(np.exp(np.median(retained_logs))) if inlier_count else None
    candidate = (
        None if scale is None else float(previous_cumulative_scale * scale)
    )
    accepted = False
    reason = "temporal_depth_normalization_accepted"
    if count < config.minimum_correspondences:
        reason = "temporal_depth_normalization_insufficient_correspondences"
    elif valid_count < config.minimum_correspondences:
        reason = "temporal_depth_normalization_insufficient_valid_ratios"
    elif inlier_count < config.minimum_inliers:
        reason = "temporal_depth_normalization_insufficient_inliers"
    elif inlier_ratio < config.minimum_inlier_ratio:
        reason = "temporal_depth_normalization_low_inlier_ratio"
    elif scale is None or not np.isfinite(scale) or scale <= 0.0:
        reason = "temporal_depth_normalization_invalid_scale"
    elif not config.minimum_scale <= scale <= config.maximum_scale:
        reason = "temporal_depth_normalization_scale_out_of_range"
    elif log_mad is None or log_mad > config.maximum_log_mad:
        reason = "temporal_depth_normalization_excessive_log_mad"
    elif candidate is None or not (
        config.minimum_cumulative_scale
        <= candidate
        <= config.maximum_cumulative_scale
    ):
        reason = "temporal_depth_normalization_cumulative_scale_guard"
    else:
        accepted = True
    cumulative = candidate if accepted and candidate is not None else previous_cumulative_scale
    return TemporalScaleEstimate(
        True, accepted, reason, count, valid_count, inlier_count, inlier_ratio,
        ratio_p50, scale, float(cumulative), candidate, log_center, log_mad,
        ratio_p05, ratio_p50, ratio_p95, valid, retained,
    )


def normalize_temporal_depth(
    points_previous: np.ndarray,
    points_current: np.ndarray,
    correspondence_mask: np.ndarray,
    previous_depth: CameraDepth,
    current_depth: CameraDepth,
    config: TemporalDepthNormalizationConfig,
    *,
    previous_cumulative_scale: float = 1.0,
) -> TemporalDepthNormalizationResult:
    """Sample geometric correspondences and optionally scale a copied current Z."""

    previous_points = np.asarray(points_previous, dtype=np.float64)
    current_points = np.asarray(points_current, dtype=np.float64)
    mask = np.asarray(correspondence_mask, dtype=bool).reshape(-1)
    if previous_points.shape != current_points.shape or previous_points.ndim != 2 or previous_points.shape[1:] != (2,):
        raise ValueError("matched point arrays must be equal Nx2 arrays")
    if mask.shape[0] != previous_points.shape[0]:
        raise ValueError("correspondence_mask length must match point arrays")
    selected = np.flatnonzero(mask)
    previous_samples, previous_valid = DepthGeometryProcessor._sample_bilinear(
        np.asarray(previous_depth.values, dtype=np.float64), previous_points[selected]
    )
    current_samples, current_valid = DepthGeometryProcessor._sample_bilinear(
        np.asarray(current_depth.values, dtype=np.float64), current_points[selected]
    )
    usable_previous = np.where(previous_valid, previous_samples, np.nan)
    usable_current = np.where(current_valid, current_samples, np.nan)
    estimate = estimate_temporal_scale(
        usable_previous,
        usable_current,
        config,
        raw_correspondence_count=int(selected.size),
        previous_cumulative_scale=previous_cumulative_scale,
    )
    retained_full = np.zeros(mask.shape, dtype=bool)
    if selected.size:
        retained_full[selected[estimate.retained_input_mask]] = True
    if estimate.accepted:
        normalized_values = np.asarray(current_depth.values).copy()
        finite_positive = np.isfinite(normalized_values) & (normalized_values > 0.0)
        normalized_values[finite_positive] *= float(estimate.temporal_scale_pairwise)
        normalized_depth = replace(current_depth, values=normalized_values)
    else:
        normalized_depth = current_depth
    original_stats = _depth_statistics(current_depth.values)
    normalized_stats = _depth_statistics(normalized_depth.values)
    diagnostics = TemporalDepthNormalizationDiagnostics(
        temporal_depth_normalization_attempted=estimate.attempted,
        temporal_depth_normalization_accepted=estimate.accepted,
        temporal_depth_normalization_reason=estimate.reason,
        temporal_depth_correspondence_count=estimate.raw_correspondence_count,
        temporal_depth_valid_ratio_count=estimate.valid_ratio_count,
        temporal_depth_inlier_count=estimate.inlier_count,
        temporal_depth_inlier_ratio=estimate.inlier_ratio,
        temporal_depth_raw_ratio_median=estimate.raw_ratio_median,
        temporal_depth_ratio_p05=estimate.ratio_p05,
        temporal_depth_ratio_p50=estimate.ratio_p50,
        temporal_depth_ratio_p95=estimate.ratio_p95,
        temporal_depth_scale_pairwise=estimate.temporal_scale_pairwise,
        temporal_depth_scale_cumulative=estimate.temporal_scale_cumulative,
        temporal_depth_scale_cumulative_candidate=(
            estimate.temporal_scale_cumulative_candidate
        ),
        temporal_depth_log_ratio_median=estimate.log_ratio_median,
        temporal_depth_log_ratio_mad=estimate.log_ratio_mad,
        original_z_median=original_stats[0],
        normalized_z_median=normalized_stats[0],
        original_z_p95=original_stats[1],
        normalized_z_p95=normalized_stats[1],
        original_z_p99=original_stats[2],
        normalized_z_p99=normalized_stats[2],
        temporal_original_alignment_a=current_depth.disparity_scale,
        temporal_original_alignment_b=current_depth.disparity_shift,
    )
    return TemporalDepthNormalizationResult(
        normalized_depth, diagnostics, retained_full
    )


def matched_world_residual_statistics(
    points_previous: np.ndarray,
    points_current: np.ndarray,
    correspondence_mask: np.ndarray,
    previous_depth: CameraDepth,
    current_depth: CameraDepth,
    camera_matrix: np.ndarray,
    previous_world_from_camera: np.ndarray,
    current_world_from_camera: np.ndarray,
) -> dict[str, float | int | None]:
    """Compare matched 3D samples in world coordinates without changing poses."""

    previous_points = np.asarray(points_previous, dtype=np.float64)
    current_points = np.asarray(points_current, dtype=np.float64)
    mask = np.asarray(correspondence_mask, dtype=bool).reshape(-1)
    indices = np.flatnonzero(mask)
    previous_samples, previous_valid = DepthGeometryProcessor._sample_bilinear(
        np.asarray(previous_depth.values, dtype=np.float64), previous_points[indices]
    )
    current_samples, current_valid = DepthGeometryProcessor._sample_bilinear(
        np.asarray(current_depth.values, dtype=np.float64), current_points[indices]
    )
    valid = previous_valid & current_valid
    if not valid.any():
        return {"count": 0, "median": None, "rmse": None}
    intrinsics = validate_camera_matrix(camera_matrix)
    previous_camera = backproject_pixels(
        previous_points[indices][valid], previous_samples[valid], intrinsics
    )
    current_camera = backproject_pixels(
        current_points[indices][valid], current_samples[valid], intrinsics
    )
    previous_pose = np.asarray(previous_world_from_camera, dtype=np.float64)
    current_pose = np.asarray(current_world_from_camera, dtype=np.float64)
    previous_world = transform_points(
        previous_camera, previous_pose[:3, :3], previous_pose[:3, 3]
    )
    current_world = transform_points(
        current_camera, current_pose[:3, :3], current_pose[:3, 3]
    )
    residuals = np.linalg.norm(previous_world - current_world, axis=1)
    return {
        "count": int(residuals.size),
        "median": float(np.median(residuals)),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
    }
