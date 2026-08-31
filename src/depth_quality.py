"""Pre-filter quality gates for aligned relative-depth geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .depth_types import CameraDepth
from .robust_filtering import distribution_statistics


@dataclass(frozen=True)
class DepthQualityThresholds:
    """Configurable heuristics used to accept aligned relative depth."""

    min_valid_depth_ratio: float | None = 0.60
    max_denominator_reject_ratio: float | None = 0.30
    min_depth_alignment_inliers: int | None = 500
    min_depth_alignment_inlier_ratio: float | None = 0.30
    max_relative_z_p99_over_median: float | None = 50.0

    def __post_init__(self) -> None:
        for name in (
            "min_valid_depth_ratio",
            "max_denominator_reject_ratio",
            "min_depth_alignment_inlier_ratio",
        ):
            value = getattr(self, name)
            if value is not None and (
                not np.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be null or between 0 and 1")
        if (
            self.min_depth_alignment_inliers is not None
            and (
                isinstance(self.min_depth_alignment_inliers, bool)
                or not isinstance(
                    self.min_depth_alignment_inliers, (int, np.integer)
                )
                or self.min_depth_alignment_inliers < 0
            )
        ):
            raise ValueError("min_depth_alignment_inliers must be null or non-negative")
        dynamic = self.max_relative_z_p99_over_median
        if dynamic is not None and (not np.isfinite(dynamic) or dynamic <= 0.0):
            raise ValueError(
                "max_relative_z_p99_over_median must be null or positive"
            )


@dataclass(frozen=True)
class DepthAlignmentQualityMetrics:
    total_depth_candidates: int
    denominator_rejects: int
    valid_aligned_depth_count: int
    denominator_rejection_ratio: float
    valid_aligned_depth_ratio: float
    alignment_input_correspondences: int
    alignment_inliers: int
    alignment_inlier_ratio: float
    aligned_z_p1: float | None
    aligned_z_median: float | None
    aligned_z_p99: float | None
    relative_z_p99_over_median: float | None
    alignment_scale: float | None
    alignment_shift: float | None


@dataclass(frozen=True)
class DepthQualityAssessment:
    accepted: bool
    rejection_reason: str | None
    metrics: DepthAlignmentQualityMetrics


def measure_depth_alignment_quality(
    camera_depth: CameraDepth,
    *,
    alignment_input_correspondences: int,
    alignment_inliers: int,
) -> DepthAlignmentQualityMetrics:
    """Measure full-resolution aligned Z before percentile filtering."""
    if not isinstance(camera_depth, CameraDepth):
        raise TypeError("camera_depth must be a CameraDepth")
    if alignment_input_correspondences < 0 or alignment_inliers < 0:
        raise ValueError("alignment correspondence counts cannot be negative")
    if alignment_inliers > alignment_input_correspondences:
        raise ValueError("alignment inliers cannot exceed input correspondences")

    values = np.asarray(camera_depth.values, dtype=np.float64)
    total = int(values.size)
    valid_values = values[np.isfinite(values) & (values > 0.0)]
    valid_count = int(valid_values.size)
    denominator_rejects = (
        camera_depth.rejected_small_denominator_count
        + camera_depth.rejected_nonfinite_denominator_count
    )
    if denominator_rejects < 0 or denominator_rejects > total:
        raise ValueError("denominator rejection count is outside the depth array")

    if valid_count:
        statistics = distribution_statistics(valid_values)
        p1 = statistics["p1"]
        median = statistics["median"]
        p99 = statistics["p99"]
        dynamic_range = p99 / median
    else:
        p1 = median = p99 = dynamic_range = None

    alignment_ratio = (
        alignment_inliers / alignment_input_correspondences
        if alignment_input_correspondences
        else 0.0
    )
    return DepthAlignmentQualityMetrics(
        total_depth_candidates=total,
        denominator_rejects=denominator_rejects,
        valid_aligned_depth_count=valid_count,
        denominator_rejection_ratio=denominator_rejects / total,
        valid_aligned_depth_ratio=valid_count / total,
        alignment_input_correspondences=alignment_input_correspondences,
        alignment_inliers=alignment_inliers,
        alignment_inlier_ratio=alignment_ratio,
        aligned_z_p1=p1,
        aligned_z_median=median,
        aligned_z_p99=p99,
        relative_z_p99_over_median=dynamic_range,
        alignment_scale=camera_depth.disparity_scale,
        alignment_shift=camera_depth.disparity_shift,
    )


def assess_depth_alignment_quality(
    camera_depth: CameraDepth,
    *,
    alignment_input_correspondences: int,
    alignment_inliers: int,
    thresholds: DepthQualityThresholds,
) -> DepthQualityAssessment:
    """Apply deterministic gates in a stable, documented priority order."""
    metrics = measure_depth_alignment_quality(
        camera_depth,
        alignment_input_correspondences=alignment_input_correspondences,
        alignment_inliers=alignment_inliers,
    )
    checks = (
        (
            thresholds.max_denominator_reject_ratio is not None
            and metrics.denominator_rejection_ratio
            > thresholds.max_denominator_reject_ratio,
            "depth_denominator_reject_ratio",
        ),
        (
            thresholds.min_valid_depth_ratio is not None
            and metrics.valid_aligned_depth_ratio
            < thresholds.min_valid_depth_ratio,
            "depth_valid_ratio",
        ),
        (
            thresholds.min_depth_alignment_inliers is not None
            and metrics.alignment_inliers
            < thresholds.min_depth_alignment_inliers,
            "depth_alignment_inliers",
        ),
        (
            thresholds.min_depth_alignment_inlier_ratio is not None
            and metrics.alignment_inlier_ratio
            < thresholds.min_depth_alignment_inlier_ratio,
            "depth_alignment_inlier_ratio",
        ),
        (
            thresholds.max_relative_z_p99_over_median is not None
            and (
                metrics.relative_z_p99_over_median is None
                or metrics.relative_z_p99_over_median
                > thresholds.max_relative_z_p99_over_median
            ),
            "depth_z_distribution",
        ),
    )
    for failed, reason in checks:
        if failed:
            return DepthQualityAssessment(False, reason, metrics)
    return DepthQualityAssessment(True, None, metrics)
