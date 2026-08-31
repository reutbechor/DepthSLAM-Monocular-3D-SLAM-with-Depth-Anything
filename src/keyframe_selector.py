"""Deterministic visual-motion keyframe selection for Stage 6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KeyframeThresholds:
    enabled: bool = True
    min_good_matches: int = 100
    min_geometric_inliers: int = 80
    min_geometric_inlier_ratio: float = 0.40
    min_median_feature_displacement_px: float = 8.0
    min_rotation_deg: float = 1.0
    max_frames_without_keyframe: int = 30

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("keyframe enabled must be boolean")
        for name in (
            "min_good_matches",
            "min_geometric_inliers",
            "max_frames_without_keyframe",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        ratio = self.min_geometric_inlier_ratio
        if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("min_geometric_inlier_ratio must be between 0 and 1")
        for name in (
            "min_median_feature_displacement_px",
            "min_rotation_deg",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class KeyframeMotionMetrics:
    good_matches: int
    geometric_inliers: int
    geometric_inlier_ratio: float
    median_feature_displacement_px: float
    p75_feature_displacement_px: float
    p90_feature_displacement_px: float
    rotation_deg: float
    frames_since_last_keyframe: int


@dataclass(frozen=True)
class KeyframeSelectionResult:
    status: str
    reason: str
    selected: bool
    metrics: KeyframeMotionMetrics | None


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    """Return the SO(3) trace angle with a numerically clamped acos input."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    cosine = float((np.trace(matrix) - 1.0) / 2.0)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def feature_displacement_statistics(
    points1: np.ndarray,
    points2: np.ndarray,
    inlier_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Return median/p75/p90 pixel displacement for geometric inliers."""
    first = np.asarray(points1, dtype=np.float64)
    second = np.asarray(points2, dtype=np.float64)
    mask = np.asarray(inlier_mask).reshape(-1).astype(bool)
    if first.ndim != 2 or first.shape[1:] != (2,):
        raise ValueError("points1 must be an Nx2 array")
    if second.shape != first.shape or mask.shape[0] != first.shape[0]:
        raise ValueError("points2 and inlier_mask must match points1")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("feature points must be finite")
    if not mask.any():
        raise ValueError("at least one geometric inlier is required")
    displacement = np.linalg.norm(second[mask] - first[mask], axis=1)
    median, p75, p90 = np.percentile(displacement, (50.0, 75.0, 90.0))
    return float(median), float(p75), float(p90)


class KeyframeSelector:
    """Select visually informative candidates without using metric distance."""

    def __init__(self, thresholds: KeyframeThresholds | None = None) -> None:
        self.thresholds = thresholds or KeyframeThresholds()

    @staticmethod
    def initial_frame() -> KeyframeSelectionResult:
        return KeyframeSelectionResult(
            status="selected",
            reason="initial_frame",
            selected=True,
            metrics=None,
        )

    def evaluate(
        self,
        *,
        points1: np.ndarray,
        points2: np.ndarray,
        inlier_mask: np.ndarray,
        rotation: np.ndarray,
        good_matches: int,
        geometric_inliers: int,
        geometric_inlier_ratio: float,
        frames_since_last_keyframe: int,
    ) -> KeyframeSelectionResult:
        if good_matches < 0 or geometric_inliers < 0:
            raise ValueError("match and inlier counts cannot be negative")
        if not np.isfinite(geometric_inlier_ratio) or not (
            0.0 <= geometric_inlier_ratio <= 1.0
        ):
            raise ValueError("geometric_inlier_ratio must be between 0 and 1")
        if frames_since_last_keyframe < 1:
            raise ValueError("frames_since_last_keyframe must be positive")

        median, p75, p90 = feature_displacement_statistics(
            points1, points2, inlier_mask
        )
        metrics = KeyframeMotionMetrics(
            good_matches=int(good_matches),
            geometric_inliers=int(geometric_inliers),
            geometric_inlier_ratio=float(geometric_inlier_ratio),
            median_feature_displacement_px=median,
            p75_feature_displacement_px=p75,
            p90_feature_displacement_px=p90,
            rotation_deg=rotation_angle_degrees(rotation),
            frames_since_last_keyframe=int(frames_since_last_keyframe),
        )
        if not self.thresholds.enabled:
            return KeyframeSelectionResult(
                "selected", "keyframe_selection_disabled", True, metrics
            )

        geometry_checks = (
            (
                metrics.good_matches < self.thresholds.min_good_matches,
                "keyframe_good_matches",
            ),
            (
                metrics.geometric_inliers < self.thresholds.min_geometric_inliers,
                "keyframe_geometric_inliers",
            ),
            (
                metrics.geometric_inlier_ratio
                < self.thresholds.min_geometric_inlier_ratio,
                "keyframe_geometric_inlier_ratio",
            ),
        )
        for failed, reason in geometry_checks:
            if failed:
                return KeyframeSelectionResult("rejected", reason, False, metrics)

        if (
            metrics.median_feature_displacement_px
            >= self.thresholds.min_median_feature_displacement_px
        ):
            return KeyframeSelectionResult(
                "selected", "sufficient_feature_displacement", True, metrics
            )
        if metrics.rotation_deg >= self.thresholds.min_rotation_deg:
            return KeyframeSelectionResult(
                "selected", "sufficient_rotation", True, metrics
            )
        if (
            metrics.frames_since_last_keyframe
            >= self.thresholds.max_frames_without_keyframe
        ):
            return KeyframeSelectionResult(
                "selected", "max_frame_gap", True, metrics
            )
        return KeyframeSelectionResult(
            "skipped", "insufficient_keyframe_motion", False, metrics
        )
