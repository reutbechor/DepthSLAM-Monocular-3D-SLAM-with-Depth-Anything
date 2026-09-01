"""Optional robust 3D-to-3D refinement of a depth-assisted PnP pose.

The transform convention is explicit throughout this module:

    P_current = R_current_previous @ P_previous + t_current_previous

All coordinates remain in the existing relative/non-metric depth units.  No
similarity scale is estimated or applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .depth_geometry import DepthGeometryProcessor
from .depth_types import CameraDepth
from .transforms import transform_points


@dataclass(frozen=True)
class PoseRefinement3DConfig:
    """Conservative acceptance and deterministic RANSAC settings."""

    enabled: bool = False
    minimum_correspondences: int = 100
    minimum_inliers: int = 80
    minimum_inlier_ratio: float = 0.40
    minimum_relative_improvement: float = 0.10
    random_seed: int = 0
    ransac_iterations: int = 512
    residual_threshold_fraction: float = 0.05
    maximum_translation_change_ratio: float = 2.0
    maximum_rotation_change_degrees: float = 10.0

    def __post_init__(self) -> None:
        if self.minimum_correspondences < 3 or self.minimum_inliers < 3:
            raise ValueError("3D refinement minima must be at least 3")
        for name, value in (
            ("minimum_inlier_ratio", self.minimum_inlier_ratio),
            ("minimum_relative_improvement", self.minimum_relative_improvement),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.ransac_iterations < 1:
            raise ValueError("ransac_iterations must be positive")
        if self.residual_threshold_fraction <= 0.0:
            raise ValueError("residual_threshold_fraction must be positive")
        if self.maximum_translation_change_ratio <= 0.0:
            raise ValueError("maximum_translation_change_ratio must be positive")
        if self.maximum_rotation_change_degrees <= 0.0:
            raise ValueError("maximum_rotation_change_degrees must be positive")


@dataclass(frozen=True)
class Correspondences3D:
    """Matched relative-depth points in previous and current camera frames."""

    previous_points: np.ndarray
    current_points: np.ndarray
    match_indices: np.ndarray
    supported_match_count: int
    coordinate_units: str
    is_metric: bool

    @property
    def count(self) -> int:
        return int(self.previous_points.shape[0])


@dataclass(frozen=True)
class ResidualMetrics:
    mean: float
    median: float
    p90: float
    rmse: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "median": self.median,
            "p90": self.p90,
            "rmse": self.rmse,
        }


@dataclass(frozen=True)
class RigidAlignmentResult:
    success: bool
    message: str
    rotation: np.ndarray | None
    translation: np.ndarray | None
    inlier_mask: np.ndarray
    residual_threshold: float | None
    residuals: np.ndarray

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))


@dataclass(frozen=True)
class PoseRefinement3DResult:
    """Baseline, candidate, selected pose, and conservative gate decision."""

    attempted: bool
    accepted: bool
    reason: str
    correspondence_count: int
    inlier_count: int
    inlier_ratio: float
    residual_threshold: float | None
    baseline_metrics: ResidualMetrics | None
    refined_metrics: ResidualMetrics | None
    relative_improvement: float | None
    baseline_rotation: np.ndarray
    baseline_translation: np.ndarray
    refined_rotation: np.ndarray | None
    refined_translation: np.ndarray | None
    selected_rotation: np.ndarray
    selected_translation: np.ndarray
    inlier_mask: np.ndarray
    rotation_change_degrees: float | None
    translation_change_ratio: float | None
    coordinate_units: str
    is_metric: bool

    def frame_diagnostics(self) -> dict[str, Any]:
        """Return fields written verbatim into frame_stats.jsonl."""

        return {
            "refinement_3d_attempted": self.attempted,
            "correspondence_3d_count": self.correspondence_count,
            "refinement_3d_inliers": self.inlier_count,
            "refinement_3d_inlier_ratio": self.inlier_ratio,
            "baseline_3d_residual_median": (
                None if self.baseline_metrics is None else self.baseline_metrics.median
            ),
            "refined_3d_residual_median": (
                None if self.refined_metrics is None else self.refined_metrics.median
            ),
            "baseline_3d_residual_rmse": (
                None if self.baseline_metrics is None else self.baseline_metrics.rmse
            ),
            "refined_3d_residual_rmse": (
                None if self.refined_metrics is None else self.refined_metrics.rmse
            ),
            "refinement_3d_relative_improvement": self.relative_improvement,
            "refinement_3d_accepted": self.accepted,
            "refinement_3d_reason": self.reason,
        }

    def metrics_dict(self) -> dict[str, Any]:
        """Return the complete JSON-safe pairwise experiment record."""

        return {
            "transform_convention": "current_from_previous",
            "equation": "P_current = R @ P_previous + t",
            "coordinate_units": self.coordinate_units,
            "is_metric": self.is_metric,
            "scale_estimation_applied": False,
            "applied_scale": 1.0,
            "3d_correspondence_count": self.correspondence_count,
            "3d_refinement_inliers": self.inlier_count,
            "3d_refinement_inlier_ratio": self.inlier_ratio,
            "residual_threshold_relative_units": self.residual_threshold,
            "baseline_3d_residual": (
                None if self.baseline_metrics is None else self.baseline_metrics.to_dict()
            ),
            "refined_3d_residual": (
                None if self.refined_metrics is None else self.refined_metrics.to_dict()
            ),
            "3d_refinement_relative_improvement": self.relative_improvement,
            "pnp_rotation_current_from_previous": self.baseline_rotation.tolist(),
            "pnp_translation_current_from_previous": self.baseline_translation.tolist(),
            "refined_rotation_current_from_previous": (
                None if self.refined_rotation is None else self.refined_rotation.tolist()
            ),
            "refined_translation_current_from_previous": (
                None
                if self.refined_translation is None
                else self.refined_translation.tolist()
            ),
            "selected_rotation_current_from_previous": self.selected_rotation.tolist(),
            "selected_translation_current_from_previous": self.selected_translation.tolist(),
            "rotation_change_degrees": self.rotation_change_degrees,
            "translation_change_ratio": self.translation_change_ratio,
            "3d_refinement_attempted": self.attempted,
            "3d_refinement_accepted": self.accepted,
            "3d_refinement_reason": self.reason,
        }


@dataclass(frozen=True)
class PairAlignmentClouds:
    """The same two camera clouds under baseline and selected transforms."""

    before_points: np.ndarray
    after_points: np.ndarray
    colors: np.ndarray
    previous_point_count: int
    current_point_count: int


def build_3d_correspondences(
    points_previous: np.ndarray,
    points_current: np.ndarray,
    supported_match_mask: np.ndarray,
    previous_camera_depth: CameraDepth,
    current_aligned_camera_depth: CameraDepth,
    camera_matrix: np.ndarray,
    *,
    sampling_method: str = "bilinear",
) -> Correspondences3D:
    """Backproject matches with valid depth on both sides.

    `supported_match_mask` should be the full-match PnP inlier mask, so this
    function cannot introduce image matches that were not already supported by
    the Essential-Matrix/PnP pipeline.
    """

    previous = np.asarray(points_previous, dtype=np.float64)
    current = np.asarray(points_current, dtype=np.float64)
    if previous.ndim != 2 or previous.shape[1:] != (2,):
        raise ValueError("points_previous must be an Nx2 array")
    if current.shape != previous.shape:
        raise ValueError("points_current must match points_previous")
    mask = np.asarray(supported_match_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != previous.shape[0]:
        raise ValueError("supported_match_mask length must equal match count")
    if previous_camera_depth.coordinate_units != current_aligned_camera_depth.coordinate_units:
        raise ValueError("previous and current camera depth units must match")
    if previous_camera_depth.is_metric != current_aligned_camera_depth.is_metric:
        raise ValueError("previous and current depth metric semantics must match")

    processor = DepthGeometryProcessor(sampling_method)
    previous_geometry = processor.process(
        previous, mask, previous_camera_depth, camera_matrix
    )
    current_geometry = processor.process(
        current, mask, current_aligned_camera_depth, camera_matrix
    )
    common, previous_locations, current_locations = np.intersect1d(
        previous_geometry.valid_match_indices,
        current_geometry.valid_match_indices,
        assume_unique=True,
        return_indices=True,
    )
    previous_3d = previous_geometry.points_3d_relative[previous_locations]
    current_3d = current_geometry.points_3d_relative[current_locations]
    finite = np.all(np.isfinite(previous_3d), axis=1) & np.all(
        np.isfinite(current_3d), axis=1
    )
    return Correspondences3D(
        previous_points=previous_3d[finite].astype(np.float64, copy=False),
        current_points=current_3d[finite].astype(np.float64, copy=False),
        match_indices=common[finite].astype(np.int64, copy=False),
        supported_match_count=int(np.count_nonzero(mask)),
        coordinate_units=previous_camera_depth.coordinate_units,
        is_metric=previous_camera_depth.is_metric,
    )


def kabsch_rigid_transform(
    source_points: np.ndarray, target_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the proper rigid transform `target = R @ source + t`."""

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or target.shape != source.shape:
        raise ValueError("source_points and target_points must be equal Nx3 arrays")
    if source.shape[0] < 3:
        raise ValueError("at least 3 correspondences are required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("3D correspondences must be finite")

    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    if np.linalg.matrix_rank(source_centered) < 2 or np.linalg.matrix_rank(
        target_centered
    ) < 2:
        raise ValueError("3D correspondences must contain 3 non-collinear points")

    covariance = source_centered.T @ target_centered
    try:
        u, _, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"Kabsch SVD failed: {exc}") from exc
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("Kabsch produced a non-finite transform")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("Kabsch rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("Kabsch rotation does not have determinant +1")
    return rotation, translation


def correspondence_residuals(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    rotation_current_previous: np.ndarray,
    translation_current_previous: np.ndarray,
) -> np.ndarray:
    transformed = transform_points(
        previous_points, rotation_current_previous, translation_current_previous
    )
    target = np.asarray(current_points, dtype=np.float64)
    if target.shape != transformed.shape or not np.isfinite(target).all():
        raise ValueError("current_points must be a finite array matching previous_points")
    return np.linalg.norm(transformed - target, axis=1)


def residual_metrics(residuals: np.ndarray) -> ResidualMetrics:
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("residuals must be a non-empty finite non-negative array")
    return ResidualMetrics(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p90=float(np.percentile(values, 90.0)),
        rmse=float(np.sqrt(np.mean(values ** 2))),
    )


def adaptive_residual_threshold(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    threshold_fraction: float,
) -> float:
    """Derive a relative-unit RANSAC threshold from robust scene spread."""

    if threshold_fraction <= 0.0:
        raise ValueError("threshold_fraction must be positive")
    spreads: list[np.ndarray] = []
    for points in (previous_points, current_points):
        array = np.asarray(points, dtype=np.float64)
        center = np.median(array, axis=0)
        spreads.append(np.linalg.norm(array - center, axis=1))
    robust_scale = float(np.median(np.concatenate(spreads)))
    maximum = float(max(np.max(np.abs(previous_points)), np.max(np.abs(current_points))))
    numerical_floor = np.finfo(np.float64).eps * max(maximum, 1e-12) * 10_000.0
    threshold = max(threshold_fraction * robust_scale, numerical_floor)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("3D correspondences do not have a usable spatial spread")
    return threshold


def robust_rigid_alignment(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    *,
    random_seed: int = 0,
    iterations: int = 512,
    threshold_fraction: float = 0.05,
) -> RigidAlignmentResult:
    """Estimate `current_from_previous` with 3-point RANSAC and final Kabsch."""

    previous = np.asarray(previous_points, dtype=np.float64)
    current = np.asarray(current_points, dtype=np.float64)
    count = previous.shape[0] if previous.ndim == 2 else 0
    empty_mask = np.zeros(count, dtype=bool)
    empty_residuals = np.empty(0, dtype=np.float64)
    if (
        previous.ndim != 2
        or previous.shape[1:] != (3,)
        or current.shape != previous.shape
        or count < 3
        or not np.isfinite(previous).all()
        or not np.isfinite(current).all()
    ):
        return RigidAlignmentResult(
            False, "invalid or insufficient 3D correspondences", None, None,
            empty_mask, None, empty_residuals,
        )
    try:
        threshold = adaptive_residual_threshold(
            previous, current, threshold_fraction
        )
    except ValueError as exc:
        return RigidAlignmentResult(
            False, str(exc), None, None, empty_mask, None, empty_residuals
        )

    generator = np.random.default_rng(random_seed)
    best_mask: np.ndarray | None = None
    best_median = float("inf")
    for _ in range(iterations):
        sample = generator.choice(count, size=3, replace=False)
        try:
            rotation, translation = kabsch_rigid_transform(
                previous[sample], current[sample]
            )
        except ValueError:
            continue
        residuals = correspondence_residuals(
            previous, current, rotation, translation
        )
        mask = residuals <= threshold
        inliers = int(np.count_nonzero(mask))
        if inliers < 3:
            continue
        median = float(np.median(residuals[mask]))
        if (
            best_mask is None
            or inliers > int(np.count_nonzero(best_mask))
            or (inliers == int(np.count_nonzero(best_mask)) and median < best_median)
        ):
            best_mask = mask
            best_median = median

    if best_mask is None:
        return RigidAlignmentResult(
            False, "3D RANSAC found no non-degenerate model", None, None,
            empty_mask, threshold, empty_residuals,
        )
    try:
        rotation, translation = kabsch_rigid_transform(
            previous[best_mask], current[best_mask]
        )
        residuals = correspondence_residuals(
            previous, current, rotation, translation
        )
        final_mask = residuals <= threshold
        if np.count_nonzero(final_mask) < 3:
            raise ValueError("fewer than 3 final 3D inliers")
        rotation, translation = kabsch_rigid_transform(
            previous[final_mask], current[final_mask]
        )
        residuals = correspondence_residuals(
            previous, current, rotation, translation
        )
        final_mask = residuals <= threshold
    except ValueError as exc:
        return RigidAlignmentResult(
            False, f"final Kabsch failed: {exc}", None, None,
            empty_mask, threshold, empty_residuals,
        )
    return RigidAlignmentResult(
        True,
        "robust current_from_previous rigid transform estimated",
        rotation,
        translation,
        final_mask,
        threshold,
        residuals,
    )


def _rotation_difference_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first) @ np.asarray(second).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class RobustPoseRefiner3D:
    """Robustly refine a baseline PnP pose and gate its use."""

    def __init__(self, config: PoseRefinement3DConfig | None = None) -> None:
        self.config = config or PoseRefinement3DConfig()

    def _fallback(
        self,
        reason: str,
        correspondences: Correspondences3D,
        baseline_rotation: np.ndarray,
        baseline_translation: np.ndarray,
        baseline_metrics: ResidualMetrics | None,
        *,
        attempted: bool,
        alignment: RigidAlignmentResult | None = None,
        refined_metrics: ResidualMetrics | None = None,
        relative_improvement: float | None = None,
        rotation_change: float | None = None,
        translation_change_ratio: float | None = None,
    ) -> PoseRefinement3DResult:
        return PoseRefinement3DResult(
            attempted=attempted,
            accepted=False,
            reason=reason,
            correspondence_count=correspondences.count,
            inlier_count=0 if alignment is None else alignment.inlier_count,
            inlier_ratio=(
                0.0
                if alignment is None or correspondences.count == 0
                else alignment.inlier_count / correspondences.count
            ),
            residual_threshold=(
                None if alignment is None else alignment.residual_threshold
            ),
            baseline_metrics=baseline_metrics,
            refined_metrics=refined_metrics,
            relative_improvement=relative_improvement,
            baseline_rotation=baseline_rotation.copy(),
            baseline_translation=baseline_translation.copy(),
            refined_rotation=(
                None if alignment is None or alignment.rotation is None
                else alignment.rotation.copy()
            ),
            refined_translation=(
                None if alignment is None or alignment.translation is None
                else alignment.translation.copy()
            ),
            selected_rotation=baseline_rotation.copy(),
            selected_translation=baseline_translation.copy(),
            inlier_mask=(
                np.zeros(correspondences.count, dtype=bool)
                if alignment is None else alignment.inlier_mask.copy()
            ),
            rotation_change_degrees=rotation_change,
            translation_change_ratio=translation_change_ratio,
            coordinate_units=correspondences.coordinate_units,
            is_metric=correspondences.is_metric,
        )

    def refine(
        self,
        correspondences: Correspondences3D,
        baseline_rotation_current_previous: np.ndarray,
        baseline_translation_current_previous: np.ndarray,
    ) -> PoseRefinement3DResult:
        baseline_rotation = np.asarray(
            baseline_rotation_current_previous, dtype=np.float64
        )
        baseline_translation = np.asarray(
            baseline_translation_current_previous, dtype=np.float64
        ).reshape(-1)
        if baseline_rotation.shape != (3, 3) or baseline_translation.shape != (3,):
            raise ValueError("baseline pose must contain a 3x3 R and 3-vector t")
        if not np.isfinite(baseline_rotation).all() or not np.isfinite(
            baseline_translation
        ).all():
            raise ValueError("baseline pose must be finite")

        baseline_metrics: ResidualMetrics | None = None
        if correspondences.count > 0:
            baseline_metrics = residual_metrics(correspondence_residuals(
                correspondences.previous_points,
                correspondences.current_points,
                baseline_rotation,
                baseline_translation,
            ))
        if not self.config.enabled:
            return self._fallback(
                "3d_refinement_disabled",
                correspondences,
                baseline_rotation,
                baseline_translation,
                baseline_metrics,
                attempted=False,
            )
        if correspondences.count < self.config.minimum_correspondences:
            return self._fallback(
                "3d_refinement_insufficient_correspondences",
                correspondences,
                baseline_rotation,
                baseline_translation,
                baseline_metrics,
                attempted=True,
            )

        alignment = robust_rigid_alignment(
            correspondences.previous_points,
            correspondences.current_points,
            random_seed=self.config.random_seed,
            iterations=self.config.ransac_iterations,
            threshold_fraction=self.config.residual_threshold_fraction,
        )
        if not alignment.success or alignment.rotation is None or alignment.translation is None:
            return self._fallback(
                "3d_refinement_invalid_transform",
                correspondences,
                baseline_rotation,
                baseline_translation,
                baseline_metrics,
                attempted=True,
                alignment=alignment,
            )

        ratio = alignment.inlier_count / correspondences.count
        refined_metrics = residual_metrics(alignment.residuals)
        if baseline_metrics is None or baseline_metrics.median <= np.finfo(float).eps:
            improvement = 0.0
        else:
            improvement = (
                baseline_metrics.median - refined_metrics.median
            ) / baseline_metrics.median
        rotation_change = _rotation_difference_degrees(
            alignment.rotation, baseline_rotation
        )
        baseline_norm = float(np.linalg.norm(baseline_translation))
        translation_delta = float(np.linalg.norm(
            alignment.translation - baseline_translation
        ))
        translation_floor = max(
            baseline_norm,
            alignment.residual_threshold or 0.0,
            np.finfo(float).eps,
        )
        translation_change_ratio = translation_delta / translation_floor

        if (
            alignment.inlier_count < self.config.minimum_inliers
            or ratio < self.config.minimum_inlier_ratio
        ):
            return self._fallback(
                "3d_refinement_insufficient_inliers",
                correspondences,
                baseline_rotation,
                baseline_translation,
                baseline_metrics,
                attempted=True,
                alignment=alignment,
                refined_metrics=refined_metrics,
                relative_improvement=improvement,
                rotation_change=rotation_change,
                translation_change_ratio=translation_change_ratio,
            )
        if improvement < self.config.minimum_relative_improvement:
            return self._fallback(
                "3d_refinement_not_improved",
                correspondences,
                baseline_rotation,
                baseline_translation,
                baseline_metrics,
                attempted=True,
                alignment=alignment,
                refined_metrics=refined_metrics,
                relative_improvement=improvement,
                rotation_change=rotation_change,
                translation_change_ratio=translation_change_ratio,
            )
        if (
            rotation_change > self.config.maximum_rotation_change_degrees
            or translation_change_ratio > self.config.maximum_translation_change_ratio
        ):
            return self._fallback(
                "3d_refinement_extreme_pose_jump",
                correspondences,
                baseline_rotation,
                baseline_translation,
                baseline_metrics,
                attempted=True,
                alignment=alignment,
                refined_metrics=refined_metrics,
                relative_improvement=improvement,
                rotation_change=rotation_change,
                translation_change_ratio=translation_change_ratio,
            )

        return PoseRefinement3DResult(
            attempted=True,
            accepted=True,
            reason="3d_refinement_accepted",
            correspondence_count=correspondences.count,
            inlier_count=alignment.inlier_count,
            inlier_ratio=ratio,
            residual_threshold=alignment.residual_threshold,
            baseline_metrics=baseline_metrics,
            refined_metrics=refined_metrics,
            relative_improvement=improvement,
            baseline_rotation=baseline_rotation.copy(),
            baseline_translation=baseline_translation.copy(),
            refined_rotation=alignment.rotation.copy(),
            refined_translation=alignment.translation.copy(),
            selected_rotation=alignment.rotation.copy(),
            selected_translation=alignment.translation.copy(),
            inlier_mask=alignment.inlier_mask.copy(),
            rotation_change_degrees=rotation_change,
            translation_change_ratio=translation_change_ratio,
            coordinate_units=correspondences.coordinate_units,
            is_metric=correspondences.is_metric,
        )


def build_pair_alignment_clouds(
    previous_points: np.ndarray,
    previous_colors: np.ndarray,
    current_points: np.ndarray,
    current_colors: np.ndarray,
    previous_world_pose: np.ndarray,
    baseline_current_world_pose: np.ndarray,
    selected_current_world_pose: np.ndarray,
) -> PairAlignmentClouds:
    """Build before/after clouds without modifying either raw camera cloud."""

    previous = np.asarray(previous_points, dtype=np.float64)
    current = np.asarray(current_points, dtype=np.float64)
    previous_rgb = np.asarray(previous_colors)
    current_rgb = np.asarray(current_colors)
    if previous_rgb.shape != previous.shape or current_rgb.shape != current.shape:
        raise ValueError("each point cloud must have matching Nx3 colors")
    poses = [
        np.asarray(pose, dtype=np.float64)
        for pose in (
            previous_world_pose,
            baseline_current_world_pose,
            selected_current_world_pose,
        )
    ]
    if any(
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0])
        for pose in poses
    ):
        raise ValueError("world poses must be finite rigid 4x4 transforms")

    previous_world = transform_points(previous, poses[0][:3, :3], poses[0][:3, 3])
    current_before = transform_points(current, poses[1][:3, :3], poses[1][:3, 3])
    current_after = transform_points(current, poses[2][:3, :3], poses[2][:3, 3])
    colors = np.vstack((previous_rgb, current_rgb)).copy()
    return PairAlignmentClouds(
        before_points=np.vstack((previous_world, current_before)),
        after_points=np.vstack((previous_world, current_after)),
        colors=colors,
        previous_point_count=previous.shape[0],
        current_point_count=current.shape[0],
    )
