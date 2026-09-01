"""Diagnostic-only direct-versus-chained camera-pose consistency analysis."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .depth_types import CameraDepth
from .feature_tracker import FeatureTrackingError
from .pose_manager import PoseManager


@dataclass(frozen=True)
class PoseChainDiagnosticConfig:
    enabled: bool = False
    minimum_geometric_inlier_ratio: float = 0.25
    minimum_pnp_inlier_ratio: float = 0.25
    maximum_reprojection_rmse_pixels: float = 3.0
    translation_relative_difference_threshold: float = 0.25
    rotation_difference_threshold_deg: float = 5.0
    increasing_fraction_threshold: float = 0.70

    def __post_init__(self) -> None:
        for name in (
            "minimum_geometric_inlier_ratio",
            "minimum_pnp_inlier_ratio",
            "increasing_fraction_threshold",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in the range 0..1")
        for name in (
            "maximum_reprojection_rmse_pixels",
            "translation_relative_difference_threshold",
            "rotation_difference_threshold_deg",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class PoseChainFrameInput:
    frame_index: int
    image_bgr: np.ndarray
    camera_depth: CameraDepth
    world_from_camera: np.ndarray


@dataclass(frozen=True)
class DirectPoseEstimate:
    available: bool
    reason: str
    world_from_camera: np.ndarray | None
    feature_matches: int = 0
    geometric_inliers: int = 0
    geometric_inlier_ratio: float = 0.0
    pnp_inliers: int = 0
    pnp_inlier_ratio: float = 0.0
    reprojection_rmse: float | None = None


@dataclass(frozen=True)
class PoseChainDiagnosticRow:
    frame_index: int
    keyframe_sequence_index: int
    direct_pose_available: bool
    direct_pose_reason: str
    comparison_confidence: str
    comparison_low_confidence: bool
    chained_position_x: float
    chained_position_y: float
    chained_position_z: float
    direct_position_x: float | None
    direct_position_y: float | None
    direct_position_z: float | None
    chained_distance_from_origin: float
    direct_distance_from_origin: float | None
    translation_difference: float | None
    relative_translation_difference: float | None
    rotation_difference_deg: float | None
    direct_feature_matches: int
    direct_geometric_inliers: int
    direct_geometric_inlier_ratio: float
    direct_pnp_inliers: int
    direct_pnp_inlier_ratio: float
    direct_reprojection_rmse: float | None


class DirectPoseProvider(Protocol):
    def estimate(
        self,
        reference: PoseChainFrameInput,
        current: PoseChainFrameInput,
    ) -> DirectPoseEstimate: ...


def _validate_world_pose(pose: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(pose, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])
    ):
        raise ValueError(f"{name} must be a finite rigid 4x4 transform")
    return matrix


def rotation_difference_degrees(
    direct_world_from_camera: np.ndarray,
    chained_world_from_camera: np.ndarray,
) -> float:
    """Angle of R_direct^T R_chained for two world-from-camera rotations."""

    direct = _validate_world_pose(direct_world_from_camera, "direct pose")
    chained = _validate_world_pose(chained_world_from_camera, "chained pose")
    delta = direct[:3, :3].T @ chained[:3, :3]
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def compare_world_poses(
    chained_world_from_camera: np.ndarray,
    direct_world_from_camera: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Compare camera centers expressed in the same reference/world frame."""

    chained = _validate_world_pose(chained_world_from_camera, "chained pose")
    direct = _validate_world_pose(direct_world_from_camera, "direct pose")
    chained_position = chained[:3, 3]
    direct_position = direct[:3, 3]
    difference = float(np.linalg.norm(chained_position - direct_position))
    direct_distance = float(np.linalg.norm(direct_position))
    return {
        "chained_distance": float(np.linalg.norm(chained_position)),
        "direct_distance": direct_distance,
        "translation_difference": difference,
        "relative_translation_difference": difference / max(direct_distance, epsilon),
        "rotation_difference_deg": rotation_difference_degrees(direct, chained),
    }


def direct_world_pose_from_reference_to_current(
    reference_world_from_camera: np.ndarray,
    rotation_reference_to_current: np.ndarray,
    translation_reference_to_current: np.ndarray,
) -> np.ndarray:
    """Convert T_current_reference from PnP into T_world_current.

    DepthPoseEstimator returns ``X_current = R @ X_reference + t``.  The stored
    pose convention is world-from-camera, so the relative transform is inverted
    before right-composition with ``T_world_reference``.
    """

    return PoseManager.compose_world_pose(
        reference_world_from_camera,
        rotation_reference_to_current,
        translation_reference_to_current,
    )


class ReferenceDirectPoseEstimator:
    """Run isolated SIFT/Essential/PnP from the fixed reference keyframe."""

    def __init__(
        self,
        feature_tracker: Any,
        motion_estimator: Any,
        depth_pose_estimator: Any,
        camera_matrix: np.ndarray,
    ) -> None:
        self.feature_tracker = feature_tracker
        self.motion_estimator = motion_estimator
        self.depth_pose_estimator = depth_pose_estimator
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).copy()

    def estimate(
        self,
        reference: PoseChainFrameInput,
        current: PoseChainFrameInput,
    ) -> DirectPoseEstimate:
        try:
            matches = self.feature_tracker.match(
                reference.image_bgr, current.image_bgr
            )
        except (FeatureTrackingError, RuntimeError, TypeError, ValueError) as exc:
            return DirectPoseEstimate(False, f"direct_feature_matching_failed: {exc}", None)
        match_count = int(matches.statistics.good_matches)
        try:
            geometry = self.motion_estimator.estimate(
                matches.points1, matches.points2, self.camera_matrix
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return DirectPoseEstimate(
                False,
                f"direct_geometric_motion_failed: {exc}",
                None,
                feature_matches=match_count,
            )
        geometry_values = {
            "feature_matches": match_count,
            "geometric_inliers": int(geometry.num_inliers),
            "geometric_inlier_ratio": float(geometry.inlier_ratio),
        }
        if not geometry.success:
            return DirectPoseEstimate(
                False,
                f"direct_geometric_motion_rejected: {geometry.message}",
                None,
                **geometry_values,
            )
        try:
            pose = self.depth_pose_estimator.estimate(
                matches.points1,
                matches.points2,
                geometry.inlier_mask,
                reference.camera_depth,
                self.camera_matrix,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return DirectPoseEstimate(
                False, f"direct_depth_pnp_failed: {exc}", None, **geometry_values
            )
        quality = {
            **geometry_values,
            "pnp_inliers": int(pose.pnp_inliers),
            "pnp_inlier_ratio": float(pose.pnp_inlier_ratio),
            "reprojection_rmse": pose.reprojection_rmse_pixels,
        }
        if not pose.success or pose.rotation is None or pose.translation is None:
            return DirectPoseEstimate(
                False, f"direct_depth_pnp_rejected: {pose.message}", None, **quality
            )
        direct = direct_world_pose_from_reference_to_current(
            reference.world_from_camera, pose.rotation, pose.translation
        )
        return DirectPoseEstimate(
            True, "direct_depth_pnp_accepted", direct, **quality
        )


def _is_high_confidence(
    estimate: DirectPoseEstimate,
    config: PoseChainDiagnosticConfig,
) -> bool:
    rmse = estimate.reprojection_rmse
    return bool(
        estimate.available
        and estimate.geometric_inlier_ratio >= config.minimum_geometric_inlier_ratio
        and estimate.pnp_inlier_ratio >= config.minimum_pnp_inlier_ratio
        and rmse is not None
        and np.isfinite(rmse)
        and rmse <= config.maximum_reprojection_rmse_pixels
    )


def analyze_pose_chain(
    frames: tuple[PoseChainFrameInput, ...] | list[PoseChainFrameInput],
    estimator: DirectPoseProvider,
    config: PoseChainDiagnosticConfig,
) -> list[PoseChainDiagnosticRow]:
    if not frames:
        return []
    reference = frames[0]
    reference_pose = _validate_world_pose(reference.world_from_camera, "reference pose")
    rows: list[PoseChainDiagnosticRow] = []
    for sequence, frame in enumerate(frames):
        chained = _validate_world_pose(frame.world_from_camera, "chained pose")
        chained_position = chained[:3, 3]
        if sequence == 0:
            estimate = DirectPoseEstimate(
                True, "reference_identity", reference_pose.copy()
            )
            confidence = "reference"
        else:
            estimate = estimator.estimate(reference, frame)
            confidence = (
                "high" if _is_high_confidence(estimate, config)
                else "low" if estimate.available
                else "unavailable"
            )
        direct_position: np.ndarray | None = None
        comparison: dict[str, float] | None = None
        if estimate.available and estimate.world_from_camera is not None:
            direct_pose = _validate_world_pose(
                estimate.world_from_camera, "direct pose"
            )
            direct_position = direct_pose[:3, 3]
            comparison = compare_world_poses(chained, direct_pose)
        rows.append(PoseChainDiagnosticRow(
            frame_index=int(frame.frame_index),
            keyframe_sequence_index=sequence,
            direct_pose_available=estimate.available,
            direct_pose_reason=estimate.reason,
            comparison_confidence=confidence,
            comparison_low_confidence=confidence in {"low", "unavailable"},
            chained_position_x=float(chained_position[0]),
            chained_position_y=float(chained_position[1]),
            chained_position_z=float(chained_position[2]),
            direct_position_x=None if direct_position is None else float(direct_position[0]),
            direct_position_y=None if direct_position is None else float(direct_position[1]),
            direct_position_z=None if direct_position is None else float(direct_position[2]),
            chained_distance_from_origin=float(np.linalg.norm(chained_position)),
            direct_distance_from_origin=(
                None if comparison is None else comparison["direct_distance"]
            ),
            translation_difference=(
                None if comparison is None else comparison["translation_difference"]
            ),
            relative_translation_difference=(
                None
                if comparison is None
                else comparison["relative_translation_difference"]
            ),
            rotation_difference_deg=(
                None if comparison is None else comparison["rotation_difference_deg"]
            ),
            direct_feature_matches=estimate.feature_matches,
            direct_geometric_inliers=estimate.geometric_inliers,
            direct_geometric_inlier_ratio=estimate.geometric_inlier_ratio,
            direct_pnp_inliers=estimate.pnp_inliers,
            direct_pnp_inlier_ratio=estimate.pnp_inlier_ratio,
            direct_reprojection_rmse=estimate.reprojection_rmse,
        ))
    return rows


def _statistics(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("min", "max", "median", "last")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "median": float(np.median(array)),
        "last": float(array[-1]),
    }


def _slope(rows: list[PoseChainDiagnosticRow], field: str) -> float | None:
    pairs = [
        (row.keyframe_sequence_index, getattr(row, field))
        for row in rows
        if getattr(row, field) is not None
    ]
    if len(pairs) < 2:
        return None
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def _increasing_fraction(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    differences = np.diff(np.asarray(values, dtype=np.float64))
    return float(np.mean(differences >= 0.0))


def build_pose_chain_summary(
    rows: list[PoseChainDiagnosticRow],
    config: PoseChainDiagnosticConfig,
) -> dict[str, Any]:
    comparisons = rows[1:]
    successful = [row for row in comparisons if row.direct_pose_available]
    confident = [row for row in comparisons if row.comparison_confidence == "high"]
    translation = [float(row.translation_difference) for row in confident]
    relative = [float(row.relative_translation_difference) for row in confident]
    rotation = [float(row.rotation_difference_deg) for row in confident]
    translation_increasing = _increasing_fraction(translation)
    rotation_increasing = _increasing_fraction(rotation)
    translation_slope = _slope(confident, "translation_difference")
    relative_slope = _slope(confident, "relative_translation_difference")
    rotation_slope = _slope(confident, "rotation_difference_deg")
    required_confident = max(2, int(np.ceil(0.5 * len(comparisons))))
    insufficient = len(confident) < required_confident
    translation_drift = bool(
        not insufficient
        and len(translation) >= 3
        and translation_slope is not None
        and translation_slope > 0.0
        and translation_increasing is not None
        and translation_increasing >= config.increasing_fraction_threshold
        and relative[-1] >= config.translation_relative_difference_threshold
    )
    rotation_drift = bool(
        not insufficient
        and len(rotation) >= 3
        and rotation_slope is not None
        and rotation_slope > 0.0
        and rotation_increasing is not None
        and rotation_increasing >= config.increasing_fraction_threshold
        and rotation[-1] >= config.rotation_difference_threshold_deg
    )
    return {
        "diagnostic_only": True,
        "coordinate_scale": "relative_non_metric",
        "transform_conventions": {
            "pnp_relative": "current_from_reference",
            "stored_pose": "world_from_camera",
            "rotation_comparison": "R_direct^T @ R_chained",
        },
        "accepted_keyframe_count": len(rows),
        "direct_pose_success_count": len(successful),
        "direct_pose_failure_count": len(comparisons) - len(successful),
        "high_confidence_comparison_count": len(confident),
        "low_confidence_or_unavailable_count": len(comparisons) - len(confident),
        "statistics_scope": "high_confidence_direct_poses_only_excluding_reference",
        "translation_difference": _statistics(translation),
        "relative_translation_difference": _statistics(relative),
        "rotation_difference_deg": _statistics(rotation),
        "descriptive_slope_per_keyframe": {
            "translation_difference": translation_slope,
            "relative_translation_difference": relative_slope,
            "rotation_difference_deg": rotation_slope,
            "interpretation": "descriptive slopes only; no statistical significance claimed",
        },
        "heuristic_warning_flags": {
            "heuristic_only": True,
            "pose_chain_translation_drift_suspected": translation_drift,
            "pose_chain_rotation_drift_suspected": rotation_drift,
            "direct_pose_quality_insufficient": insufficient,
            "conditions": {
                "translation_relative_difference_threshold": config.translation_relative_difference_threshold,
                "rotation_difference_threshold_deg": config.rotation_difference_threshold_deg,
                "increasing_fraction_threshold": config.increasing_fraction_threshold,
                "minimum_required_high_confidence_comparisons": required_confident,
            },
            "observed": {
                "translation_increasing_fraction": translation_increasing,
                "rotation_increasing_fraction": rotation_increasing,
                "final_relative_translation_difference": relative[-1] if relative else None,
                "final_rotation_difference_deg": rotation[-1] if rotation else None,
            },
        },
    }


def _plot_series(
    path: Path,
    rows: list[PoseChainDiagnosticRow],
    fields: tuple[str, ...],
    labels: tuple[str, ...],
    ylabel: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for field, label in zip(fields, labels):
        valid = [row for row in rows if getattr(row, field) is not None]
        axis.plot(
            [row.frame_index for row in valid],
            [getattr(row, field) for row in valid],
            marker="o",
            label=label,
        )
    axis.set_xlabel("Accepted keyframe source-frame index")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    if len(fields) > 1:
        axis.legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_trajectory(
    path: Path,
    rows: list[PoseChainDiagnosticRow],
    axes: tuple[str, str],
    labels: tuple[str, str],
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for prefix, label in (("chained", "Chained"), ("direct", "Direct")):
        x_field = f"{prefix}_position_{axes[0]}"
        y_field = f"{prefix}_position_{axes[1]}"
        valid = [
            row for row in rows
            if getattr(row, x_field) is not None and getattr(row, y_field) is not None
        ]
        axis.plot(
            [getattr(row, x_field) for row in valid],
            [getattr(row, y_field) for row in valid],
            marker="o",
            label=label,
        )
    axis.set_xlabel(f"{labels[0]} (relative/non-metric)")
    axis.set_ylabel(f"{labels[1]} (relative/non-metric)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.axis("equal")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_pose_chain_diagnostics(
    relative_map_directory: str | Path,
    rows: list[PoseChainDiagnosticRow],
    config: PoseChainDiagnosticConfig,
) -> dict[str, Any]:
    output = Path(relative_map_directory) / "pose_chain_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) for row in rows]
    csv_path = output / "pose_chain_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        if dictionaries:
            writer = csv.DictWriter(file, fieldnames=list(dictionaries[0]))
            writer.writeheader()
            writer.writerows(dictionaries)
    json_path = output / "pose_chain_diagnostics.json"
    json_path.write_text(json.dumps(dictionaries, indent=2), encoding="utf-8")
    summary = build_pose_chain_summary(rows, config)
    summary_path = output / "pose_chain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plots = {
        "translation_difference_vs_frame.png": (
            ("translation_difference",), ("Translation difference",),
            "Translation difference (relative/non-metric)",
        ),
        "relative_translation_difference_vs_frame.png": (
            ("relative_translation_difference",), ("Relative difference",),
            "Relative translation difference (ratio)",
        ),
        "rotation_difference_vs_frame.png": (
            ("rotation_difference_deg",), ("Rotation difference",),
            "Rotation difference (degrees)",
        ),
        "chained_vs_direct_distance.png": (
            ("chained_distance_from_origin", "direct_distance_from_origin"),
            ("Chained", "Direct"),
            "Distance from origin (relative/non-metric)",
        ),
    }
    plot_paths: list[Path] = []
    for filename, (fields, labels, ylabel) in plots.items():
        path = output / filename
        _plot_series(path, rows, fields, labels, ylabel)
        plot_paths.append(path)
    for filename, axes, labels in (
        ("chained_vs_direct_xz.png", ("x", "z"), ("X", "Z")),
        ("chained_vs_direct_xy.png", ("x", "y"), ("X", "Y")),
    ):
        path = output / filename
        _plot_trajectory(path, rows, axes, labels)
        plot_paths.append(path)

    overview_path = output / "pose_chain_overview.png"
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    overview_fields = (
        ("translation_difference", "Translation difference", "Relative/non-metric"),
        ("relative_translation_difference", "Relative translation difference", "Ratio"),
        ("rotation_difference_deg", "Rotation difference", "Degrees"),
    )
    for axis, (field, title, ylabel) in zip(axes.flat[:3], overview_fields):
        valid = [row for row in rows if getattr(row, field) is not None]
        axis.plot(
            [row.frame_index for row in valid],
            [getattr(row, field) for row in valid],
            marker="o",
        )
        axis.set_title(title)
        axis.set_xlabel("Accepted source-frame index")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    distance_axis = axes.flat[3]
    distance_axis.plot(
        [row.frame_index for row in rows],
        [row.chained_distance_from_origin for row in rows],
        marker="o", label="Chained",
    )
    direct_rows = [row for row in rows if row.direct_distance_from_origin is not None]
    distance_axis.plot(
        [row.frame_index for row in direct_rows],
        [row.direct_distance_from_origin for row in direct_rows],
        marker="o", label="Direct",
    )
    distance_axis.set_title("Distance from origin")
    distance_axis.set_xlabel("Accepted source-frame index")
    distance_axis.set_ylabel("Relative/non-metric")
    distance_axis.grid(True, alpha=0.3)
    distance_axis.legend()
    figure.suptitle("Pose-chain consistency — diagnostic only, relative/non-metric")
    figure.savefig(overview_path, dpi=150)
    plt.close(figure)
    plot_paths.append(overview_path)
    return {
        "directory": output,
        "csv": csv_path,
        "json": json_path,
        "summary_path": summary_path,
        "plots": plot_paths,
        "summary": summary,
    }
