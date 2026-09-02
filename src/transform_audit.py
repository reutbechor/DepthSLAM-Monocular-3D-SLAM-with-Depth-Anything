"""Diagnostic helpers for auditing rigid-transform and fusion conventions.

Canonical notation is ``T_A_from_B``: a column point in frame B is mapped to
frame A by ``p_A = R_A_from_B @ p_B + t_A_from_B``.  Project point arrays are
stored as Nx3 rows, so the equivalent implementation is ``points @ R.T + t``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .transforms import invert_transform, make_transform, transform_points


def _validate_rigid_transform(transform: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1e-6)
    ):
        raise ValueError(f"{name} must be a finite rigid 4x4 transform")
    return matrix


def camera_center_world_from_world_from_camera(
    transform_world_from_camera: np.ndarray,
) -> np.ndarray:
    """Return the camera origin expressed in world coordinates."""

    transform = _validate_rigid_transform(
        transform_world_from_camera, "T_world_from_camera"
    )
    return transform[:3, 3].copy()


def camera_center_world_from_camera_from_world(
    transform_camera_from_world: np.ndarray,
) -> np.ndarray:
    """Return ``-R.T @ t`` for an OpenCV-style world-to-camera pose."""

    transform = _validate_rigid_transform(
        transform_camera_from_world, "T_camera_from_world"
    )
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return -(rotation.T @ translation)


@dataclass(frozen=True)
class TransformTraceEntry:
    frame_index: int
    local_transform_convention: str
    T_current_from_previous: list[list[float]]
    T_previous_from_current: list[list[float]]
    T_world_from_camera: list[list[float]]
    camera_center_world: list[float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def make_transform_trace_entry(
    frame_index: int,
    rotation_current_from_previous: np.ndarray,
    translation_current_from_previous: np.ndarray,
    transform_world_from_camera: np.ndarray,
) -> TransformTraceEntry:
    """Capture explicit local, inverse-local, and accumulated 4x4 transforms."""

    current_from_previous = make_transform(
        rotation_current_from_previous, translation_current_from_previous
    )
    previous_from_current = invert_transform(current_from_previous)
    world_from_camera = _validate_rigid_transform(
        transform_world_from_camera, "T_world_from_camera"
    )
    center = camera_center_world_from_world_from_camera(world_from_camera)
    return TransformTraceEntry(
        frame_index=int(frame_index),
        local_transform_convention="p_current = T_current_from_previous @ p_previous",
        T_current_from_previous=current_from_previous.tolist(),
        T_previous_from_current=previous_from_current.tolist(),
        T_world_from_camera=world_from_camera.tolist(),
        camera_center_world=center.tolist(),
    )


@dataclass(frozen=True)
class TwoFrameTransformConsistency:
    previous_frame_index: int
    current_frame_index: int
    correspondence_count: int
    local_median_discrepancy: float | None
    local_rmse_discrepancy: float | None
    world_median_discrepancy: float | None
    world_rmse_discrepancy: float | None
    maximum_local_world_residual_difference: float | None
    coordinate_units: str
    is_metric: bool
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update({
            "canonical_notation": "T_A_from_B maps p_B to p_A",
            "local_equation": (
                "p_current = T_current_from_previous @ p_previous"
            ),
            "fusion_equations": (
                "p_world_previous = T_world_from_previous @ p_previous; "
                "p_world_current = T_world_from_current @ p_current"
            ),
            "diagnostic_only": True,
        })
        return result


def evaluate_two_frame_transform_consistency(
    previous_points_camera: np.ndarray,
    current_points_camera: np.ndarray,
    rotation_current_from_previous: np.ndarray,
    translation_current_from_previous: np.ndarray,
    transform_world_from_previous: np.ndarray,
    transform_world_from_current: np.ndarray,
    *,
    previous_frame_index: int,
    current_frame_index: int,
    coordinate_units: str,
    is_metric: bool,
) -> TwoFrameTransformConsistency:
    """Compare local and independently world-transformed matched 3D points."""

    previous = np.asarray(previous_points_camera, dtype=np.float64)
    current = np.asarray(current_points_camera, dtype=np.float64)
    if previous.ndim != 2 or previous.shape[1:] != (3,) or current.shape != previous.shape:
        raise ValueError("matched camera points must be equal Nx3 arrays")
    if not np.isfinite(previous).all() or not np.isfinite(current).all():
        raise ValueError("matched camera points must be finite")
    world_from_previous = _validate_rigid_transform(
        transform_world_from_previous, "T_world_from_previous"
    )
    world_from_current = _validate_rigid_transform(
        transform_world_from_current, "T_world_from_current"
    )
    count = previous.shape[0]
    if count == 0:
        return TwoFrameTransformConsistency(
            int(previous_frame_index), int(current_frame_index), 0,
            None, None, None, None, None, str(coordinate_units), bool(is_metric),
        )

    previous_in_current = transform_points(
        previous,
        rotation_current_from_previous,
        translation_current_from_previous,
    )
    local_residuals = np.linalg.norm(previous_in_current - current, axis=1)
    previous_in_world = transform_points(
        previous, world_from_previous[:3, :3], world_from_previous[:3, 3]
    )
    current_in_world = transform_points(
        current, world_from_current[:3, :3], world_from_current[:3, 3]
    )
    world_residuals = np.linalg.norm(previous_in_world - current_in_world, axis=1)

    return TwoFrameTransformConsistency(
        previous_frame_index=int(previous_frame_index),
        current_frame_index=int(current_frame_index),
        correspondence_count=count,
        local_median_discrepancy=float(np.median(local_residuals)),
        local_rmse_discrepancy=float(np.sqrt(np.mean(local_residuals ** 2))),
        world_median_discrepancy=float(np.median(world_residuals)),
        world_rmse_discrepancy=float(np.sqrt(np.mean(world_residuals ** 2))),
        maximum_local_world_residual_difference=float(
            np.max(np.abs(local_residuals - world_residuals))
        ),
        coordinate_units=str(coordinate_units),
        is_metric=bool(is_metric),
    )


def save_transform_trace(
    path: Path,
    trace: tuple[TransformTraceEntry, ...] | list[TransformTraceEntry],
    pair_consistency: (
        tuple[TwoFrameTransformConsistency, ...]
        | list[TwoFrameTransformConsistency]
    ),
) -> Path:
    """Save the read-only audit trace as JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "canonical_notation": {
            "name": "T_A_from_B",
            "meaning": "maps a point expressed in B into coordinate system A",
            "column_equation": "p_A = R_A_from_B @ p_B + t_A_from_B",
            "numpy_row_equation": "points_A = points_B @ R_A_from_B.T + t_A_from_B",
        },
        "pose_composition": (
            "T_world_from_current = T_world_from_previous @ "
            "inverse(T_current_from_previous)"
        ),
        "diagnostic_only": True,
        "transform_trace": [entry.to_dict() for entry in trace],
        "two_frame_consistency": [entry.to_dict() for entry in pair_consistency],
    }
    with destination.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return destination
