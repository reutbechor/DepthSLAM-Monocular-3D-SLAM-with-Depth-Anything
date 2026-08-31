"""Sequential camera-to-world pose accumulation in arbitrary relative units."""

from __future__ import annotations

import numpy as np

from .transforms import invert_transform, make_transform


class PoseManager:
    """Accumulate camera-to-world poses relative to the first camera.

    OpenCV recoverPose returns R and t for the convention
    X_current = R @ X_previous + t. This is T_current_previous. Stored poses use
    the opposite camera-to-world convention T_world_camera, so each relative
    transform is inverted before right-composition with the previous pose.

    Preferred callers supply a translation whose magnitude was estimated in the
    same units as their camera Z. Fixed-step direction scaling is exposed only
    through an explicitly named debug method.
    """

    def __init__(self) -> None:
        self._poses: list[np.ndarray] = [np.eye(4, dtype=np.float64)]

    def add_scaled_relative_pose(
        self, rotation_previous_to_current: np.ndarray, translation_previous_to_current: np.ndarray
    ) -> np.ndarray:
        """Append T_world_current from a previous-to-current scaled R,t."""
        translation = np.asarray(translation_previous_to_current, dtype=np.float64)
        if translation.shape not in {(3,), (3, 1), (1, 3)}:
            raise ValueError("translation must contain exactly three values")
        translation = translation.reshape(3)
        if not np.isfinite(translation).all():
            raise ValueError("translation must be finite")
        transform_current_previous = make_transform(
            rotation_previous_to_current, translation
        )
        transform_previous_current = invert_transform(transform_current_previous)
        transform_world_current = self._poses[-1] @ transform_previous_current
        self._poses.append(transform_world_current)
        return transform_world_current.copy()

    def add_fixed_step_relative_pose(
        self,
        rotation_previous_to_current: np.ndarray,
        translation_direction: np.ndarray,
        relative_translation_step: float,
    ) -> np.ndarray:
        """Debug only: assign an arbitrary magnitude to a direction vector."""
        direction = np.asarray(translation_direction, dtype=np.float64).reshape(-1)
        step = float(relative_translation_step)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError("translation_direction must contain three finite values")
        norm = float(np.linalg.norm(direction))
        if norm == 0.0 or not np.isfinite(step) or step <= 0.0:
            raise ValueError("direction must be non-zero and fixed step positive")
        return self.add_scaled_relative_pose(
            rotation_previous_to_current, direction / norm * step
        )

    def current_pose(self) -> np.ndarray:
        """Return the latest camera-to-world transform T_world_camera."""
        return self._poses[-1].copy()

    def poses(self) -> list[np.ndarray]:
        """Return copies of all accepted camera-to-world transforms."""
        return [pose.copy() for pose in self._poses]

    def trajectory_positions(self) -> np.ndarray:
        """Return accepted camera centers in the first-camera world frame."""
        return np.asarray([pose[:3, 3] for pose in self._poses], dtype=np.float64)
