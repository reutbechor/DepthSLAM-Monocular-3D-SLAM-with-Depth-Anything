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

    Translation magnitude uses an arbitrary configured relative step, not metres.
    """

    def __init__(self, relative_translation_step: float = 1.0) -> None:
        step = float(relative_translation_step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("relative_translation_step must be finite and positive")
        self.relative_translation_step = step
        self._poses: list[np.ndarray] = [np.eye(4, dtype=np.float64)]

    def add_relative_pose(
        self, rotation_previous_to_current: np.ndarray, translation_direction: np.ndarray
    ) -> np.ndarray:
        """Append and return T_world_current using an unknown-scale t direction."""
        direction = np.asarray(translation_direction, dtype=np.float64)
        if direction.shape not in {(3,), (3, 1), (1, 3)}:
            raise ValueError("translation_direction must contain exactly three values")
        direction = direction.reshape(3)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction).all() or norm == 0.0:
            raise ValueError("translation_direction must be finite and non-zero")
        relative_translation = direction / norm * self.relative_translation_step

        transform_current_previous = make_transform(
            rotation_previous_to_current, relative_translation
        )
        transform_previous_current = invert_transform(transform_current_previous)
        transform_world_current = self._poses[-1] @ transform_previous_current
        self._poses.append(transform_world_current)
        return transform_world_current.copy()

    def current_pose(self) -> np.ndarray:
        """Return the latest camera-to-world transform T_world_camera."""
        return self._poses[-1].copy()

    def poses(self) -> list[np.ndarray]:
        """Return copies of all accepted camera-to-world transforms."""
        return [pose.copy() for pose in self._poses]

    def trajectory_positions(self) -> np.ndarray:
        """Return accepted camera centers in the first-camera world frame."""
        return np.asarray([pose[:3, 3] for pose in self._poses], dtype=np.float64)
