"""Rigid coordinate transformations for future relative map infrastructure."""

from __future__ import annotations

import numpy as np


def _validate_rotation(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6):
        raise ValueError("rotation must have determinant +1")
    return matrix


def _validate_translation(translation: np.ndarray) -> np.ndarray:
    vector = np.asarray(translation, dtype=np.float64)
    if vector.shape not in {(3,), (3, 1), (1, 3)}:
        raise ValueError("translation must have shape (3,), (3,1), or (1,3)")
    vector = vector.reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError("translation must contain only finite values")
    return vector


def transform_points(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply P' = R @ P + t to Nx3 points.

    Translation and point coordinates must already use compatible units. This
    function does not infer a scale for monocular translation directions.
    """
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1:] != (3,):
        raise ValueError("points must be an Nx3 array")
    if not np.isfinite(point_array).all():
        raise ValueError("points must contain only finite values")
    matrix = _validate_rotation(rotation)
    vector = _validate_translation(translation)
    return (matrix @ point_array.T).T + vector


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform; translation units must match future point units."""
    matrix = _validate_rotation(rotation)
    vector = _validate_translation(translation)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = matrix
    transform[:3, 3] = vector
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a validated 4x4 rigid homogeneous transform."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("transform must have final row [0, 0, 0, 1]")
    rotation = _validate_rotation(matrix[:3, :3])
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse
