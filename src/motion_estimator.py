"""Calibrated two-view motion estimation with Essential Matrix geometry."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionEstimateResult:
    """Relative pose result; translation is a direction with unknown scale."""

    success: bool
    message: str
    essential_matrix: np.ndarray | None
    rotation: np.ndarray | None
    translation_direction: np.ndarray | None
    inlier_mask: np.ndarray
    num_inliers: int
    inlier_ratio: float


class MotionEstimator:
    """Estimate relative calibrated camera motion from matched image points."""

    def __init__(
        self,
        ransac_probability: float = 0.999,
        ransac_threshold_pixels: float = 1.0,
        minimum_correspondences: int = 8,
        minimum_inliers: int = 8,
        minimum_inlier_ratio: float = 0.25,
    ) -> None:
        if not 0.0 < ransac_probability < 1.0:
            raise ValueError("ransac_probability must be between 0 and 1")
        if ransac_threshold_pixels <= 0:
            raise ValueError("ransac_threshold_pixels must be positive")
        if minimum_correspondences < 5:
            raise ValueError("minimum_correspondences must be at least 5")
        if minimum_inliers < 5:
            raise ValueError("minimum_inliers must be at least 5")
        if not 0.0 <= minimum_inlier_ratio <= 1.0:
            raise ValueError("minimum_inlier_ratio must be between 0 and 1")
        self.ransac_probability = ransac_probability
        self.ransac_threshold_pixels = ransac_threshold_pixels
        self.minimum_correspondences = minimum_correspondences
        self.minimum_inliers = minimum_inliers
        self.minimum_inlier_ratio = minimum_inlier_ratio

    @staticmethod
    def _validate_points(points1: np.ndarray, points2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first = np.asarray(points1, dtype=np.float64)
        second = np.asarray(points2, dtype=np.float64)
        if first.ndim != 2 or first.shape[1:] != (2,):
            raise ValueError("points1 must be an Nx2 array")
        if second.ndim != 2 or second.shape[1:] != (2,):
            raise ValueError("points2 must be an Nx2 array")
        if first.shape[0] != second.shape[0]:
            raise ValueError("points1 and points2 must contain the same number of points")
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError("matched points must contain only finite values")
        return first, second

    @staticmethod
    def _validate_intrinsics(camera_matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(camera_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("camera intrinsics K must be a finite 3x3 matrix")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or matrix[2, 2] == 0:
            raise ValueError("camera intrinsics must contain positive fx and fy")
        return matrix

    @staticmethod
    def _failure(message: str, correspondence_count: int) -> MotionEstimateResult:
        return MotionEstimateResult(
            success=False,
            message=message,
            essential_matrix=None,
            rotation=None,
            translation_direction=None,
            inlier_mask=np.zeros(correspondence_count, dtype=bool),
            num_inliers=0,
            inlier_ratio=0.0,
        )

    def estimate(
        self, points1: np.ndarray, points2: np.ndarray, camera_matrix: np.ndarray
    ) -> MotionEstimateResult:
        """Estimate R and unit-length t direction; monocular t scale is unknown."""
        first, second = self._validate_points(points1, points2)
        intrinsics = self._validate_intrinsics(camera_matrix)
        count = first.shape[0]
        if count < self.minimum_correspondences:
            return self._failure(
                f"Only {count} correspondences were supplied; "
                f"at least {self.minimum_correspondences} are required",
                count,
            )

        try:
            essential, ransac_mask = cv2.findEssentialMat(
                first,
                second,
                intrinsics,
                method=cv2.RANSAC,
                prob=self.ransac_probability,
                threshold=self.ransac_threshold_pixels,
            )
        except cv2.error as exc:
            return self._failure(f"Essential Matrix estimation failed: {exc}", count)
        if essential is None or ransac_mask is None:
            return self._failure("Essential Matrix estimation found no valid model", count)
        if essential.ndim != 2 or essential.shape[1] != 3 or essential.shape[0] % 3 != 0:
            return self._failure("OpenCV returned an invalid Essential Matrix shape", count)

        best: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        for start in range(0, essential.shape[0], 3):
            candidate = essential[start : start + 3]
            try:
                _, rotation, translation, pose_mask = cv2.recoverPose(
                    candidate, first, second, intrinsics, mask=ransac_mask.copy()
                )
            except cv2.error:
                continue
            inlier_mask = pose_mask.reshape(-1).astype(bool)
            num_inliers = int(np.count_nonzero(inlier_mask))
            if best is None or num_inliers > best[0]:
                best = (num_inliers, candidate, rotation, translation, inlier_mask)

        if best is None:
            return self._failure("Pose recovery failed for the Essential Matrix", count)

        num_inliers, essential, rotation, translation, inlier_mask = best
        translation_norm = float(np.linalg.norm(translation))
        if translation_norm == 0.0:
            return self._failure("Pose recovery returned a zero translation direction", count)
        translation_direction = translation / translation_norm
        inlier_ratio = num_inliers / count
        success = (
            num_inliers >= self.minimum_inliers
            and inlier_ratio >= self.minimum_inlier_ratio
        )
        message = (
            "Motion estimation successful"
            if success
            else (
                f"Pose rejected: {num_inliers} inliers ({inlier_ratio:.1%}); "
                f"requires at least {self.minimum_inliers} inliers and "
                f"{self.minimum_inlier_ratio:.1%} inlier ratio"
            )
        )
        return MotionEstimateResult(
            success=success,
            message=message,
            essential_matrix=essential.astype(np.float64),
            rotation=rotation.astype(np.float64),
            translation_direction=translation_direction.astype(np.float64),
            inlier_mask=inlier_mask,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
        )
