"""Depth-assisted relative pose estimation with translation magnitude."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .backprojection import validate_camera_matrix
from .depth_geometry import DepthGeometryProcessor
from .depth_types import CameraDepth


@dataclass(frozen=True)
class DepthPoseEstimateResult:
    success: bool
    message: str
    rotation: np.ndarray | None
    translation: np.ndarray | None
    inlier_mask: np.ndarray
    geometric_inlier_count: int
    valid_depth_correspondences: int
    pnp_inliers: int
    pnp_inlier_ratio: float
    reprojection_rmse_pixels: float | None
    reprojection_median_pixels: float | None
    translation_magnitude: float | None
    translation_units: str
    correspondence_match_indices: np.ndarray
    object_points_previous: np.ndarray
    image_points_current: np.ndarray
    pnp_inlier_correspondence_mask: np.ndarray


class DepthPoseEstimator:
    """Recover previous-to-current R,t using previous-frame camera Z and PnP."""

    def __init__(
        self,
        sampling_method: str = "bilinear",
        minimum_correspondences: int = 6,
        minimum_inliers: int = 6,
        minimum_inlier_ratio: float = 0.25,
        reprojection_error_pixels: float = 3.0,
        confidence: float = 0.999,
        iterations: int = 200,
    ) -> None:
        if minimum_correspondences < 4 or minimum_inliers < 4:
            raise ValueError("PnP correspondence and inlier minima must be at least 4")
        if not 0.0 <= minimum_inlier_ratio <= 1.0:
            raise ValueError("minimum_inlier_ratio must be between 0 and 1")
        if reprojection_error_pixels <= 0.0:
            raise ValueError("reprojection_error_pixels must be positive")
        if not 0.0 < confidence < 1.0 or iterations < 1:
            raise ValueError("confidence and iterations must be valid")
        self.geometry = DepthGeometryProcessor(sampling_method)
        self.minimum_correspondences = minimum_correspondences
        self.minimum_inliers = minimum_inliers
        self.minimum_inlier_ratio = minimum_inlier_ratio
        self.reprojection_error_pixels = reprojection_error_pixels
        self.confidence = confidence
        self.iterations = iterations

    @staticmethod
    def _failure(
        message: str,
        match_count: int,
        geometric_count: int,
        units: str,
        geometry=None,
    ) -> DepthPoseEstimateResult:
        indices = (
            geometry.valid_match_indices
            if geometry is not None
            else np.empty(0, dtype=np.int64)
        )
        objects = (
            geometry.points_3d_relative
            if geometry is not None
            else np.empty((0, 3), dtype=np.float64)
        )
        return DepthPoseEstimateResult(
            success=False,
            message=message,
            rotation=None,
            translation=None,
            inlier_mask=np.zeros(match_count, dtype=bool),
            geometric_inlier_count=geometric_count,
            valid_depth_correspondences=objects.shape[0],
            pnp_inliers=0,
            pnp_inlier_ratio=0.0,
            reprojection_rmse_pixels=None,
            reprojection_median_pixels=None,
            translation_magnitude=None,
            translation_units=units,
            correspondence_match_indices=indices,
            object_points_previous=objects,
            image_points_current=np.empty((0, 2), dtype=np.float64),
            pnp_inlier_correspondence_mask=np.zeros(objects.shape[0], dtype=bool),
        )

    def estimate(
        self,
        points_previous: np.ndarray,
        points_current: np.ndarray,
        geometric_inlier_mask: np.ndarray,
        previous_camera_depth: CameraDepth,
        camera_matrix: np.ndarray,
    ) -> DepthPoseEstimateResult:
        previous = np.asarray(points_previous, dtype=np.float64)
        current = np.asarray(points_current, dtype=np.float64)
        if previous.ndim != 2 or previous.shape[1:] != (2,):
            raise ValueError("points_previous must be an Nx2 array")
        if current.shape != previous.shape or not np.isfinite(current).all():
            raise ValueError("points_current must be a finite Nx2 array of equal length")
        mask = np.asarray(geometric_inlier_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != previous.shape[0]:
            raise ValueError("geometric_inlier_mask length must equal match count")
        intrinsics = validate_camera_matrix(camera_matrix)
        geometry = self.geometry.process(
            previous, mask, previous_camera_depth, intrinsics
        )
        count = geometry.valid_depth_sample_count
        geometric_count = int(np.count_nonzero(mask))
        units = previous_camera_depth.coordinate_units
        if count < self.minimum_correspondences:
            return self._failure(
                f"Only {count} valid depth correspondences; "
                f"at least {self.minimum_correspondences} are required",
                previous.shape[0], geometric_count, units, geometry,
            )

        object_points = geometry.points_3d_relative.astype(np.float64, copy=False)
        image_points = current[geometry.valid_match_indices]
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                intrinsics,
                None,
                iterationsCount=self.iterations,
                reprojectionError=self.reprojection_error_pixels,
                confidence=self.confidence,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error as exc:
            return self._failure(
                f"solvePnPRansac failed: {exc}", previous.shape[0],
                geometric_count, units, geometry,
            )
        if not ok or rvec is None or tvec is None or inliers is None:
            return self._failure(
                "solvePnPRansac did not find a pose", previous.shape[0],
                geometric_count, units, geometry,
            )

        local_inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)
        pnp_count = local_inliers.shape[0]
        ratio = pnp_count / count
        if pnp_count < self.minimum_inliers or ratio < self.minimum_inlier_ratio:
            failed = self._failure(
                f"PnP pose rejected: {pnp_count}/{count} inliers "
                f"({ratio:.3f})", previous.shape[0], geometric_count, units, geometry,
            )
            return DepthPoseEstimateResult(
                **{**failed.__dict__, "pnp_inliers": pnp_count,
                   "pnp_inlier_ratio": ratio,
                   "image_points_current": image_points}
            )

        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[local_inliers], image_points[local_inliers],
                intrinsics, None, rvec, tvec,
            )
        except cv2.error:
            pass
        rotation, _ = cv2.Rodrigues(rvec)
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)
        projected, _ = cv2.projectPoints(
            object_points[local_inliers], rvec, translation, intrinsics, None
        )
        errors = np.linalg.norm(
            projected.reshape(-1, 2) - image_points[local_inliers], axis=1
        )
        if not (
            np.isfinite(rotation).all()
            and np.isfinite(translation).all()
            and np.isfinite(errors).all()
        ):
            return self._failure(
                "PnP produced non-finite pose or residuals", previous.shape[0],
                geometric_count, units, geometry,
            )

        full_mask = np.zeros(previous.shape[0], dtype=bool)
        full_mask[geometry.valid_match_indices[local_inliers]] = True
        correspondence_mask = np.zeros(count, dtype=bool)
        correspondence_mask[local_inliers] = True
        magnitude = float(np.linalg.norm(translation))
        return DepthPoseEstimateResult(
            success=True,
            message="depth-assisted PnP pose accepted",
            rotation=rotation,
            translation=translation,
            inlier_mask=full_mask,
            geometric_inlier_count=geometric_count,
            valid_depth_correspondences=count,
            pnp_inliers=pnp_count,
            pnp_inlier_ratio=ratio,
            reprojection_rmse_pixels=float(np.sqrt(np.mean(errors ** 2))),
            reprojection_median_pixels=float(np.median(errors)),
            translation_magnitude=magnitude,
            translation_units=units,
            correspondence_match_indices=geometry.valid_match_indices,
            object_points_previous=object_points,
            image_points_current=image_points,
            pnp_inlier_correspondence_mask=correspondence_mask,
        )
