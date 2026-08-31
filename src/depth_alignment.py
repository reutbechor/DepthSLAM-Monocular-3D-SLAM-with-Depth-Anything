"""Inter-frame alignment for affine-ambiguous relative disparity predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .depth_geometry import DepthGeometryProcessor
from .depth_pose_estimator import DepthPoseEstimateResult
from .depth_types import CameraDepth, DepthPrediction


@dataclass(frozen=True)
class DepthAlignmentResult:
    success: bool
    message: str
    camera_depth: CameraDepth | None
    method: str
    input_correspondences: int
    inliers: int
    disparity_scale: float | None
    disparity_shift: float | None
    residual_rmse: float | None


def align_prediction_to_pose(
    prediction: DepthPrediction,
    points_current: np.ndarray,
    pose: DepthPoseEstimateResult,
    *,
    minimum_correspondences: int = 6,
) -> DepthAlignmentResult:
    """Fit d=a/Z+b at PnP inliers and convert a relative prediction to Z."""
    if not pose.success or pose.rotation is None or pose.translation is None:
        return DepthAlignmentResult(
            False, "a successful PnP pose is required", None, "none", 0, 0,
            None, None, None,
        )
    if prediction.is_metric:
        depth = prediction.to_camera_depth(alignment_method="metric_model")
        return DepthAlignmentResult(
            True, "metric model needs no inter-frame disparity alignment", depth,
            "metric_model", pose.pnp_inliers, pose.pnp_inliers, None, None, 0.0,
        )

    current = np.asarray(points_current, dtype=np.float64)
    if current.ndim != 2 or current.shape[1:] != (2,):
        raise ValueError("points_current must be an Nx2 array")
    local_mask = pose.pnp_inlier_correspondence_mask
    objects = pose.object_points_previous[local_mask]
    match_indices = pose.correspondence_match_indices[local_mask]
    if objects.shape[0] < minimum_correspondences:
        return DepthAlignmentResult(
            False, "insufficient PnP inliers for disparity alignment", None,
            "scale_and_shift", objects.shape[0], 0, None, None, None,
        )

    points_in_current = (
        pose.rotation @ objects.T
    ).T + pose.translation.reshape(1, 3)
    geometric_z = points_in_current[:, 2]
    raw_samples, raw_valid = DepthGeometryProcessor._sample_bilinear(
        np.asarray(prediction.values, dtype=np.float64), current[match_indices]
    )
    valid = raw_valid & np.isfinite(geometric_z) & (geometric_z > 1e-9)
    x = 1.0 / geometric_z[valid]
    y = raw_samples[valid]
    count = x.shape[0]
    if count < minimum_correspondences:
        return DepthAlignmentResult(
            False, f"Only {count} valid two-frame depth correspondences", None,
            "scale_and_shift", count, 0, None, None, None,
        )
    if np.ptp(x) <= 1e-9 * max(1.0, float(np.max(np.abs(x)))):
        return DepthAlignmentResult(
            False, "geometric depths do not span enough range for affine alignment",
            None, "scale_and_shift", count, 0, None, None, None,
        )

    inliers = np.ones(count, dtype=bool)
    coefficients = np.zeros(2, dtype=np.float64)
    for _ in range(8):
        design = np.column_stack((x[inliers], np.ones(np.count_nonzero(inliers))))
        coefficients, _, rank, _ = np.linalg.lstsq(design, y[inliers], rcond=None)
        if rank < 2:
            break
        residuals = y - (coefficients[0] * x + coefficients[1])
        center = float(np.median(residuals[inliers]))
        mad = float(np.median(np.abs(residuals[inliers] - center)))
        threshold = max(
            3.5 * 1.4826 * mad,
            1e-4 * max(1.0, float(np.ptp(y))),
        )
        updated = np.abs(residuals - center) <= threshold
        if np.count_nonzero(updated) < minimum_correspondences:
            break
        if np.array_equal(updated, inliers):
            inliers = updated
            break
        inliers = updated

    inlier_count = int(np.count_nonzero(inliers))
    scale, shift = (float(coefficients[0]), float(coefficients[1]))
    if inlier_count < minimum_correspondences or not np.isfinite([scale, shift]).all():
        return DepthAlignmentResult(
            False, "robust affine disparity fit was underconstrained", None,
            "scale_and_shift", count, inlier_count, None, None, None,
        )
    if scale <= 0.0:
        return DepthAlignmentResult(
            False, "affine disparity fit has non-positive inverse-depth scale", None,
            "scale_and_shift", count, inlier_count, scale, shift, None,
        )
    residuals = y[inliers] - (scale * x[inliers] + shift)
    try:
        camera_depth = prediction.to_camera_depth(
            disparity_scale=scale,
            disparity_shift=shift,
            alignment_method="scale_and_shift",
        )
    except ValueError as exc:
        return DepthAlignmentResult(
            False, f"aligned disparity conversion failed: {exc}", None,
            "scale_and_shift", count, inlier_count, scale, shift, None,
        )
    return DepthAlignmentResult(
        True, "relative disparity aligned with robust affine scale and shift",
        camera_depth, "scale_and_shift", count, inlier_count, scale, shift,
        float(np.sqrt(np.mean(residuals ** 2))),
    )
