"""Sequential relative map construction from already selected RGB frames."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from .backprojection import validate_camera_matrix
from .depth_alignment import align_prediction_to_pose
from .depth_pose_estimator import DepthPoseEstimator
from .depth_quality import (
    DepthAlignmentQualityMetrics,
    DepthQualityThresholds,
    assess_depth_alignment_quality,
    measure_depth_alignment_quality,
)
from .depth_types import CameraDepth, DepthPrediction
from .feature_tracker import FeatureTrackingError
from .keyframe_selector import KeyframeSelectionResult, KeyframeSelector
from .map_fusion import FusedPointCloud, RelativeMapFusion
from .point_cloud import PointCloudResult, generate_colored_point_cloud
from .pose_manager import PoseManager
from .robust_filtering import GlobalOutlierFilterResult, filter_global_radius
from .transforms import transform_points


@dataclass(frozen=True)
class MappingFrame:
    image: np.ndarray
    frame_index: int
    timestamp_seconds: float = 0.0


@dataclass(frozen=True)
class FrameMapStatistics:
    frame_index: int
    timestamp_seconds: float
    accepted: bool
    reason: str
    status: str = "rejected"
    keyframe_reason: str | None = None
    skip_reason: str | None = None
    rejection_reason: str | None = None
    depth_inference_executed: bool = False
    good_matches: int = 0
    geometric_inliers: int = 0
    geometric_inlier_ratio: float = 0.0
    median_feature_displacement_px: float | None = None
    p75_feature_displacement_px: float | None = None
    p90_feature_displacement_px: float | None = None
    rotation_deg: float | None = None
    frames_since_last_keyframe: int = 0
    valid_depth_correspondences: int = 0
    pnp_inliers: int = 0
    pnp_inlier_ratio: float = 0.0
    reprojection_rmse_pixels: float | None = None
    reprojection_median_pixels: float | None = None
    translation_magnitude: float | None = None
    translation_units: str | None = None
    scale_estimation_method: str | None = None
    depth_representation: str | None = None
    depth_alignment_method: str | None = None
    depth_alignment_inliers: int = 0
    depth_alignment_input_correspondences: int = 0
    depth_alignment_inlier_ratio: float = 0.0
    disparity_scale: float | None = None
    disparity_shift: float | None = None
    denominator_epsilon: float | None = None
    minimum_absolute_denominator: float | None = None
    rejected_small_denominator_count: int = 0
    rejected_nonfinite_denominator_count: int = 0
    rejected_invalid_z_count: int = 0
    total_depth_candidates: int = 0
    valid_aligned_depth_count: int = 0
    denominator_rejection_ratio: float = 0.0
    valid_aligned_depth_ratio: float = 0.0
    aligned_z_p1: float | None = None
    aligned_z_median: float | None = None
    aligned_z_p99: float | None = None
    relative_z_p99_over_median: float | None = None
    z_statistics: dict[str, float] | None = None
    depth_filter_method: str | None = None
    depth_percentile_low: float | None = None
    depth_percentile_high: float | None = None
    depth_lower_bound: float | None = None
    depth_upper_bound: float | None = None
    depth_outlier_rejected_count: int = 0
    cloud_points: int = 0
    camera_position: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        valid_statuses = {"accepted_keyframe", "skipped_non_keyframe", "rejected"}
        if self.status not in valid_statuses:
            raise ValueError(f"status must be one of {sorted(valid_statuses)}")
        if self.accepted != (self.status == "accepted_keyframe"):
            raise ValueError("accepted must match accepted_keyframe status")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelativeMapResult:
    sampled_frame_count: int
    accepted_frame_count: int
    skipped_non_keyframe_count: int
    rejected_frame_count: int
    depth_inference_count: int
    raw_fused_point_count: int
    voxel_downsampled_point_count: int
    fused_cloud: FusedPointCloud
    global_filter: GlobalOutlierFilterResult
    trajectory_frame_indices: np.ndarray
    trajectory_positions: np.ndarray
    frame_statistics: list[FrameMapStatistics]
    image_width: int
    image_height: int
    depth_type: str
    depth_representation: str
    is_metric: bool
    scale_mode: str
    translation_units: str
    keyframes_enabled: bool
    stage_timings: dict[str, float]
    # Presentation assets from the already-computed first prediction.  Exposing
    # these avoids a second model inference when report previews are saved.
    preview_image_bgr: np.ndarray
    preview_depth_values: np.ndarray


class RelativeMapBuilder:
    """Build a drift-prone relative map with depth-scaled PnP by default."""

    def __init__(
        self,
        depth_estimator: Any,
        feature_tracker: Any,
        motion_estimator: Any,
        camera_matrix: np.ndarray,
        scale_mode: str = "depth-pnp",
        translation_step: float = 1.0,
        point_cloud_stride: int = 6,
        voxel_size: float = 0.05,
        disparity_denominator_epsilon: float = 1e-3,
        depth_percentile_low: float | None = 1.0,
        depth_percentile_high: float | None = 99.0,
        global_outlier_percentile: float | None = 99.5,
        min_valid_depth_ratio: float | None = 0.60,
        max_denominator_reject_ratio: float | None = 0.30,
        min_depth_alignment_inliers: int | None = 500,
        min_depth_alignment_inlier_ratio: float | None = 0.30,
        max_relative_z_p99_over_median: float | None = 50.0,
        keyframe_selector: KeyframeSelector | None = None,
        depth_pose_estimator: Any | None = None,
        depth_aligner: Callable[..., Any] = align_prediction_to_pose,
        point_cloud_generator: Callable[..., PointCloudResult] = generate_colored_point_cloud,
    ) -> None:
        if scale_mode not in {"depth-pnp", "fixed-step"}:
            raise ValueError("scale_mode must be 'depth-pnp' or 'fixed-step'")
        self.depth_estimator = depth_estimator
        self.feature_tracker = feature_tracker
        self.motion_estimator = motion_estimator
        self.depth_pose_estimator = depth_pose_estimator or DepthPoseEstimator()
        self.camera_matrix = validate_camera_matrix(camera_matrix)
        self.scale_mode = scale_mode
        self.translation_step = float(translation_step)
        if not np.isfinite(self.translation_step) or self.translation_step <= 0:
            raise ValueError("translation_step must be finite and positive")
        if isinstance(point_cloud_stride, bool) or not isinstance(
            point_cloud_stride, (int, np.integer)
        ) or point_cloud_stride < 1:
            raise ValueError("point_cloud_stride must be a positive integer")
        self.point_cloud_stride = int(point_cloud_stride)
        self.voxel_size = float(voxel_size)
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        self.disparity_denominator_epsilon = float(disparity_denominator_epsilon)
        if (
            not np.isfinite(self.disparity_denominator_epsilon)
            or self.disparity_denominator_epsilon <= 0.0
        ):
            raise ValueError("disparity_denominator_epsilon must be positive")
        self.depth_percentile_low = depth_percentile_low
        self.depth_percentile_high = depth_percentile_high
        self.global_outlier_percentile = global_outlier_percentile
        self.depth_quality_thresholds = DepthQualityThresholds(
            min_valid_depth_ratio=min_valid_depth_ratio,
            max_denominator_reject_ratio=max_denominator_reject_ratio,
            min_depth_alignment_inliers=min_depth_alignment_inliers,
            min_depth_alignment_inlier_ratio=min_depth_alignment_inlier_ratio,
            max_relative_z_p99_over_median=max_relative_z_p99_over_median,
        )
        self.keyframe_selector = keyframe_selector or KeyframeSelector()
        self.depth_aligner = depth_aligner
        self.point_cloud_generator = point_cloud_generator
        self._stage_timings: dict[str, float] = {}

    def _add_timing(self, name: str, seconds: float) -> None:
        self._stage_timings[name] = self._stage_timings.get(name, 0.0) + seconds

    def _prediction(self, image_bgr: np.ndarray) -> DepthPrediction:
        started = perf_counter()
        try:
            prediction = self.depth_estimator.predict_result(image_bgr)
        finally:
            self._add_timing("depth_inference_seconds", perf_counter() - started)
        if not isinstance(prediction, DepthPrediction):
            raise TypeError("depth estimator must return a DepthPrediction")
        return prediction

    def _camera_cloud(
        self, image_bgr: np.ndarray, camera_depth: CameraDepth
    ) -> PointCloudResult:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        started = perf_counter()
        try:
            cloud = self.point_cloud_generator(
                image_rgb,
                camera_depth,
                self.camera_matrix,
                stride=self.point_cloud_stride,
                depth_percentile_low=self.depth_percentile_low,
                depth_percentile_high=self.depth_percentile_high,
            )
        finally:
            self._add_timing(
                "point_cloud_fusion_seconds", perf_counter() - started
            )
        if cloud.valid_point_count == 0:
            raise RuntimeError("camera depth produced no valid point-cloud samples")
        return cloud

    @staticmethod
    def _depth_diagnostics(
        camera_depth: CameraDepth,
        cloud: PointCloudResult,
        quality: DepthAlignmentQualityMetrics,
    ) -> dict[str, Any]:
        return {
            **RelativeMapBuilder._quality_diagnostics(camera_depth, quality),
            "z_statistics": cloud.z_statistics,
            "depth_filter_method": cloud.depth_filter_method,
            "depth_percentile_low": cloud.depth_percentile_low,
            "depth_percentile_high": cloud.depth_percentile_high,
            "depth_lower_bound": cloud.depth_lower_bound,
            "depth_upper_bound": cloud.depth_upper_bound,
            "depth_outlier_rejected_count": cloud.depth_outlier_rejected_count,
        }

    @staticmethod
    def _quality_diagnostics(
        camera_depth: CameraDepth,
        quality: DepthAlignmentQualityMetrics,
    ) -> dict[str, Any]:
        return {
            "disparity_scale": camera_depth.disparity_scale,
            "disparity_shift": camera_depth.disparity_shift,
            "denominator_epsilon": camera_depth.denominator_epsilon,
            "minimum_absolute_denominator": camera_depth.minimum_absolute_denominator,
            "rejected_small_denominator_count": (
                camera_depth.rejected_small_denominator_count
            ),
            "rejected_nonfinite_denominator_count": (
                camera_depth.rejected_nonfinite_denominator_count
            ),
            "rejected_invalid_z_count": camera_depth.rejected_invalid_z_count,
            "total_depth_candidates": quality.total_depth_candidates,
            "valid_aligned_depth_count": quality.valid_aligned_depth_count,
            "denominator_rejection_ratio": quality.denominator_rejection_ratio,
            "valid_aligned_depth_ratio": quality.valid_aligned_depth_ratio,
            "depth_alignment_input_correspondences": (
                quality.alignment_input_correspondences
            ),
            "depth_alignment_inliers": quality.alignment_inliers,
            "depth_alignment_inlier_ratio": quality.alignment_inlier_ratio,
            "aligned_z_p1": quality.aligned_z_p1,
            "aligned_z_median": quality.aligned_z_median,
            "aligned_z_p99": quality.aligned_z_p99,
            "relative_z_p99_over_median": quality.relative_z_p99_over_median,
        }

    @staticmethod
    def _keyframe_diagnostics(
        selection: KeyframeSelectionResult,
    ) -> dict[str, Any]:
        metrics = selection.metrics
        if metrics is None:
            return {}
        return {
            "geometric_inlier_ratio": metrics.geometric_inlier_ratio,
            "median_feature_displacement_px": (
                metrics.median_feature_displacement_px
            ),
            "p75_feature_displacement_px": metrics.p75_feature_displacement_px,
            "p90_feature_displacement_px": metrics.p90_feature_displacement_px,
            "rotation_deg": metrics.rotation_deg,
            "frames_since_last_keyframe": metrics.frames_since_last_keyframe,
        }

    @staticmethod
    def _validate_frame(frame: MappingFrame) -> None:
        if (
            not isinstance(frame.image, np.ndarray)
            or frame.image.ndim != 3
            or frame.image.shape[2] != 3
            or frame.image.size == 0
        ):
            raise ValueError(f"frame {frame.frame_index} must be a non-empty BGR image")

    def build(self, frames: Iterable[MappingFrame]) -> RelativeMapResult:
        """Process selected frames; rejected frames receive no pose or cloud."""
        pipeline_started = perf_counter()
        self._stage_timings = {
            "feature_motion_seconds": 0.0,
            "depth_inference_seconds": 0.0,
            "pnp_depth_alignment_seconds": 0.0,
            "point_cloud_fusion_seconds": 0.0,
        }
        iterator = iter(frames)
        try:
            first = next(iterator)
        except StopIteration as exc:
            raise ValueError("at least one mapping frame is required") from exc
        self._validate_frame(first)

        pose_manager = PoseManager()
        fusion = RelativeMapFusion(self.voxel_size)
        statistics: list[FrameMapStatistics] = []
        trajectory_indices = [first.frame_index]
        sampled_count = 1

        try:
            first_prediction = self._prediction(first.image)
            first_depth = first_prediction.to_camera_depth(
                alignment_method="metric_model" if first_prediction.is_metric else "none",
                denominator_epsilon=self.disparity_denominator_epsilon,
            )
            first_cloud = self._camera_cloud(first.image, first_depth)
            first_quality = measure_depth_alignment_quality(
                first_depth,
                alignment_input_correspondences=0,
                alignment_inliers=0,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"first frame could not initialize the map: {exc}") from exc
        fusion_started = perf_counter()
        fusion.add(first_cloud.points, first_cloud.colors)
        self._add_timing(
            "point_cloud_fusion_seconds", perf_counter() - fusion_started
        )
        statistics.append(FrameMapStatistics(
            frame_index=first.frame_index,
            timestamp_seconds=first.timestamp_seconds,
            accepted=True,
            reason="world_origin",
            status="accepted_keyframe",
            keyframe_reason="initial_frame",
            depth_inference_executed=True,
            translation_magnitude=0.0,
            translation_units=first_depth.coordinate_units,
            scale_estimation_method="world_origin",
            depth_representation=first_prediction.representation,
            depth_alignment_method=first_depth.alignment_method,
            cloud_points=first_cloud.valid_point_count,
            camera_position=(0.0, 0.0, 0.0),
            **self._depth_diagnostics(first_depth, first_cloud, first_quality),
        ))
        previous_accepted = first
        previous_depth = first_depth
        frames_since_last_keyframe = 0

        for frame in iterator:
            sampled_count += 1
            frames_since_last_keyframe += 1
            self._validate_frame(frame)
            visual_started = perf_counter()
            try:
                matches = self.feature_tracker.match(previous_accepted.image, frame.image)
            except (FeatureTrackingError, RuntimeError, ValueError) as exc:
                self._add_timing(
                    "feature_motion_seconds", perf_counter() - visual_started
                )
                statistics.append(FrameMapStatistics(
                    frame.frame_index, frame.timestamp_seconds, False,
                    f"feature_matching_failed: {exc}",
                    status="rejected",
                    rejection_reason="feature_matching",
                    frames_since_last_keyframe=frames_since_last_keyframe,
                ))
                continue

            good_matches = int(matches.statistics.good_matches)
            try:
                geometry_pose = self.motion_estimator.estimate(
                    matches.points1, matches.points2, self.camera_matrix
                )
            except (RuntimeError, ValueError) as exc:
                self._add_timing(
                    "feature_motion_seconds", perf_counter() - visual_started
                )
                statistics.append(FrameMapStatistics(
                    frame.frame_index, frame.timestamp_seconds, False,
                    f"geometric_motion_failed: {exc}",
                    status="rejected",
                    rejection_reason="geometric_filtering",
                    good_matches=good_matches,
                    frames_since_last_keyframe=frames_since_last_keyframe,
                ))
                continue
            self._add_timing(
                "feature_motion_seconds", perf_counter() - visual_started
            )
            if not geometry_pose.success:
                statistics.append(FrameMapStatistics(
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    accepted=False,
                    reason=f"geometric_motion_rejected: {geometry_pose.message}",
                    status="rejected",
                    rejection_reason="geometric_filtering",
                    good_matches=good_matches,
                    geometric_inliers=geometry_pose.num_inliers,
                    geometric_inlier_ratio=geometry_pose.inlier_ratio,
                    frames_since_last_keyframe=frames_since_last_keyframe,
                ))
                continue

            assert geometry_pose.rotation is not None
            selection = self.keyframe_selector.evaluate(
                points1=matches.points1,
                points2=matches.points2,
                inlier_mask=geometry_pose.inlier_mask,
                rotation=geometry_pose.rotation,
                good_matches=good_matches,
                geometric_inliers=geometry_pose.num_inliers,
                geometric_inlier_ratio=geometry_pose.inlier_ratio,
                frames_since_last_keyframe=frames_since_last_keyframe,
            )
            keyframe_diagnostic = self._keyframe_diagnostics(selection)
            if selection.status == "rejected":
                statistics.append(FrameMapStatistics(
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    accepted=False,
                    reason=f"keyframe_geometry_rejected: {selection.reason}",
                    status="rejected",
                    rejection_reason=selection.reason,
                    good_matches=good_matches,
                    geometric_inliers=geometry_pose.num_inliers,
                    **keyframe_diagnostic,
                ))
                continue
            if selection.status == "skipped":
                statistics.append(FrameMapStatistics(
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    accepted=False,
                    reason=f"skipped_non_keyframe: {selection.reason}",
                    status="skipped_non_keyframe",
                    skip_reason=selection.reason,
                    good_matches=good_matches,
                    geometric_inliers=geometry_pose.num_inliers,
                    **keyframe_diagnostic,
                ))
                continue
            keyframe_reason = selection.reason

            alignment_inliers = 0
            if self.scale_mode == "depth-pnp":
                pose_started = perf_counter()
                try:
                    scaled_pose = self.depth_pose_estimator.estimate(
                        matches.points1,
                        matches.points2,
                        geometry_pose.inlier_mask,
                        previous_depth,
                        self.camera_matrix,
                    )
                except (RuntimeError, TypeError, ValueError) as exc:
                    self._add_timing(
                        "pnp_depth_alignment_seconds", perf_counter() - pose_started
                    )
                    statistics.append(FrameMapStatistics(
                        frame.frame_index,
                        frame.timestamp_seconds,
                        False,
                        f"depth_pnp_failed: {exc}",
                        status="rejected",
                        keyframe_reason=keyframe_reason,
                        rejection_reason="pnp",
                        good_matches=good_matches,
                        geometric_inliers=geometry_pose.num_inliers,
                        scale_estimation_method="depth_pnp",
                        depth_representation=first_prediction.representation,
                        **keyframe_diagnostic,
                    ))
                    continue
                self._add_timing(
                    "pnp_depth_alignment_seconds", perf_counter() - pose_started
                )
                diagnostic = dict(
                    good_matches=good_matches,
                    geometric_inliers=geometry_pose.num_inliers,
                    valid_depth_correspondences=scaled_pose.valid_depth_correspondences,
                    pnp_inliers=scaled_pose.pnp_inliers,
                    pnp_inlier_ratio=scaled_pose.pnp_inlier_ratio,
                    reprojection_rmse_pixels=scaled_pose.reprojection_rmse_pixels,
                    reprojection_median_pixels=scaled_pose.reprojection_median_pixels,
                    translation_magnitude=scaled_pose.translation_magnitude,
                    translation_units=scaled_pose.translation_units,
                    scale_estimation_method="depth_pnp",
                    depth_representation=first_prediction.representation,
                    **keyframe_diagnostic,
                )
                if not scaled_pose.success:
                    statistics.append(FrameMapStatistics(
                        frame.frame_index, frame.timestamp_seconds, False,
                        f"depth_pnp_rejected: {scaled_pose.message}",
                        status="rejected",
                        keyframe_reason=keyframe_reason,
                        rejection_reason="pnp",
                        **diagnostic,
                    ))
                    continue
                alignment_started: float | None = None
                try:
                    depth_inference_executed = True
                    prediction = self._prediction(frame.image)
                    alignment_started = perf_counter()
                    alignment = self.depth_aligner(
                        prediction,
                        matches.points2,
                        scaled_pose,
                        denominator_epsilon=self.disparity_denominator_epsilon,
                    )
                except (RuntimeError, TypeError, ValueError) as exc:
                    if alignment_started is not None:
                        self._add_timing(
                            "pnp_depth_alignment_seconds",
                            perf_counter() - alignment_started,
                        )
                    statistics.append(FrameMapStatistics(
                        frame.frame_index, frame.timestamp_seconds, False,
                        f"depth_alignment_failed: {exc}",
                        status="rejected",
                        keyframe_reason=keyframe_reason,
                        rejection_reason="depth_alignment",
                        depth_inference_executed=depth_inference_executed,
                        **diagnostic,
                    ))
                    continue
                self._add_timing(
                    "pnp_depth_alignment_seconds", perf_counter() - alignment_started
                )
                if not alignment.success or alignment.camera_depth is None:
                    statistics.append(FrameMapStatistics(
                        frame.frame_index, frame.timestamp_seconds, False,
                        f"depth_alignment_rejected: {alignment.message}",
                        status="rejected",
                        keyframe_reason=keyframe_reason,
                        rejection_reason="depth_alignment",
                        depth_inference_executed=True,
                        depth_alignment_method=alignment.method,
                        depth_alignment_inliers=alignment.inliers,
                        **diagnostic,
                    ))
                    continue
                camera_depth = alignment.camera_depth
                alignment_inliers = alignment.inliers
                quality_assessment = assess_depth_alignment_quality(
                    camera_depth,
                    alignment_input_correspondences=alignment.input_correspondences,
                    alignment_inliers=alignment.inliers,
                    thresholds=self.depth_quality_thresholds,
                )
                if not quality_assessment.accepted:
                    statistics.append(FrameMapStatistics(
                        frame.frame_index,
                        frame.timestamp_seconds,
                        False,
                        f"depth_quality_rejected: {quality_assessment.rejection_reason}",
                        status="rejected",
                        keyframe_reason=keyframe_reason,
                        rejection_reason=quality_assessment.rejection_reason,
                        depth_inference_executed=True,
                        depth_alignment_method=camera_depth.alignment_method,
                        **self._quality_diagnostics(
                            camera_depth, quality_assessment.metrics
                        ),
                        **diagnostic,
                    ))
                    continue
                depth_quality = quality_assessment.metrics
                rotation = scaled_pose.rotation
                translation = scaled_pose.translation
                assert rotation is not None and translation is not None
                scale_method = "depth_pnp"
                pose_diagnostic = diagnostic
            else:
                try:
                    depth_inference_executed = True
                    prediction = self._prediction(frame.image)
                    camera_depth = prediction.to_camera_depth(
                        alignment_method="none",
                        denominator_epsilon=self.disparity_denominator_epsilon,
                    )
                except (RuntimeError, TypeError, ValueError) as exc:
                    statistics.append(FrameMapStatistics(
                        frame.frame_index, frame.timestamp_seconds, False,
                        f"depth_failed: {exc}", good_matches=good_matches,
                        status="rejected",
                        keyframe_reason=keyframe_reason,
                        rejection_reason="depth_alignment",
                        depth_inference_executed=depth_inference_executed,
                        geometric_inliers=geometry_pose.num_inliers,
                        scale_estimation_method="fixed_step_debug",
                        **keyframe_diagnostic,
                    ))
                    continue
                rotation = geometry_pose.rotation
                translation = geometry_pose.translation_direction
                scale_method = "fixed_step_debug"
                pose_diagnostic = dict(
                    good_matches=good_matches,
                    geometric_inliers=geometry_pose.num_inliers,
                    translation_magnitude=self.translation_step,
                    translation_units="arbitrary_fixed_step_units",
                    scale_estimation_method=scale_method,
                    depth_representation=prediction.representation,
                    **keyframe_diagnostic,
                )
                depth_quality = measure_depth_alignment_quality(
                    camera_depth,
                    alignment_input_correspondences=0,
                    alignment_inliers=0,
                )

            try:
                camera_cloud = self._camera_cloud(frame.image, camera_depth)
            except (RuntimeError, TypeError, ValueError) as exc:
                statistics.append(FrameMapStatistics(
                    frame.frame_index, frame.timestamp_seconds, False,
                    f"point_cloud_rejected: {exc}",
                    status="rejected",
                    keyframe_reason=keyframe_reason,
                    rejection_reason="point_cloud",
                    depth_inference_executed=True,
                    depth_alignment_method=camera_depth.alignment_method,
                    **self._quality_diagnostics(camera_depth, depth_quality),
                    **pose_diagnostic,
                ))
                continue

            if scale_method == "depth_pnp":
                world_pose = pose_manager.add_scaled_relative_pose(rotation, translation)
            else:
                world_pose = pose_manager.add_fixed_step_relative_pose(
                    rotation, translation, self.translation_step
                )
            fusion_started = perf_counter()
            world_points = transform_points(
                camera_cloud.points, world_pose[:3, :3], world_pose[:3, 3]
            )
            fusion.add(world_points, camera_cloud.colors)
            self._add_timing(
                "point_cloud_fusion_seconds", perf_counter() - fusion_started
            )
            position = tuple(float(value) for value in world_pose[:3, 3])
            trajectory_indices.append(frame.frame_index)
            statistics.append(FrameMapStatistics(
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                accepted=True,
                reason="accepted" if scale_method == "depth_pnp" else "accepted_debug_fixed_step",
                status="accepted_keyframe",
                keyframe_reason=keyframe_reason,
                depth_inference_executed=True,
                depth_alignment_method=camera_depth.alignment_method,
                cloud_points=camera_cloud.valid_point_count,
                camera_position=position,
                **self._depth_diagnostics(
                    camera_depth, camera_cloud, depth_quality
                ),
                **pose_diagnostic,
            ))
            previous_accepted = frame
            previous_depth = camera_depth
            frames_since_last_keyframe = 0

        fusion_started = perf_counter()
        voxelized = fusion.finalize()
        global_filter = filter_global_radius(
            voxelized.points,
            voxelized.colors,
            self.global_outlier_percentile,
        )
        self._add_timing(
            "point_cloud_fusion_seconds", perf_counter() - fusion_started
        )
        fused = FusedPointCloud(
            points=global_filter.points,
            colors=global_filter.colors,
            input_point_count=fusion.raw_point_count,
            output_point_count=global_filter.output_count,
            voxel_size=self.voxel_size,
            coordinate_units=(
                first_depth.coordinate_units
                if self.scale_mode == "depth-pnp"
                else "arbitrary_fixed_step_units"
            ),
        )
        accepted_count = sum(item.accepted for item in statistics)
        skipped_count = sum(
            item.status == "skipped_non_keyframe" for item in statistics
        )
        rejected_count = sum(item.status == "rejected" for item in statistics)
        depth_inference_count = sum(
            item.depth_inference_executed for item in statistics
        )
        height, width = first.image.shape[:2]
        self._stage_timings["mapping_pipeline_seconds"] = (
            perf_counter() - pipeline_started
        )
        return RelativeMapResult(
            sampled_frame_count=sampled_count,
            accepted_frame_count=accepted_count,
            skipped_non_keyframe_count=skipped_count,
            rejected_frame_count=rejected_count,
            depth_inference_count=depth_inference_count,
            raw_fused_point_count=fusion.raw_point_count,
            voxel_downsampled_point_count=voxelized.output_point_count,
            fused_cloud=fused,
            global_filter=global_filter,
            trajectory_frame_indices=np.asarray(trajectory_indices, dtype=np.int64),
            trajectory_positions=pose_manager.trajectory_positions(),
            frame_statistics=statistics,
            image_width=width,
            image_height=height,
            depth_type=first_prediction.depth_type,
            depth_representation=first_prediction.representation,
            is_metric=first_prediction.is_metric,
            scale_mode=self.scale_mode,
            translation_units=(
                first_depth.coordinate_units
                if self.scale_mode == "depth-pnp"
                else "arbitrary_fixed_step_units"
            ),
            keyframes_enabled=self.keyframe_selector.thresholds.enabled,
            stage_timings=dict(self._stage_timings),
            preview_image_bgr=first.image.copy(),
            preview_depth_values=first_prediction.values.copy(),
        )
