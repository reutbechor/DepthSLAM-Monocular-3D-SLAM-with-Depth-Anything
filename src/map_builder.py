"""Sequential relative map construction from already selected RGB frames."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from .backprojection import validate_camera_matrix
from .depth_alignment import align_prediction_to_pose
from .depth_pose_estimator import DepthPoseEstimator
from .depth_types import CameraDepth, DepthPrediction
from .feature_tracker import FeatureTrackingError
from .map_fusion import FusedPointCloud, RelativeMapFusion
from .point_cloud import PointCloudResult, generate_colored_point_cloud
from .pose_manager import PoseManager
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
    good_matches: int = 0
    geometric_inliers: int = 0
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
    cloud_points: int = 0
    camera_position: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelativeMapResult:
    sampled_frame_count: int
    accepted_frame_count: int
    rejected_frame_count: int
    raw_fused_point_count: int
    fused_cloud: FusedPointCloud
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
        depth_pose_estimator: Any | None = None,
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
        self.point_cloud_generator = point_cloud_generator

    def _prediction(self, image_bgr: np.ndarray) -> DepthPrediction:
        prediction = self.depth_estimator.predict_result(image_bgr)
        if not isinstance(prediction, DepthPrediction):
            raise TypeError("depth estimator must return a DepthPrediction")
        return prediction

    def _camera_cloud(
        self, image_bgr: np.ndarray, camera_depth: CameraDepth
    ) -> PointCloudResult:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        cloud = self.point_cloud_generator(
            image_rgb,
            camera_depth,
            self.camera_matrix,
            stride=self.point_cloud_stride,
        )
        if cloud.valid_point_count == 0:
            raise RuntimeError("camera depth produced no valid point-cloud samples")
        return cloud

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
                alignment_method="metric_model" if first_prediction.is_metric else "none"
            )
            first_cloud = self._camera_cloud(first.image, first_depth)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"first frame could not initialize the map: {exc}") from exc
        fusion.add(first_cloud.points, first_cloud.colors)
        statistics.append(FrameMapStatistics(
            frame_index=first.frame_index,
            timestamp_seconds=first.timestamp_seconds,
            accepted=True,
            reason="world_origin",
            translation_magnitude=0.0,
            translation_units=first_depth.coordinate_units,
            scale_estimation_method="world_origin",
            depth_representation=first_prediction.representation,
            depth_alignment_method=first_depth.alignment_method,
            cloud_points=first_cloud.valid_point_count,
            camera_position=(0.0, 0.0, 0.0),
        ))
        previous_accepted = first
        previous_depth = first_depth

        for frame in iterator:
            sampled_count += 1
            self._validate_frame(frame)
            try:
                matches = self.feature_tracker.match(previous_accepted.image, frame.image)
            except (FeatureTrackingError, RuntimeError, ValueError) as exc:
                statistics.append(FrameMapStatistics(
                    frame.frame_index, frame.timestamp_seconds, False,
                    f"feature_matching_failed: {exc}"
                ))
                continue

            good_matches = int(matches.statistics.good_matches)
            try:
                geometry_pose = self.motion_estimator.estimate(
                    matches.points1, matches.points2, self.camera_matrix
                )
            except (RuntimeError, ValueError) as exc:
                statistics.append(FrameMapStatistics(
                    frame.frame_index, frame.timestamp_seconds, False,
                    f"geometric_motion_failed: {exc}", good_matches=good_matches
                ))
                continue
            if not geometry_pose.success:
                statistics.append(FrameMapStatistics(
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    accepted=False,
                    reason=f"geometric_motion_rejected: {geometry_pose.message}",
                    good_matches=good_matches,
                    geometric_inliers=geometry_pose.num_inliers,
                ))
                continue

            alignment_inliers = 0
            if self.scale_mode == "depth-pnp":
                try:
                    scaled_pose = self.depth_pose_estimator.estimate(
                        matches.points1,
                        matches.points2,
                        geometry_pose.inlier_mask,
                        previous_depth,
                        self.camera_matrix,
                    )
                except (RuntimeError, TypeError, ValueError) as exc:
                    statistics.append(FrameMapStatistics(
                        frame.frame_index,
                        frame.timestamp_seconds,
                        False,
                        f"depth_pnp_failed: {exc}",
                        good_matches=good_matches,
                        geometric_inliers=geometry_pose.num_inliers,
                        scale_estimation_method="depth_pnp",
                        depth_representation=first_prediction.representation,
                    ))
                    continue
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
                )
                if not scaled_pose.success:
                    statistics.append(FrameMapStatistics(
                        frame.frame_index, frame.timestamp_seconds, False,
                        f"depth_pnp_rejected: {scaled_pose.message}", **diagnostic
                    ))
                    continue
                try:
                    prediction = self._prediction(frame.image)
                    alignment = align_prediction_to_pose(
                        prediction, matches.points2, scaled_pose
                    )
                except (RuntimeError, TypeError, ValueError) as exc:
                    statistics.append(FrameMapStatistics(
                        frame.frame_index, frame.timestamp_seconds, False,
                        f"depth_alignment_failed: {exc}", **diagnostic
                    ))
                    continue
                if not alignment.success or alignment.camera_depth is None:
                    statistics.append(FrameMapStatistics(
                        frame.frame_index, frame.timestamp_seconds, False,
                        f"depth_alignment_rejected: {alignment.message}",
                        depth_alignment_method=alignment.method,
                        depth_alignment_inliers=alignment.inliers,
                        **diagnostic,
                    ))
                    continue
                camera_depth = alignment.camera_depth
                alignment_inliers = alignment.inliers
                rotation = scaled_pose.rotation
                translation = scaled_pose.translation
                assert rotation is not None and translation is not None
                scale_method = "depth_pnp"
                pose_diagnostic = diagnostic
            else:
                try:
                    prediction = self._prediction(frame.image)
                    camera_depth = prediction.to_camera_depth(alignment_method="none")
                except (RuntimeError, TypeError, ValueError) as exc:
                    statistics.append(FrameMapStatistics(
                        frame.frame_index, frame.timestamp_seconds, False,
                        f"depth_failed: {exc}", good_matches=good_matches,
                        geometric_inliers=geometry_pose.num_inliers,
                        scale_estimation_method="fixed_step_debug",
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
                )

            try:
                camera_cloud = self._camera_cloud(frame.image, camera_depth)
            except (RuntimeError, TypeError, ValueError) as exc:
                statistics.append(FrameMapStatistics(
                    frame.frame_index, frame.timestamp_seconds, False,
                    f"point_cloud_rejected: {exc}",
                    depth_alignment_method=camera_depth.alignment_method,
                    depth_alignment_inliers=alignment_inliers,
                    **pose_diagnostic,
                ))
                continue

            if scale_method == "depth_pnp":
                world_pose = pose_manager.add_scaled_relative_pose(rotation, translation)
            else:
                world_pose = pose_manager.add_fixed_step_relative_pose(
                    rotation, translation, self.translation_step
                )
            world_points = transform_points(
                camera_cloud.points, world_pose[:3, :3], world_pose[:3, 3]
            )
            fusion.add(world_points, camera_cloud.colors)
            position = tuple(float(value) for value in world_pose[:3, 3])
            trajectory_indices.append(frame.frame_index)
            statistics.append(FrameMapStatistics(
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                accepted=True,
                reason="accepted" if scale_method == "depth_pnp" else "accepted_debug_fixed_step",
                depth_alignment_method=camera_depth.alignment_method,
                depth_alignment_inliers=alignment_inliers,
                cloud_points=camera_cloud.valid_point_count,
                camera_position=position,
                **pose_diagnostic,
            ))
            previous_accepted = frame
            previous_depth = camera_depth

        fused = fusion.finalize()
        accepted_count = sum(item.accepted for item in statistics)
        height, width = first.image.shape[:2]
        return RelativeMapResult(
            sampled_frame_count=sampled_count,
            accepted_frame_count=accepted_count,
            rejected_frame_count=sampled_count - accepted_count,
            raw_fused_point_count=fusion.raw_point_count,
            fused_cloud=fused,
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
        )
