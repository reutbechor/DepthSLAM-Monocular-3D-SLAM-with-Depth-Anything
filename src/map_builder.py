"""Sequential relative map construction from already selected RGB frames."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from .backprojection import validate_camera_matrix
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
    pose_inliers: int = 0
    pose_inlier_ratio: float = 0.0
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


class RelativeMapBuilder:
    """Build a drift-prone, arbitrary-scale relative map from selected frames."""

    def __init__(
        self,
        depth_estimator: Any,
        feature_tracker: Any,
        motion_estimator: Any,
        camera_matrix: np.ndarray,
        translation_step: float = 1.0,
        point_cloud_stride: int = 6,
        voxel_size: float = 0.05,
        point_cloud_generator: Callable[..., PointCloudResult] = generate_colored_point_cloud,
    ) -> None:
        self.depth_estimator = depth_estimator
        self.feature_tracker = feature_tracker
        self.motion_estimator = motion_estimator
        self.camera_matrix = validate_camera_matrix(camera_matrix)
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

    def _camera_cloud(self, image_bgr: np.ndarray) -> PointCloudResult:
        depth = self.depth_estimator.predict(image_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        cloud = self.point_cloud_generator(
            image_rgb,
            depth,
            self.camera_matrix,
            stride=self.point_cloud_stride,
        )
        if cloud.valid_point_count == 0:
            raise RuntimeError("relative depth produced no valid point-cloud samples")
        return cloud

    @staticmethod
    def _validate_frame(frame: MappingFrame) -> None:
        if not isinstance(frame.image, np.ndarray) or frame.image.size == 0:
            raise ValueError(f"frame {frame.frame_index} has an empty image")

    def build(self, frames: Iterable[MappingFrame]) -> RelativeMapResult:
        """Process selected frames; rejected frames receive no pose or cloud."""
        iterator = iter(frames)
        try:
            first = next(iterator)
        except StopIteration as exc:
            raise ValueError("at least one mapping frame is required") from exc
        self._validate_frame(first)

        pose_manager = PoseManager(self.translation_step)
        fusion = RelativeMapFusion(self.voxel_size)
        statistics: list[FrameMapStatistics] = []
        trajectory_indices = [first.frame_index]
        sampled_count = 1

        try:
            first_cloud = self._camera_cloud(first.image)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f"first frame could not initialize the map: {exc}") from exc
        fusion.add(first_cloud.points, first_cloud.colors)
        statistics.append(
            FrameMapStatistics(
                frame_index=first.frame_index,
                timestamp_seconds=first.timestamp_seconds,
                accepted=True,
                reason="world_origin",
                cloud_points=first_cloud.valid_point_count,
                camera_position=(0.0, 0.0, 0.0),
            )
        )
        previous_accepted = first

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

            try:
                pose = self.motion_estimator.estimate(
                    matches.points1, matches.points2, self.camera_matrix
                )
            except (RuntimeError, ValueError) as exc:
                statistics.append(FrameMapStatistics(
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    accepted=False,
                    reason=f"motion_estimation_failed: {exc}",
                    good_matches=matches.statistics.good_matches,
                ))
                continue
            if not pose.success:
                statistics.append(FrameMapStatistics(
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    accepted=False,
                    reason=f"motion_rejected: {pose.message}",
                    good_matches=matches.statistics.good_matches,
                    pose_inliers=pose.num_inliers,
                    pose_inlier_ratio=pose.inlier_ratio,
                ))
                continue

            try:
                camera_cloud = self._camera_cloud(frame.image)
            except (RuntimeError, ValueError) as exc:
                statistics.append(FrameMapStatistics(
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    accepted=False,
                    reason=f"depth_or_cloud_failed: {exc}",
                    good_matches=matches.statistics.good_matches,
                    pose_inliers=pose.num_inliers,
                    pose_inlier_ratio=pose.inlier_ratio,
                ))
                continue

            world_pose = pose_manager.add_relative_pose(
                pose.rotation, pose.translation_direction
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
                reason="accepted",
                good_matches=matches.statistics.good_matches,
                pose_inliers=pose.num_inliers,
                pose_inlier_ratio=pose.inlier_ratio,
                cloud_points=camera_cloud.valid_point_count,
                camera_position=position,
            ))
            previous_accepted = frame

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
        )
