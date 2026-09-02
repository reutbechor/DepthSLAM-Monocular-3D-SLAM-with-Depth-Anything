import unittest
from types import SimpleNamespace

import numpy as np

from src.depth_alignment import DepthAlignmentResult
from src.depth_stabilization import DepthStabilizationConfig
from src.temporal_depth_normalization import TemporalDepthNormalizationConfig
from src.sliding_window_pose_optimization import (
    SlidingWindowPoseOptimizationConfig,
)
from src.map_builder import MappingFrame, RelativeMapBuilder
from src.keyframe_selector import KeyframeSelector, KeyframeThresholds
from src.depth_types import CameraDepth, DepthPrediction


class FakeDepthEstimator:
    def __init__(self) -> None:
        self.calls = 0

    def predict_result(self, image: np.ndarray) -> DepthPrediction:
        self.calls += 1
        return DepthPrediction(
            np.ones(image.shape[:2], dtype=np.float32),
            "relative", False, "relative_inverse_depth", "synthetic",
        )


class FakeFeatureTracker:
    def match(self, first: np.ndarray, second: np.ndarray) -> SimpleNamespace:
        del first, second
        points = np.zeros((8, 2), dtype=np.float64)
        return SimpleNamespace(
            points1=points,
            points2=points.copy(),
            statistics=SimpleNamespace(good_matches=8),
        )


class RejectingMotionEstimator:
    def estimate(self, points1, points2, camera_matrix) -> SimpleNamespace:
        del points1, points2, camera_matrix
        return SimpleNamespace(
            success=False,
            message="synthetic rejection",
            num_inliers=0,
            inlier_ratio=0.0,
        )


class SuccessfulGeometryEstimator:
    def estimate(self, points1, points2, camera_matrix) -> SimpleNamespace:
        del points2, camera_matrix
        return SimpleNamespace(
            success=True,
            message="accepted geometry",
            rotation=np.eye(3),
            translation_direction=np.array([1.0, 0.0, 0.0]),
            inlier_mask=np.ones(points1.shape[0], dtype=bool),
            num_inliers=points1.shape[0],
            inlier_ratio=1.0,
        )


class RejectingDepthPoseEstimator:
    def estimate(self, *args) -> SimpleNamespace:
        del args
        return SimpleNamespace(
            success=False,
            message="synthetic scaled-pose rejection",
            valid_depth_correspondences=8,
            pnp_inliers=0,
            pnp_inlier_ratio=0.0,
            reprojection_rmse_pixels=None,
            reprojection_median_pixels=None,
            translation_magnitude=None,
            translation_units="relative_depth_units",
        )


class SuccessfulDepthPoseEstimator:
    def estimate(self, *args) -> SimpleNamespace:
        del args
        return SimpleNamespace(
            success=True,
            message="synthetic accepted pose",
            valid_depth_correspondences=8,
            pnp_inliers=8,
            pnp_inlier_ratio=1.0,
            reprojection_rmse_pixels=0.5,
            reprojection_median_pixels=0.4,
            translation_magnitude=0.1,
            translation_units="relative_depth_units",
            rotation=np.eye(3),
            translation=np.array([0.1, 0.0, 0.0]),
            inlier_mask=np.ones(8, dtype=bool),
        )


class RecordingFeatureTracker(FakeFeatureTracker):
    def __init__(self) -> None:
        self.reference_values: list[int] = []

    def match(self, first: np.ndarray, second: np.ndarray) -> SimpleNamespace:
        self.reference_values.append(int(first[0, 0, 0]))
        return super().match(first, second)


class DisplacementFeatureTracker:
    def __init__(self, displacements: list[float]) -> None:
        self.displacements = iter(displacements)
        self.reference_values: list[int] = []

    def match(self, first: np.ndarray, second: np.ndarray) -> SimpleNamespace:
        del second
        self.reference_values.append(int(first[0, 0, 0]))
        displacement = next(self.displacements)
        points1 = np.zeros((100, 2), dtype=np.float64)
        points2 = points1.copy()
        points2[:, 0] = displacement
        return SimpleNamespace(
            points1=points1,
            points2=points2,
            statistics=SimpleNamespace(good_matches=100),
        )


class RejectThenAcceptAligner:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs) -> DepthAlignmentResult:
        del args, kwargs
        self.calls += 1
        if self.calls == 1:
            values = np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
            rejects = 3
        else:
            values = np.ones((2, 2), dtype=np.float32)
            rejects = 0
        depth = CameraDepth(
            values,
            "relative",
            False,
            "relative_camera_z_proxy",
            "synthetic",
            "relative_depth_units",
            "reciprocal_affine_disparity_alignment",
            "scale_and_shift",
            disparity_scale=1.0,
            disparity_shift=0.0,
            denominator_epsilon=0.001,
            rejected_small_denominator_count=rejects,
        )
        return DepthAlignmentResult(
            True,
            "synthetic alignment",
            depth,
            "scale_and_shift",
            1000,
            600,
            1.0,
            0.0,
            0.0,
        )


class AlwaysAcceptAligner:
    def __call__(self, *args, **kwargs) -> DepthAlignmentResult:
        del args, kwargs
        depth = CameraDepth(
            np.ones((2, 2), dtype=np.float32),
            "relative",
            False,
            "relative_camera_z_proxy",
            "synthetic",
            "relative_depth_units",
            "reciprocal_affine_disparity_alignment",
            "scale_and_shift",
            disparity_scale=1.0,
            disparity_shift=0.0,
            denominator_epsilon=0.001,
        )
        return DepthAlignmentResult(
            True, "synthetic alignment", depth, "scale_and_shift",
            1000, 600, 1.0, 0.0, 0.0,
        )


def disabled_keyframes() -> KeyframeSelector:
    return KeyframeSelector(KeyframeThresholds(enabled=False))


class MapBuilderTests(unittest.TestCase):
    @staticmethod
    def frame(index: int) -> MappingFrame:
        return MappingFrame(
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            frame_index=index,
            timestamp_seconds=index / 10.0,
        )

    @staticmethod
    def marked_frame(index: int) -> MappingFrame:
        return MappingFrame(
            image=np.full((2, 2, 3), index, dtype=np.uint8),
            frame_index=index,
            timestamp_seconds=index / 10.0,
        )

    def builder(self, depth: FakeDepthEstimator) -> RelativeMapBuilder:
        return RelativeMapBuilder(
            depth_estimator=depth,
            feature_tracker=FakeFeatureTracker(),
            motion_estimator=RejectingMotionEstimator(),
            camera_matrix=np.eye(3),
            keyframe_selector=disabled_keyframes(),
            point_cloud_stride=1,
            voxel_size=0.1,
        )

    def test_first_frame_defines_identity_world_origin(self) -> None:
        depth = FakeDepthEstimator()

        result = self.builder(depth).build([self.frame(42)])

        self.assertEqual(result.sampled_frame_count, 1)
        self.assertEqual(result.accepted_frame_count, 1)
        self.assertEqual(result.rejected_frame_count, 0)
        self.assertEqual(result.raw_fused_point_count, 4)
        self.assertEqual(depth.calls, 1)
        np.testing.assert_array_equal(result.trajectory_frame_indices, [42])
        np.testing.assert_allclose(result.trajectory_positions, [[0.0, 0.0, 0.0]])
        self.assertEqual(result.frame_statistics[0].reason, "world_origin")

    def test_enabled_depth_stabilization_adds_independent_comparison_fusion(self) -> None:
        depth = FakeDepthEstimator()
        builder = RelativeMapBuilder(
            depth_estimator=depth,
            feature_tracker=FakeFeatureTracker(),
            motion_estimator=RejectingMotionEstimator(),
            camera_matrix=np.eye(3),
            keyframe_selector=disabled_keyframes(),
            point_cloud_stride=1,
            voxel_size=0.1,
            depth_stabilization=DepthStabilizationConfig(enabled=True),
        )

        result = builder.build([self.frame(0)])

        self.assertTrue(result.depth_stabilization_enabled)
        self.assertIsNotNone(result.stabilized_fused_cloud)
        assert result.stabilized_fused_cloud is not None
        np.testing.assert_array_equal(
            result.stabilized_fused_cloud.points, result.fused_cloud.points
        )
        self.assertTrue(
            result.frame_statistics[0].depth_stabilization_accepted
        )

    def test_temporal_normalization_preserves_keyframes_and_pose_chain(self) -> None:
        def make_builder(temporal=None):
            return RelativeMapBuilder(
                depth_estimator=FakeDepthEstimator(),
                feature_tracker=FakeFeatureTracker(),
                motion_estimator=SuccessfulGeometryEstimator(),
                depth_pose_estimator=SuccessfulDepthPoseEstimator(),
                depth_aligner=AlwaysAcceptAligner(),
                camera_matrix=np.eye(3),
                keyframe_selector=disabled_keyframes(),
                point_cloud_stride=1,
                voxel_size=0.1,
                temporal_depth_normalization=temporal,
            )

        baseline = make_builder().build([self.frame(0), self.frame(1)])
        experimental = make_builder(TemporalDepthNormalizationConfig(
            enabled=True,
            minimum_correspondences=4,
            minimum_inliers=3,
        )).build([self.frame(0), self.frame(1)])

        np.testing.assert_array_equal(
            experimental.trajectory_frame_indices,
            baseline.trajectory_frame_indices,
        )
        np.testing.assert_array_equal(
            experimental.trajectory_positions, baseline.trajectory_positions
        )
        np.testing.assert_array_equal(
            experimental.fused_cloud.points, baseline.fused_cloud.points
        )
        self.assertIsNotNone(experimental.temporal_normalized_fused_cloud)

    def test_transform_audit_is_diagnostic_only_and_captures_pair_trace(self) -> None:
        def make_builder(capture: bool) -> RelativeMapBuilder:
            return RelativeMapBuilder(
                depth_estimator=FakeDepthEstimator(),
                feature_tracker=FakeFeatureTracker(),
                motion_estimator=SuccessfulGeometryEstimator(),
                depth_pose_estimator=SuccessfulDepthPoseEstimator(),
                depth_aligner=AlwaysAcceptAligner(),
                camera_matrix=np.eye(3),
                keyframe_selector=disabled_keyframes(),
                point_cloud_stride=1,
                voxel_size=0.1,
                capture_transform_audit=capture,
            )

        baseline = make_builder(False).build([self.frame(0), self.frame(1)])
        audited = make_builder(True).build([self.frame(0), self.frame(1)])

        np.testing.assert_array_equal(
            audited.trajectory_positions, baseline.trajectory_positions
        )
        np.testing.assert_array_equal(audited.fused_cloud.points, baseline.fused_cloud.points)
        assert audited.transform_trace is not None
        assert audited.transform_pair_consistency is not None
        self.assertEqual(len(audited.transform_trace), 2)
        self.assertEqual(len(audited.transform_pair_consistency), 1)
        self.assertEqual(
            audited.transform_pair_consistency[0].correspondence_count, 8
        )

    def test_disabled_sliding_window_reproduces_previous_mapping_behavior(self) -> None:
        def make_builder(config=None) -> RelativeMapBuilder:
            return RelativeMapBuilder(
                depth_estimator=FakeDepthEstimator(),
                feature_tracker=FakeFeatureTracker(),
                motion_estimator=SuccessfulGeometryEstimator(),
                depth_pose_estimator=SuccessfulDepthPoseEstimator(),
                depth_aligner=AlwaysAcceptAligner(),
                camera_matrix=np.eye(3),
                keyframe_selector=disabled_keyframes(),
                point_cloud_stride=1,
                voxel_size=0.1,
                sliding_window_pose_optimization=config,
            )

        baseline = make_builder().build([
            self.frame(0), self.frame(1), self.frame(2)
        ])
        explicitly_disabled = make_builder(
            SlidingWindowPoseOptimizationConfig(enabled=False)
        ).build([self.frame(0), self.frame(1), self.frame(2)])

        np.testing.assert_array_equal(
            explicitly_disabled.trajectory_positions,
            baseline.trajectory_positions,
        )
        np.testing.assert_array_equal(
            explicitly_disabled.fused_cloud.points, baseline.fused_cloud.points
        )
        self.assertIsNone(
            explicitly_disabled.sliding_window_pose_optimization_result
        )

    def test_enabled_sliding_window_reuses_keyframes_depths_and_cloud_inputs(self) -> None:
        depth = FakeDepthEstimator()
        builder = RelativeMapBuilder(
            depth_estimator=depth,
            feature_tracker=FakeFeatureTracker(),
            motion_estimator=SuccessfulGeometryEstimator(),
            depth_pose_estimator=SuccessfulDepthPoseEstimator(),
            depth_aligner=AlwaysAcceptAligner(),
            camera_matrix=np.eye(3),
            keyframe_selector=disabled_keyframes(),
            point_cloud_stride=1,
            voxel_size=0.1,
            sliding_window_pose_optimization=(
                SlidingWindowPoseOptimizationConfig(
                    enabled=True,
                    minimum_observations=1,
                    minimum_relative_median_improvement=0.0,
                    maximum_rotation_change_deg=10.0,
                    maximum_translation_change_relative=10.0,
                    max_nfev=100,
                )
            ),
        )

        result = builder.build([self.frame(0), self.frame(1), self.frame(2)])

        self.assertTrue(result.sliding_window_pose_optimization_enabled)
        self.assertIsNotNone(result.sliding_window_pose_optimization_result)
        self.assertIsNotNone(result.pose_optimized_fused_cloud)
        np.testing.assert_array_equal(result.trajectory_frame_indices, [0, 1, 2])
        self.assertEqual(depth.calls, 3)
        self.assertEqual(
            result.pose_optimized_raw_fused_point_count,
            result.raw_fused_point_count,
        )
        assert result.sliding_window_pose_optimization_result is not None
        self.assertEqual(
            result.sliding_window_pose_optimization_result.window_frame_indices,
            (0, 1, 2),
        )

    def test_rejected_motion_adds_neither_pose_nor_cloud(self) -> None:
        depth = FakeDepthEstimator()

        result = self.builder(depth).build([self.frame(0), self.frame(10)])

        self.assertEqual(result.sampled_frame_count, 2)
        self.assertEqual(result.accepted_frame_count, 1)
        self.assertEqual(result.rejected_frame_count, 1)
        self.assertEqual(result.raw_fused_point_count, 4)
        self.assertEqual(depth.calls, 1)
        np.testing.assert_array_equal(result.trajectory_frame_indices, [0])
        np.testing.assert_allclose(result.trajectory_positions, [[0.0, 0.0, 0.0]])
        self.assertIn("synthetic rejection", result.frame_statistics[1].reason)

    def test_failed_scaled_pose_adds_neither_pose_nor_cloud(self) -> None:
        depth = FakeDepthEstimator()
        builder = RelativeMapBuilder(
            depth_estimator=depth,
            feature_tracker=FakeFeatureTracker(),
            motion_estimator=SuccessfulGeometryEstimator(),
            depth_pose_estimator=RejectingDepthPoseEstimator(),
            camera_matrix=np.eye(3),
            keyframe_selector=disabled_keyframes(),
            point_cloud_stride=1,
            voxel_size=0.1,
        )

        result = builder.build([self.frame(0), self.frame(10)])

        self.assertEqual(result.accepted_frame_count, 1)
        self.assertEqual(result.rejected_frame_count, 1)
        self.assertEqual(result.trajectory_positions.shape, (1, 3))
        self.assertEqual(depth.calls, 1)
        self.assertIn("scaled-pose rejection", result.frame_statistics[1].reason)

    def test_fixed_step_mode_is_marked_as_debug(self) -> None:
        depth = FakeDepthEstimator()
        builder = RelativeMapBuilder(
            depth_estimator=depth,
            feature_tracker=FakeFeatureTracker(),
            motion_estimator=SuccessfulGeometryEstimator(),
            camera_matrix=np.eye(3),
            keyframe_selector=disabled_keyframes(),
            scale_mode="fixed-step",
            translation_step=0.5,
            point_cloud_stride=1,
            voxel_size=0.1,
        )

        result = builder.build([self.frame(0), self.frame(10)])

        statistics = result.frame_statistics[1]
        self.assertTrue(statistics.accepted)
        self.assertEqual(statistics.reason, "accepted_debug_fixed_step")
        self.assertEqual(statistics.scale_estimation_method, "fixed_step_debug")
        self.assertEqual(statistics.translation_units, "arbitrary_fixed_step_units")

    def test_depth_quality_rejection_preserves_last_accepted_reference(self) -> None:
        depth = FakeDepthEstimator()
        tracker = RecordingFeatureTracker()
        aligner = RejectThenAcceptAligner()
        builder = RelativeMapBuilder(
            depth_estimator=depth,
            feature_tracker=tracker,
            motion_estimator=SuccessfulGeometryEstimator(),
            depth_pose_estimator=SuccessfulDepthPoseEstimator(),
            depth_aligner=aligner,
            camera_matrix=np.eye(3),
            keyframe_selector=disabled_keyframes(),
            point_cloud_stride=1,
            voxel_size=0.1,
        )

        result = builder.build([
            self.marked_frame(0),
            self.marked_frame(1),
            self.marked_frame(2),
        ])

        self.assertEqual(result.accepted_frame_count, 2)
        self.assertEqual(result.skipped_non_keyframe_count, 0)
        self.assertEqual(result.rejected_frame_count, 1)
        self.assertEqual(result.raw_fused_point_count, 8)
        np.testing.assert_array_equal(result.trajectory_frame_indices, [0, 2])
        self.assertEqual(tracker.reference_values, [0, 0])
        self.assertEqual(
            result.frame_statistics[1].rejection_reason,
            "depth_denominator_reject_ratio",
        )
        self.assertEqual(
            result.frame_statistics[1].to_dict()["rejection_reason"],
            "depth_denominator_reject_ratio",
        )
        self.assertEqual(result.frame_statistics[1].status, "rejected")
        self.assertTrue(result.frame_statistics[2].accepted)

    def test_skipped_frame_avoids_depth_and_preserves_keyframe_reference(self) -> None:
        depth = FakeDepthEstimator()
        tracker = DisplacementFeatureTracker([1.0, 10.0])
        builder = RelativeMapBuilder(
            depth_estimator=depth,
            feature_tracker=tracker,
            motion_estimator=SuccessfulGeometryEstimator(),
            depth_pose_estimator=SuccessfulDepthPoseEstimator(),
            depth_aligner=AlwaysAcceptAligner(),
            camera_matrix=np.eye(3),
            point_cloud_stride=1,
            voxel_size=0.1,
        )

        result = builder.build([
            self.marked_frame(0),
            self.marked_frame(1),
            self.marked_frame(2),
        ])

        self.assertEqual(result.accepted_frame_count, 2)
        self.assertEqual(result.skipped_non_keyframe_count, 1)
        self.assertEqual(result.rejected_frame_count, 0)
        self.assertEqual(result.depth_inference_count, 2)
        self.assertEqual(depth.calls, 2)
        self.assertEqual(tracker.reference_values, [0, 0])
        self.assertEqual(result.raw_fused_point_count, 8)
        np.testing.assert_array_equal(result.trajectory_frame_indices, [0, 2])
        skipped = result.frame_statistics[1]
        accepted = result.frame_statistics[2]
        self.assertEqual(skipped.status, "skipped_non_keyframe")
        self.assertEqual(skipped.skip_reason, "insufficient_keyframe_motion")
        self.assertEqual(
            skipped.to_dict()["status"], "skipped_non_keyframe"
        )
        self.assertFalse(skipped.depth_inference_executed)
        self.assertEqual(accepted.status, "accepted_keyframe")
        self.assertEqual(
            accepted.keyframe_reason, "sufficient_feature_displacement"
        )
        self.assertTrue(accepted.depth_inference_executed)
        self.assertEqual(accepted.pnp_inliers, 8)

    def test_disabled_keyframes_reproduces_fixed_sample_attempt(self) -> None:
        depth = FakeDepthEstimator()
        tracker = DisplacementFeatureTracker([0.0])
        builder = RelativeMapBuilder(
            depth_estimator=depth,
            feature_tracker=tracker,
            motion_estimator=SuccessfulGeometryEstimator(),
            depth_pose_estimator=SuccessfulDepthPoseEstimator(),
            depth_aligner=AlwaysAcceptAligner(),
            keyframe_selector=disabled_keyframes(),
            camera_matrix=np.eye(3),
            point_cloud_stride=1,
            voxel_size=0.1,
        )

        result = builder.build([self.marked_frame(0), self.marked_frame(1)])

        self.assertEqual(result.accepted_frame_count, 2)
        self.assertEqual(result.skipped_non_keyframe_count, 0)
        self.assertEqual(result.depth_inference_count, 2)
        self.assertEqual(
            result.frame_statistics[1].keyframe_reason,
            "keyframe_selection_disabled",
        )


if __name__ == "__main__":
    unittest.main()
