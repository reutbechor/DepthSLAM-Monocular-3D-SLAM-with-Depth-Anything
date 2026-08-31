import unittest
from types import SimpleNamespace

import numpy as np

from src.map_builder import MappingFrame, RelativeMapBuilder
from src.depth_types import DepthPrediction


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


class MapBuilderTests(unittest.TestCase):
    @staticmethod
    def frame(index: int) -> MappingFrame:
        return MappingFrame(
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            frame_index=index,
            timestamp_seconds=index / 10.0,
        )

    def builder(self, depth: FakeDepthEstimator) -> RelativeMapBuilder:
        return RelativeMapBuilder(
            depth_estimator=depth,
            feature_tracker=FakeFeatureTracker(),
            motion_estimator=RejectingMotionEstimator(),
            camera_matrix=np.eye(3),
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


if __name__ == "__main__":
    unittest.main()
