import unittest

import cv2
import numpy as np

from src.backprojection import backproject_pixels
from src.depth_pose_estimator import DepthPoseEstimator
from src.depth_types import CameraDepth


class DepthPoseEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        u = np.array([100, 160, 220, 280, 340, 400, 460, 520] * 2, dtype=float)
        v = np.array([120] * 8 + [340] * 8, dtype=float)
        self.points1 = np.column_stack((u, v))
        self.depths = np.array(
            [3.0, 4.2, 5.1, 3.7, 6.0, 4.8, 5.5, 3.4,
             5.8, 3.3, 4.5, 6.2, 3.9, 5.3, 4.0, 6.5]
        )
        self.objects = backproject_pixels(
            self.points1, self.depths, self.camera_matrix
        )
        rvec = np.array([0.025, -0.035, 0.015])
        self.rotation, _ = cv2.Rodrigues(rvec)
        self.translation = np.array([0.35, -0.12, 0.22])
        projected, _ = cv2.projectPoints(
            self.objects, rvec, self.translation, self.camera_matrix, None
        )
        self.points2 = projected.reshape(-1, 2)
        values = np.full((480, 640), np.nan, dtype=np.float32)
        values[v.astype(int), u.astype(int)] = self.depths
        self.camera_depth = CameraDepth(
            values, "relative", False, "relative_camera_z_proxy", "synthetic",
            "relative_depth_units", "synthetic", "none",
        )

    def test_recovers_rotation_and_translation_magnitude(self) -> None:
        result = DepthPoseEstimator(
            sampling_method="nearest",
            reprojection_error_pixels=1.0,
            minimum_inliers=8,
        ).estimate(
            self.points1,
            self.points2,
            np.ones(self.points1.shape[0], dtype=bool),
            self.camera_depth,
            self.camera_matrix,
        )

        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.rotation, self.rotation, atol=1e-5)
        np.testing.assert_allclose(result.translation, self.translation, atol=1e-5)
        self.assertAlmostEqual(
            result.translation_magnitude,
            float(np.linalg.norm(self.translation)),
            places=5,
        )
        self.assertEqual(result.pnp_inliers, self.points1.shape[0])
        self.assertLess(result.reprojection_rmse_pixels, 1e-4)

    def test_insufficient_valid_depth_rejects_pose(self) -> None:
        sparse = self.camera_depth.values.copy()
        sparse[:] = np.nan
        for point, depth in zip(self.points1[:3], self.depths[:3]):
            sparse[int(point[1]), int(point[0])] = depth
        depth = CameraDepth(
            sparse, "relative", False, "relative_camera_z_proxy", "synthetic",
            "relative_depth_units", "synthetic", "none",
        )

        result = DepthPoseEstimator(sampling_method="nearest").estimate(
            self.points1,
            self.points2,
            np.ones(self.points1.shape[0], dtype=bool),
            depth,
            self.camera_matrix,
        )

        self.assertFalse(result.success)
        self.assertIn("valid depth correspondences", result.message)


if __name__ == "__main__":
    unittest.main()
