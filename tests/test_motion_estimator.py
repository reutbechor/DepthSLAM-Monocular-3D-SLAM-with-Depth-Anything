import unittest

import numpy as np

from src.motion_estimator import MotionEstimator


class MotionEstimatorTests(unittest.TestCase):
    def test_too_few_correspondences_are_rejected(self) -> None:
        points = np.zeros((7, 2), dtype=np.float64)
        result = MotionEstimator(minimum_correspondences=8).estimate(
            points, points.copy(), np.eye(3)
        )

        self.assertFalse(result.success)
        self.assertIn("at least 8", result.message)
        self.assertEqual(result.inlier_mask.shape, (7,))

    def test_synthetic_calibrated_pose_has_expected_shapes(self) -> None:
        rng = np.random.default_rng(7)
        camera_matrix = np.array(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
        )
        points_3d = np.column_stack(
            (
                rng.uniform(-1.5, 1.5, 80),
                rng.uniform(-1.0, 1.0, 80),
                rng.uniform(4.0, 8.0, 80),
            )
        )
        angle = np.deg2rad(5.0)
        rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        translation = np.array([0.4, 0.02, 0.1])

        def project(points: np.ndarray) -> np.ndarray:
            pixels = (camera_matrix @ points.T).T
            return pixels[:, :2] / pixels[:, 2:3]

        points1 = project(points_3d)
        points2 = project((rotation @ points_3d.T).T + translation)
        result = MotionEstimator(
            ransac_threshold_pixels=0.5,
            minimum_inliers=20,
            minimum_inlier_ratio=0.5,
        ).estimate(points1, points2, camera_matrix)

        self.assertTrue(result.success, result.message)
        self.assertEqual(result.essential_matrix.shape, (3, 3))
        self.assertEqual(result.rotation.shape, (3, 3))
        self.assertEqual(result.translation_direction.shape, (3, 1))
        self.assertEqual(result.inlier_mask.shape, (80,))
        self.assertEqual(result.rotation.dtype, np.float64)
        self.assertEqual(result.translation_direction.dtype, np.float64)
        self.assertAlmostEqual(float(np.linalg.norm(result.translation_direction)), 1.0)
        # points1 is previous and points2 is current, so recoverPose must return
        # T_current_from_previous rather than its inverse.
        np.testing.assert_allclose(result.rotation, rotation, atol=1e-5)
        expected_direction = translation / np.linalg.norm(translation)
        np.testing.assert_allclose(
            result.translation_direction.reshape(3), expected_direction, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
